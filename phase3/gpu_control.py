"""GPU Control Center for the existing Windows desktop broker."""
import json
import math
import secrets
import shutil
import subprocess
import threading
import time
from pathlib import Path
from queue import Empty, Queue
from tempfile import TemporaryDirectory

import gpu_credentials
from gpu_setup import (HOST_KEY_COMMAND, HostKeyUnavailable, authenticated_host_key, endpoint, parse_host_key,
                       pin_host, stage_worker)
from live_guard import (A100_SXM_80GB, DEFAULT_SETUP, EMERGENCY, LIMIT, SHUTDOWN_MARGIN, START_WAIT, STOPPED, Approval,
                        PodProfile, PreflightError, ProviderError, api, compatible, compute_rate, container_storage_rate, list_pods,
                        other_active_pod, pending_record, pending_session, pod_id, reconcile_pending, setup_deadline, state_directory,
                        total_rate)
from live_manager import LiveManager


def read_profiles():
    path = Path(__file__).with_name("gpu_profiles.json")
    profiles = {}
    names = set()
    for entry in json.loads(path.read_text(encoding="utf-8")):
        name, aliases, cap = entry["gpu"], entry["aliases"], entry["rate_cap"]
        if (not isinstance(name, str) or not name or name in profiles or not isinstance(aliases, list)
                or not aliases or not all(isinstance(alias, str) and alias and alias not in names for alias in aliases)
                or (cap is not None and (type(cap) not in (int, float) or not math.isfinite(cap) or cap <= 0))):
            raise ValueError("Invalid GPU profile")
        profiles[name] = (tuple(aliases), cap)
        names.update(aliases)
    if not profiles:
        raise ValueError("Missing GPU profiles")
    return profiles


class GPUControlCenter(LiveManager):
    terminal = {"idle", "error", "stopped", "shutdown_unverified"}

    def __init__(self, setup_seconds=DEFAULT_SETUP):
        super().__init__(None, initialize=False)
        self.setup_seconds = setup_seconds
        self.candidates = {}
        self.active_pod = None
        self.discovery_detail = ""
        self.start_deadline = None
        self.approval = None
        self.remote_armed = False
        self.pin_request = None
        self.fingerprint = None
        self.saved = False
        try:
            self.profiles = read_profiles()
        except (OSError, ValueError, KeyError, TypeError):
            self.profiles = {}
            self.detail = "GPU configuration could not be loaded. Check gpu_profiles.json"
        try:
            self.key = gpu_credentials.load_key()
            self.saved = bool(self.key)
        except (OSError, ValueError):
            self.detail = "Windows Credential Manager could not load the saved RunPod key"

    def working(self):
        return self.state not in self.terminal or (self.process is not None and self.process.is_alive())

    def pending(self):
        try:
            return pending_session(state_directory())
        except (OSError, ValueError, TypeError, KeyError):
            return ("unknown", None, None)

    def credentials(self, body):
        with self.lock:
            if self.working():
                raise PreflightError("Wait for the current GPU session to finish")
            if body.get("forget") is True:
                if self.pending():
                    raise PreflightError("Resolve GPU shutdown before removing its credential")
                gpu_credentials.forget_key()
                self.key, self.saved = None, False
            else:
                key = body.get("key")
                if not isinstance(key, str) or not 1 <= len(key) <= 512 or not all("!" <= c <= "~" for c in key):
                    raise PreflightError("Enter a valid scoped RunPod API key")
                if body.get("remember") is True:
                    gpu_credentials.save_key(key)
                    self.saved = True
                elif self.saved:
                    gpu_credentials.forget_key()
                    self.saved = False
                self.key = key
            self.candidates.clear()
            self.active_pod = None
            self.discovery_detail = ""

    def discover(self, body):
        with self.lock:
            if self.working():
                raise PreflightError("Wait for the current GPU session to finish")
            key = self.key
        if not key:
            raise PreflightError("Enter your scoped RunPod key first")
        if not self.profiles:
            raise PreflightError("GPU configuration could not be loaded. Check gpu_profiles.json")
        requested = body.get("pod")
        try:
            rows = [api(key, pod=pod_id(requested))] if requested else list(list_pods(key))
            active_pod = other_active_pod(rows, None) if not requested else None
            found = {}
            rejected = set()
            for info in rows:
                try:
                    candidate = self.candidate(info)
                except PreflightError as error:
                    if requested:
                        raise
                    rejected.add(str(error))
                    continue
                found[candidate["pod"]] = candidate
        except ProviderError as error:
            if error.code in {401, 403}:
                raise PreflightError(f"RunPod denied Pod discovery (HTTP {error.code}). Check that the key is active and allows Pod read") from None
            raise
        except OSError:
            raise PreflightError("RunPod discovery failed. Check connectivity and the key scope") from None
        with self.lock:
            if key != self.key or self.working():
                raise PreflightError("GPU session changed during discovery")
            if not requested:
                self.candidates.clear()
                self.active_pod = active_pod
                self.discovery_detail = ""
                if not rows:
                    self.discovery_detail = "RunPod returned no Pods. Check the key’s workspace or look up an existing Pod by ID."
                elif not found:
                    reason = ". ".join(sorted(rejected))
                    self.discovery_detail = f"No compatible Pods. {reason}. Check the Pod configuration in RunPod."
            elif active_pod is None and self.active_pod == requested and found[requested]["status"] != "RUNNING":
                self.active_pod = None
            self.candidates.update(found)

    def candidate(self, info):
        pod = pod_id(info.get("id") if isinstance(info, dict) else None)
        hardware = info.get("gpu")
        gpu = next((name for name, (aliases, _) in self.profiles.items()
                    if isinstance(hardware, dict) and hardware.get("id") in aliases), None)
        if not gpu:
            raise PreflightError("Pod GPU does not match a configured profile")
        aliases, cap = self.profiles[gpu]
        profile = PodProfile(pod, gpu, cap if cap is not None else math.inf, aliases=aliases)
        compatible(info, profile)
        status = info.get("status")
        if status not in {"RUNNING", "EXITED", "STOPPED"}:
            raise PreflightError("Pod is neither running nor available for restart")
        rate, storage = None, None
        issue = ""
        try:
            try:
                rate = compute_rate(info)
            except PreflightError:
                if status in STOPPED:
                    raise PreflightError("RunPod did not report a positive restart quote. Check this Pod’s Start offer in RunPod, then refresh") from None
                raise
            storage = container_storage_rate(info)
            if not isinstance(info.get("image"), str) or not info["image"]:
                raise PreflightError("RunPod did not report the container image")
            Approval(profile, status, info.get("startedAt"), rate, storage, self.setup_seconds, disk=info["disk"]).validate(info)
        except PreflightError as error:
            issue = str(error)
        return dict(pod=pod, gpu=gpu, status=status, rate=rate, storage=storage, disk=info.get("disk"), cap=cap, issue=issue,
                    generation=info.get("startedAt") if isinstance(info.get("startedAt"), str) else None,
                    image=info.get("image") if isinstance(info.get("image"), str) else None)

    def begin(self, owner, body):
        with self.lock:
            if self.working() or self.pending():
                raise PreflightError("A GPU session or unverified shutdown is already in progress")
            if not self.key or not owner or body.get("confirmed") is not True:
                raise PreflightError("Confirm the exact Pod and total hourly rate")
            target = self.candidates.get(pod_id(body.get("pod")))
            if not target or target["issue"]:
                raise PreflightError("Refresh and select an eligible Pod with a verified compute rate")
            if target["gpu"] in (A100_SXM_80GB, 'NVIDIA H100 80GB HBM3') and target["status"] != "RUNNING":
                raise PreflightError("Start the selected GPU manually in RunPod, then refresh and attach")
            spending_cap = body.get("ceiling")
            if (type(spending_cap) not in (float, int) or not 0 < spending_cap or
                    not math.isfinite(spending_cap)):
                raise PreflightError("Enter and confirm a positive spending ceiling")
            storage = body.get("storage")
            if type(storage) not in (float, int) or not 0 < storage < spending_cap or not math.isfinite(storage):
                raise PreflightError("Enter the current positive hourly storage rate")
            if storage < target["storage"]:
                raise PreflightError("Storage allowance is below the published container disk rate")
            aliases, profile_cap = self.profiles[target["gpu"]]
            cap = min(spending_cap, profile_cap) if profile_cap is not None else spending_cap
            profile = PodProfile(target["pod"], target["gpu"], cap, aliases=aliases)
            approval = Approval(profile, target["status"], target["generation"], target["rate"], storage,
                                self.setup_seconds, target["image"], target["disk"])
            if (body.get("status") != approval.status or body.get("generation") != approval.generation
                    or body.get("compute") != approval.compute or body.get("total") != float(f"{approval.total:.15g}")):
                raise PreflightError("Pod or quote changed. Refresh and confirm again")
            if approval.total > profile.rate_cap:
                raise PreflightError(f"Total hourly rate exceeds the ${profile.rate_cap:.2f} ceiling")
            if not self.watchdog_available:
                raise PreflightError("Install Windows Python with pythonw.exe to arm the independent watchdog")
            if not shutil.which("ssh") or not shutil.which("scp"):
                raise PreflightError("Install the Windows OpenSSH Client before starting a GPU session")
            self.profile, self.approval = profile, approval
            self.runtime = {}
            self.remote_armed, self.pin_request, self.fingerprint = False, None, None
            self.info = {"status": approval.status, "cost": approval.compute}
            self.launch_guard(owner, None if approval.starting else approval.generation, approval)
            self.detail = "Checking Pod and arming the Windows watchdog"

    def status(self, owner):
        with self.lock:
            base = super().status(None)
            now = time.time()
            guard_alive = bool(self.process and self.process.is_alive())
            ready = self.state == "ready" and guard_alive and not self.stop_request.is_set()
            if ready and owner and owner == self.owner:
                self.heartbeat.value = now
            working = self.working()
            pending = None if working else self.pending()
            pending_rejected = False
            if pending and pending[0] != "unknown":
                try:
                    pending_rejected = (pending_record(state_directory()) or {}).get("rejected") is True
                except (OSError, ValueError, TypeError, KeyError):
                    pass
            if pending and self.state != "shutdown_unverified":
                base.update(state="shutdown_unverified", detail="A previous shutdown still needs verification")
            elif owner and working and self.owner and owner != self.owner and self.state in {"host_key", "ready"}:
                base["detail"] = "Return to the tab that started this session, or use Stop GPU"
            base.update(ready=ready and owner == self.owner, available=not working and not pending,
                        runtime=getattr(self, "runtime", {}),
                        can_stop=working or bool(pending), key_configured=bool(self.key), key_saved=self.saved,
                        candidates=list(self.candidates.values()), active_pod=self.active_pod,
                        discovery_detail=self.discovery_detail,
                        windows_guard=guard_alive and self.state != "arming",
                        remote_guard=ready or (self.remote_armed and self.state == "loading"),
                        pending_pod=pending[0] if pending else None,
                        pending_unknown=bool(pending and not pending[1] and not pending_rejected),
                        start_remaining=max(0, math.ceil(self.start_deadline - now))
                        if self.state == "watching_delayed_start" and self.start_deadline is not None else None,
                        pending_remaining=max(0, math.ceil(pending[2] + START_WAIT - now))
                        if pending and not pending[1] and pending[2] and not pending_rejected else None,
                        cancel_requested=self.state == "watching_delayed_start" and self.stop_request.is_set(),
                        setup_seconds=self.setup_seconds,
                        setup_remaining=max(0, math.ceil(setup_deadline(self.began, self.setup_seconds) - now))
                        if self.began and not self.ended and self.state not in self.terminal | {"ready", "stopping"}
                        else None,
                        fingerprint=self.fingerprint, pin_command=self.pin_request[0]
                        if self.state == "host_key" and owner == self.owner and self.pin_request else None,
                        approved_total=self.approval.total if self.approval else None, busy=working,
                        current_total=self.bill_rate if self.info.get("status") == "RUNNING" and self.bill_rate else None,
                        shutdown_at=self.began + LIMIT - SHUTDOWN_MARGIN if self.began and not self.ended else None)
            if not self.approval:
                base.update(pod=None, gpu=None)
            return base

    def record_pod_status(self, fields):
        super().record_pod_status(fields)
        if self.approval and "actual_rate" in fields:
            self.bill_rate = total_rate(fields["actual_rate"], self.approval.storage)
        if fields["pod_status"] in STOPPED and self.profile.pod in self.candidates:
            # A new confirmation requires a refreshed generation and quote.
            self.candidates.pop(self.profile.pod)
        if fields["pod_status"] in STOPPED and self.active_pod == self.profile.pod:
            self.active_pod = None

    def on_setup(self, fields):
        self.state, self.detail = "loading", "Waiting for RunPod to publish direct SSH"
        threading.Thread(target=self.prepare, args=(self.approval, fields["generation"], self.stop_request),
                         daemon=True).start()

    def submission(self, owner, body):
        with self.lock:
            if (owner != self.owner or self.state != "host_key" or self.stop_request.is_set()
                    or body.get("trusted") is not True or not self.pin_request):
                raise PreflightError("No host key confirmation is open for this session")
            output = body.get("output")
            if not isinstance(output, str) or len(output) > 4096:
                raise PreflightError("Paste the three key lines and session line from RunPod")
            _, challenge, inbox = self.pin_request
            result = parse_host_key(output, self.profile.pod, challenge)
            self.state, self.detail = "loading", "Host key confirmed from the RunPod web terminal"
            self.pin_request = None
            inbox.put_nowait(result)

    def prepare(self, approval, generation, stop):
        def remaining():
            if stop.is_set() or self.stop_request is not stop:
                raise PreflightError("GPU setup was stopped")
            seconds = setup_deadline(self.began, approval.setup_seconds) - time.time()
            if seconds <= 0:
                raise PreflightError("Setup deadline before worker readiness")
            return seconds

        def progress(detail):
            with self.lock:
                remaining()
                if self.state == "loading":
                    self.detail = detail

        def lookup():
            remaining()
            info = api(self.key, pod=approval.profile.pod)
            approval.running(info, generation, setup=True)
            return endpoint(info)

        try:
            direct = lookup()
            while not direct:
                stop.wait(1)
                direct = lookup()
            progress("Checking the SSH host key through RunPod")
            try:
                encoded, fingerprint = authenticated_host_key(self.key, approval.profile.pod)
            except HostKeyUnavailable:
                challenge, inbox = secrets.token_hex(6), Queue(1)
                command = f"printf 'SignToCap session {challenge}\\n'; " + HOST_KEY_COMMAND
                with self.lock:
                    remaining()
                    self.pin_request = command, challenge, inbox
                    self.state = "host_key"
                    self.detail = "Confirm the host key from the selected Pod’s RunPod web terminal"
                while True:
                    try:
                        encoded, fingerprint = inbox.get(timeout=min(1, remaining()))
                        break
                    except Empty:
                        pass
            if lookup() != direct:
                raise PreflightError("The direct SSH endpoint changed during host verification")
            with TemporaryDirectory(prefix="SignToCap-GPU-") as temporary:
                directory = Path(temporary)
                pin = pin_host(directory, direct, encoded)
                with self.lock:
                    remaining()
                    self.fingerprint = fingerprint
                command = stage_worker(directory, direct, pin, approval.profile.pod, secrets.token_hex(12),
                                       remaining, progress)
                if lookup() != direct:
                    raise PreflightError("The direct SSH endpoint changed during staging")
                self.load_worker(command, approval, generation, stop, remaining)
        except Exception as error:
            reason = str(error) if isinstance(error, PreflightError) else "GPU setup failed. Check the Pod runtime and SSH."
            with self.lock:
                if self.stop_request is stop and not stop.is_set() and self.state in {"loading", "host_key"}:
                    self.stop(reason)

    def load_worker(self, command, approval, generation, stop, remaining):
        with self.lock:
            remaining()
            self.detail = "Starting the private worker and Pod watchdog"
            proc = self.ssh = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                               stderr=subprocess.DEVNULL, text=True, encoding="utf-8", bufsize=1,
                                               creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        if proc.stdout.readline().strip() != "pod-verified":
            code = proc.wait(timeout=2)
            details = {42: "Worker SSH session reported a different Pod ID",
                       43: "The approved /workspace volume is not mounted",
                       45: "The approved volume is missing its bootstrapped Python environment"}
            raise PreflightError(details.get(code, "Could not launch the verified worker over pinned SSH"))
        remaining()
        profile = approval.profile
        proc.stdin.write(json.dumps(dict(key=self.key, pod=profile.pod, gpu=profile.gpu, generation=generation,
                                         cap=profile.rate_cap, compute=approval.compute, storage=approval.storage,
                                         setup_seconds=self.setup_seconds, aliases=profile.gpu_names,
                                         image=approval.image, disk=approval.disk)) + "\n")
        proc.stdin.flush()
        first = self.worker_message(proc)
        if first.get("guard") is not True:
            raise PreflightError("Pod-side watchdog did not arm")
        with self.lock:
            remaining()
            self.remote_armed = True
            self.detail = "Both watchdogs armed. Checking assets and loading Uni-Sign"
        response = self.worker_message(proc)
        runtime = response.pop("runtime", {})
        if response != {"ready": True, "generation": generation, "guard": True} or not isinstance(runtime, dict):
            raise PreflightError("Worker readiness or generation did not match")
        with self.lock:
            remaining()
            if self.ssh is proc and self.state == "loading" and not stop.is_set():
                self.heartbeat.value = time.time()
                self.runtime = runtime
                self.state, self.detail = "ready", "Ready for the 60-second live test"

    @staticmethod
    def worker_message(proc):
        try:
            response = json.loads(proc.stdout.readline())
            if not isinstance(response, dict):
                raise ValueError
        except ValueError:
            raise PreflightError("Worker exited before readiness. Check its Python/CUDA runtime") from None
        errors = {"generation": "Worker could not verify the exact Pod generation, GPU, volume and rate",
                  "model": "Model readiness failed. Check the staged Uni-Sign and pose assets, CUDA and Python"}
        if response.get("error") in errors:
            if response.get("error") == "model" and response.get("diagnostic"):
                import os
                from pathlib import Path
                try:
                    folder = Path(os.environ["LOCALAPPDATA"]) / "SignToCap"
                    folder.mkdir(parents=True, exist_ok=True)
                    (folder / "model-readiness.log").write_text(
                        json.dumps(response["diagnostic"], indent=2),
                        encoding="utf-8"
                    )
                except OSError:
                    pass
            raise PreflightError(errors[response["error"]])
        return response

    def stop(self, reason=EMERGENCY, *, manual=False):
        with self.lock:
            if self.process and self.process.is_alive() and self.state not in self.terminal:
                if not self.stop_request.is_set():
                    self.stop_reason.value = reason.encode()[:159]
                    self.stop_request.set()
                    waiting = self.state in {"arming", "starting", "booting", "watching_delayed_start"}
                    already_watching = self.state == "watching_delayed_start"
                    self.state = "watching_delayed_start" if waiting else "stopping"
                    if not already_watching:
                        self.detail = reason
                    self.close_ssh()
            elif not self.working() and self.pending() and self.key:
                self.state, self.detail = "stopping", "Reconciling the previous shutdown"
                threading.Thread(target=self.recover, args=(manual,), daemon=True).start()
            elif self.state != "stopped":
                raise PreflightError("Open the RunPod console, stop the selected Pod and confirm it shows Exited")

    def recover(self, manual):
        try:
            verified = reconcile_pending(self.key, manual)
        except (ValueError, OSError, RuntimeError, KeyError, TypeError):
            verified = False
        with self.lock:
            self.detail = ("RunPod confirmed shutdown" if verified else
                           "Check the selected Pod in the RunPod console and confirm it is Exited")
            self.finish_shutdown(verified, time.time())
