"""Windows-side broker. The browser never receives an API key or an SSH endpoint."""
import ipaddress
import json
import multiprocessing as mp
import os
from pathlib import Path
import subprocess
import sys
import threading
import time
from datetime import datetime, timezone

from live_guard import (EMERGENCY, GRACE, L4_PROFILE, LIMIT, SETUP, STOPPED, TEST, PreflightError, api,
                        attach_preflight, check_setup_time, guard, preflight, record_shutdown, verified_stop)


class LiveManager:
    def __init__(self, key, known_hosts=None, profile=L4_PROFILE, *, initialize=True):
        self.key = key
        self.known_hosts = known_hosts
        self.profile = profile
        self.owner = None
        self.generation = None
        self.state = "idle"
        self.detail = ""
        self.lock = threading.RLock()
        self.io = threading.Lock()
        self.process = self.ssh = None
        self.began = self.ended = self.rate = 0
        self.bill_started = self.bill_ended = self.bill_rate = 0
        try:
            self.info = api(key, pod=profile.pod) if initialize else {}
            if profile.started_at:
                attach_preflight(self.info, profile.rate_cap, profile=profile, setup=True)
        except PreflightError:
            raise
        except (OSError, ValueError):
            raise ValueError("Initial RunPod status check failed") from None
        self.last_info = time.monotonic()
        self.live = False
        self.request_since = None
        self.watchdog_available = True
        if os.name == "nt":
            windowless = Path(sys.executable).with_name("pythonw.exe")
            if not windowless.is_file():
                self.watchdog_available = False
                if initialize:
                    raise ValueError("Windows Python with pythonw.exe is required for the independent watchdog")
            else:
                mp.set_executable(str(windowless))

    def status(self, owner):
        if owner and self.state in {"idle","error","stopped","shutdown_unverified"}:
            with self.lock:
                if time.monotonic() - self.last_info >= 15:
                    self.last_info = time.monotonic()
                    threading.Thread(target=self.refresh_info,daemon=True).start()
        with self.lock:
            if owner and owner == self.owner and self.state == "ready" and not self.stop_request.is_set():
                self.heartbeat.value = time.time()
        now = time.time()
        seconds = max(0, (self.ended or now) - self.began) if self.began else 0
        bill_seconds = max(0, (self.bill_ended or now) - self.bill_started) if self.bill_started else 0
        guard_alive = self.process is not None and self.process.is_alive()
        return dict(state=self.state, detail=self.detail, gpu=self.profile.gpu, pod=self.profile.pod,
                    pod_status=self.info.get("status"), rate=self.rate or self.info.get("cost"),
                    current_rate=self.info.get("cost"), rate_cap=self.profile.rate_cap,
                    began=self.began, seconds=round(seconds), bill_seconds=round(bill_seconds),
                    cost=bill_seconds * self.bill_rate / 3600,
                    remaining=max(0, int(LIMIT-seconds)) if self.began and not self.ended else 0,
                    test_remaining=max(0,int(TEST-(now-self.test_since.value))) if self.live else None,
                    ready=self.state == "ready" and owner == self.owner, active=self.live,
                    can_stop=not self.profile.started_at or bool(self.began and self.state != "stopped"),
                    available=(self.state == "setup" and guard_alive) or (
                        not self.profile.started_at and self.state in {"idle","stopped","error"} and not guard_alive))

    def refresh_info(self):
        try:
            info = api(self.key, pod=self.profile.pod)
        except (OSError,ValueError):
            info = dict(self.info,status="UNKNOWN",cost=None)
        with self.lock:
            if self.state in {"idle","error","stopped","shutdown_unverified"}:
                self.info = info

    def begin(self, owner, body):
        with self.lock:
            if (not owner or body.get("confirmed") is not True or body.get("pod") != self.profile.pod
                    or body.get("attach") is not True):
                raise ValueError("Confirm attachment to the approved running Pod")
            if not self.status(None)["available"]:
                raise ValueError("A session or shutdown is already in progress")
            try:
                self.info = api(self.key, pod=self.profile.pod)
            except PreflightError:
                raise
            except (OSError, ValueError):
                raise ValueError("RunPod status check failed") from None
            armed = self.state == "setup"
            if armed and self.stop_request.is_set():
                raise ValueError("Shutdown already requested")
            generation, _, _ = attach_preflight(self.info, self.profile.rate_cap, self.generation if armed else None,
                                                profile=self.profile, setup=True)
            fields = {"generation": generation, "ssh": (self.info.get("ssh") or {}).get("direct")}
            command = self.ssh_command(fields["ssh"])
            if armed:
                self.owner = owner
                self.start_connection(fields, command)
            else:
                self.launch_guard(owner, generation)

    def arm(self):
        if self.profile.started_at:
            generation, _, _ = attach_preflight(self.info, self.profile.rate_cap, profile=self.profile, setup=True)
        else:
            preflight(self.info, self.profile.rate_cap)
            generation = None
        self.launch_guard(None, generation)

    def launch_guard(self, owner, generation, approval=None):
        self.owner, self.state, self.detail = owner, "arming", ""
        self.generation = generation
        self.start_deadline = None
        self.began = self.ended = 0
        self.bill_started = self.bill_ended = self.bill_rate = self.rate = 0
        self.heartbeat, self.test_since = mp.Value("d", 0), mp.Value("d", 0)
        self.stop_request, self.stop_reason, queue = mp.Event(), mp.Array("c", 160), mp.Queue()
        self.process = mp.Process(target=guard, args=(self.key, self.profile.rate_cap, self.stop_request, self.heartbeat,
                                                     self.test_since, queue, generation, self.stop_reason, self.profile,
                                                     approval))
        try:
            self.process.start()
        except OSError:
            self.process, self.state = None, "error"
            raise PreflightError("Windows watchdog could not start") from None
        threading.Thread(target=self.observe, args=(queue, self.process), daemon=True).start()

    def start_connection(self, fields, command):
        self.state, self.detail = "loading", "Checking SSH identity and loading the staged worker"
        threading.Thread(target=self.connect, args=(fields, command, self.stop_request), daemon=True).start()

    def observe(self, queue, process):
        from queue import Empty
        while True:
            try:
                state, fields = queue.get(timeout=1)
            except Empty:
                if self.request_since and time.monotonic()-self.request_since >= GRACE:
                    self.stop("Inference request timed out")
                if self.ssh and self.ssh.poll() is not None and self.state in {"loading","ready"}:
                    self.stop("SSH connection lost")
                if process.is_alive():
                    continue
                with self.lock:
                    if self.process is process and self.state not in {"stopped","error","shutdown_unverified"}:
                        self.detail = "Windows watchdog exited before confirming shutdown"
                        self.finish_shutdown(False, time.time())
                        self.log_shutdown(False)
                return
            with self.lock:
                if self.process is not process:
                    return
                if state == "pod_status":
                    self.record_pod_status(fields)
                    continue
                self.began = fields.get("began",self.began)
                self.rate = fields.get("rate",self.rate)
                if state in {"setup", "booting"}:
                    self.generation = fields["generation"]
                if "start_deadline" in fields:
                    self.start_deadline = fields["start_deadline"]
                if state in {"waiting", "setup", "starting", "booting"} and self.stop_request.is_set():
                    continue
                self.detail = fields.get("detail",self.detail)
                self.state = state
                if state in {"waiting", "error"}:
                    print(self.detail, flush=True)
                if state == "setup":
                    self.on_setup(fields)
                if state in {"stopping","watching_delayed_start","shutdown_unverified","stopped"}:
                    self.close_ssh()
                    if state in {"stopped","shutdown_unverified"}:
                        self.finish_shutdown(state == "stopped", fields["ended"])
                if state in {"stopped","shutdown_unverified"}:
                    return

    def on_setup(self, fields):
        self.detail = "Prepare the verified SSH connection, then attach this tab"
        if self.profile.started_at:
            setup = datetime.fromtimestamp(self.began + SETUP, timezone.utc).isoformat()
            cutoff = datetime.fromtimestamp(self.began + LIMIT, timezone.utc).isoformat()
            print(f"Watchdog armed for {self.profile.pod} at ${self.rate:g}/hour. "
                  f"Setup closes {setup}. Original hard limit {cutoff}.", flush=True)
        if self.owner:
            try:
                self.start_connection(fields, self.ssh_command(fields["ssh"]))
            except ValueError as error:
                self.stop(str(error))

    def record_pod_status(self, fields):
        status, observed = fields["pod_status"], fields["observed"]
        self.info["status"] = status
        if status in STOPPED:
            self.info["cost"] = 0
            if self.bill_started and not self.bill_ended:
                self.bill_ended = observed
        elif status == "RUNNING":
            if "actual_rate" in fields:
                self.bill_rate = self.info["cost"] = fields["actual_rate"]
            if not self.bill_started:
                self.bill_started = max(self.began, min(observed, fields.get("started_at", observed)))

    def finish_shutdown(self, verified, ended):
        self.ended = self.ended or ended
        if self.bill_started:
            self.bill_ended = self.bill_ended or ended
        if verified:
            if self.info.get("status") not in STOPPED:
                self.info["status"] = "STOPPED"
            self.info["cost"] = 0
        self.close_ssh()
        self.state = "stopped" if verified else "shutdown_unverified"

    def close_ssh(self):
        self.live = False
        if self.ssh:
            try:
                self.ssh.terminate()
            except OSError:
                pass
            self.ssh = None

    def ssh_command(self, direct):
        if not direct:
            raise ValueError("RunPod has not published the direct SSH endpoint yet")
        try:
            address = str(ipaddress.ip_address(direct["host"]))
            port = int(direct["port"])
            if not 0 < port < 65536 or direct["username"] != "root":
                raise ValueError
        except (KeyError, TypeError, ValueError):
            raise ValueError("Unsupported direct SSH endpoint") from None
        host = address if port == 22 else f"[{address}]:{port}"
        if not self.known_hosts or not self.known_hosts.is_file():
            raise ValueError("Missing per-run known_hosts file. Pin the current host key from the RunPod web terminal")
        try:
            result = subprocess.run(["ssh-keygen", "-F", host, "-f", str(self.known_hosts)],
                                    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=5)
        except (OSError, subprocess.TimeoutExpired):
            raise ValueError("Local SSH host key lookup failed") from None
        if result.returncode:
            raise ValueError("Pin the direct SSH endpoint using the current RunPod web terminal")
        command = ["ssh", "-F", "none", "-T", "-o", "BatchMode=yes", "-o", "StrictHostKeyChecking=yes",
                   "-o", "UpdateHostKeys=no",
                   "-o", "GlobalKnownHostsFile=none", "-o", f'UserKnownHostsFile="{self.known_hosts.as_posix()}"',
                   "-o", "HostKeyAlgorithms=ssh-ed25519", "-o", "ConnectTimeout=8",
                   "-o", "ServerAliveInterval=5", "-o", "ServerAliveCountMax=2", "-p", str(port)]
        if os.getenv("SIGNTOCAP_SSH_KEY"):
            command += ["-i", os.environ["SIGNTOCAP_SSH_KEY"]]
        remote = (
            "unset RUNPOD_POD_ID; set -a; . /etc/rp_environment || exit 41; set +a; "
            f'test "$RUNPOD_POD_ID" = {self.profile.pod} || exit 42; '
            "printf 'pod-verified\\n'; "
            'exec "$(cat /workspace/signgpt-validation/venv-path.txt)/bin/python" '
            f"-Bu {self.profile.worker_directory}/live_worker.py"
        )
        return command + ["root@" + address, remote]

    def connect(self, fields, command, stop):
        reason = "Strict SSH connection or Pod environment verification failed"
        try:
            with self.lock:
                if self.state != "loading" or stop.is_set() or self.stop_request is not stop:
                    return
                check_setup_time(self.began)
                deadline = self.began + LIMIT
                proc = self.ssh = subprocess.Popen(command,stdin=subprocess.PIPE,stdout=subprocess.PIPE,
                                                    stderr=subprocess.DEVNULL, text=True,encoding="utf-8",bufsize=1)
            if proc.stdout.readline().strip() != "pod-verified":
                if proc.wait(timeout=2) == 42:
                    reason = "SSH reported a different or missing RUNPOD_POD_ID"
                raise ValueError
            reason = "Staged GPU worker failed readiness"
            proc.stdin.write(json.dumps(dict(key=self.key, pod=self.profile.pod, gpu=self.profile.gpu,
                                             generation=fields["generation"], deadline=deadline))+"\n")
            proc.stdin.flush()
            if json.loads(proc.stdout.readline()).get("ready") is not True:
                raise ValueError("GPU worker failed readiness")
            with self.lock:
                if self.ssh is proc and self.state == "loading" and not stop.is_set():
                    check_setup_time(self.began)
                    self.heartbeat.value = time.time()
                    self.state = "ready"
                    self.detail = ""
        except Exception:
            with self.lock:
                if self.stop_request is stop and self.state == "loading" and not stop.is_set():
                    self.stop(reason)

    def stop(self, reason=EMERGENCY):
        with self.lock:
            if self.profile.started_at and not self.began:
                raise ValueError("Watchdog has not verified this running session. Use the RunPod console")
            if self.process and self.process.is_alive():
                if not self.stop_request.is_set():
                    self.detail = reason
                    self.stop_reason.value = reason.encode()
                    self.stop_request.set()
                    if self.state in {"arming", "waiting"} and not self.profile.started_at:
                        self.state = "watching_delayed_start"
                        self.detail += ". Watching for an in-flight manual start"
                    elif self.state not in {"stopped", "shutdown_unverified"}:
                        self.state = "stopping"
            elif self.state not in {"stopping", "stopped"}:
                self.state, self.detail = "stopping", reason
                threading.Thread(target=self.recover_stop, args=(self.generation,), daemon=True).start()

    def log_shutdown(self, verified):
        directory = Path(os.environ["LOCALAPPDATA"]) / "SignToCap"
        try:
            directory.mkdir(exist_ok=True)
        except OSError:
            return
        record_shutdown(directory / "live-shutdown.log", self.detail, verified)

    def recover_stop(self, generation):
        try:
            if generation is None:
                generation = api(self.key, pod=self.profile.pod).get("startedAt")
            ok = verified_stop(self.key, generation, pod=self.profile.pod)
        except Exception:
            ok = False
        with self.lock:
            self.finish_shutdown(ok, time.time())
            self.log_shutdown(ok)

    def infer(self, owner, action, body):
        with self.io:
            if self.state != "ready" or owner != self.owner or self.stop_request.is_set():
                raise ValueError("Live model unavailable for this session")
            if action == "prerecorded" and body.get("generation") != self.generation:
                raise ValueError("The GPU generation changed during video processing")
            if (action in {"start", "prerecorded"}) == self.live:
                raise ValueError("Duplicate start or inactive test")
            if action == "start":
                if body.get("mode", "live") not in ("live", "sentence"):
                    raise ValueError("Unknown capture mode")
                self.test_since.value = time.time()
                self.live = True
            try:
                self.request_since = time.monotonic()
                proc = self.ssh
                keys = {"frame": ("slot", "jpeg", "at_ms"), "frames": ("frames",),
                        "prerecorded": ("poses",), "start": ("mode",)}.get(action, ())
                req = {k:body[k] for k in keys if k in body}
                proc.stdin.write(json.dumps(dict(req,action=action))+"\n")
                proc.stdin.flush()
                result = json.loads(proc.stdout.readline())
                if action == "stop":
                    self.stop("Live test ended by operator")
                return result
            except Exception:
                self.stop("GPU inference connection failed")
                raise ValueError("GPU connection failed. Shutdown requested")
            finally:
                self.request_since = None
