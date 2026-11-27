"""Short rehearsal only. Pod transitions are serialized by the Windows watchdog."""
import json
import math
import os
from pathlib import Path
import re
import subprocess
import time
from dataclasses import dataclass
from datetime import datetime
from urllib.error import HTTPError
from contextlib import contextmanager
from decimal import Decimal, ROUND_CEILING
from urllib.parse import urlencode
from urllib.request import HTTPRedirectHandler, Request, build_opener

POD = "1jce4o05fmadxn"
VOLUME = "4q8k7ibb17"
STOPPED = {"STOPPED", "EXITED", "TERMINATED"}
LIMIT = 900
TEST = 60
GRACE = 20
SETUP = 300
DEFAULT_SETUP = 600
SHUTDOWN_MARGIN = 15
START_WAIT = 600
RATE_CAP = 0.70
EMERGENCY = "Emergency stop requested"
RTX_4090 = "NVIDIA RTX 4090"
A100_SXM_80GB = "NVIDIA A100-SXM4-80GB"
GPU_NAMES = {
    "NVIDIA L4": {"NVIDIA L4", "L4"},
    RTX_4090: {RTX_4090, "NVIDIA GeForce RTX 4090", "GeForce RTX 4090", "RTX 4090"},
    A100_SXM_80GB: {A100_SXM_80GB, "NVIDIA A100 80GB SXM", "A100 SXM 80 GB"},
}
GPU_NAMES.setdefault('NVIDIA H100 80GB HBM3', set()).update(['H100 SXM', 'NVIDIA H100 80GB HBM3', 'NVIDIA H100 SXM'])


class PreflightError(ValueError):
    """A validation failure whose message is safe to show."""


class ProviderError(OSError):
    def __init__(self, code, category, action):
        self.code, self.category = code, category
        super().__init__(f"RunPod {action} returned HTTP {code}: {category}")

    @property
    def rejected_start(self):
        return self.code in {401, 403} or (self.code in {400, 409, 422} and self.category in {
            "GPU capacity unavailable", "insufficient account balance", "authorization refused",
            "Start request validation failed",
        })


def pod_id(value):
    # This is a path/shell safety check. A successful API lookup proves identity.
    if not isinstance(value, str) or not re.fullmatch(r"[A-Za-z0-9_-]{1,128}", value):
        raise PreflightError("Invalid Pod identifier")
    return value


class NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, *args, **kwargs):
        return None


def urlopen(request, timeout):
    return build_opener(NoRedirect()).open(request, timeout=timeout)


@dataclass(frozen=True)
class PodProfile:
    pod: str = POD
    gpu: str = "NVIDIA L4"
    rate_cap: float = RATE_CAP
    started_at: str | None = None
    aliases: tuple[str, ...] = ()

    @property
    def gpu_names(self):
        return self.aliases or GPU_NAMES[self.gpu]

    @classmethod
    def rtx4090(cls, pod, rate_cap, started_at):
        pod_id(pod)
        if not math.isfinite(rate_cap) or rate_cap <= 0:
            raise PreflightError("Enter the explicitly approved positive USD/hour ceiling")
        check_setup_time(deployment_time(started_at))
        return cls(pod, RTX_4090, rate_cap, started_at)

    @property
    def worker_directory(self):
        return f"/workspace/phase3-rtx4090-{self.pod}" if self.started_at else "/workspace/phase3"


L4_PROFILE = PodProfile()


def request_error(error, key, action):
    # Never surface raw response text, headers, URLs, or arbitrary provider messages.
    # Match known categories and return only constant strings.
    with error:
        raw = error.read(4096).decode("utf-8", errors="replace").replace(key, "") if key else ""
    categories = (
        (r"this api key cannot run commands on pods", "API key cannot run commands on Pods"),
        (r"no (?:available )?gpus?|zero gpus?|gpu.{0,40}(unavailable|availability|not available|already rented)|insufficient.{0,20}gpu|out of capacity", "GPU capacity unavailable"),
        (r"insufficient.{0,25}(funds|balance|credits)", "insufficient account balance"),
        (r"invalid.{0,20}(action|transition)|cannot (start|stop)|not in.{0,20}state", "invalid Pod transition"),
        (r"permission|unauthori[sz]ed|forbidden", "authorization refused"),
        (r"invalid (request body|json)|malformed json|request body validation failed|unsupported action", "Start request validation failed"),
    )
    default = "authorization refused" if error.code in {401, 403} else "response did not match a known error category"
    detail = next((label for pattern, label in categories if re.search(pattern, raw, re.I)), default)
    return ProviderError(error.code, detail, action or "status")


def provider(key, path, action, body=None, timeout=5, *, response=True):
    data = None if body is None else json.dumps(body).encode()
    req = Request("https://api.runpod.io/v2/pods" + path, data=data,
                  headers={"Authorization": "Bearer " + key, "Content-Type": "application/json"})
    try:
        with urlopen(req, timeout=timeout) as result:
            if response:
                info = json.load(result)
                if not isinstance(info, dict):
                    raise PreflightError("Unexpected RunPod response")
                return info
    except HTTPError as error:
        raise request_error(error, key, action) from None


def api(key, action=None, *, pod=POD, allow_start=False):
    if action not in {None, "stop"} and not (action == "start" and allow_start):
        raise PreflightError("Pod start requires the armed watchdog and explicit approval")
    path = "/" + pod_id(pod)
    if action:
        return provider(key, path + "/action", action, {"action": action}, response=False)
    info = provider(key, path, "status")
    if info.get("id") != pod:
        raise PreflightError("Unexpected Pod identity")
    return info


def list_pods(key):
    cursor = None
    seen = set()
    while True:
        query = "?" + urlencode({"cursor": cursor}) if cursor else ""
        page = provider(key, query, "list")
        rows, pagination = page.get("pods"), page.get("pagination")
        if not isinstance(rows, list) or not isinstance(pagination, dict):
            raise PreflightError("Unexpected RunPod list format")
        yield from rows
        if pagination.get("hasNextPage") is False:
            return
        cursor = pagination.get("nextCursor")
        if not isinstance(cursor, str) or not cursor or cursor in seen:
            raise PreflightError("Invalid RunPod pagination")
        seen.add(cursor)


def compatible(info, profile=L4_PROFILE):
    gpu = info.get("gpu") or {}
    if (not isinstance(gpu, dict) or info.get("id") != profile.pod or gpu.get("id") not in profile.gpu_names
            or type(gpu.get("count")) is not int or gpu["count"] != 1
            or info.get("dataCenterId") != "EUR-IS-1"):
        raise PreflightError("Pod GPU or data center differs from the approved configuration")
    mounts = info.get("mounts") or {}
    if not isinstance(mounts, dict):
        raise PreflightError("Unexpected Pod mounts")
    network = mounts.get("network", [])
    if not isinstance(network, list):
        raise PreflightError("Unexpected network mount format")
    workspace = [
        m for m in network
        if isinstance(m, dict) and m.get("path") == "/workspace"
    ]
    if len(workspace) != 1:
        raise PreflightError("Expected /workspace mount missing or ambiguous")
    if workspace[0].get("volumeId") != VOLUME:
        raise PreflightError("Different network volume attached")
    if info.get("locked"):
        raise PreflightError("Pod is locked")


def preflight(info, quote):
    compatible(info)
    check_quote(quote)
    if info["status"] not in {"STOPPED", "EXITED"} or "start" not in info.get("actions", []):
        raise PreflightError("The approved Pod must be stopped and offer manual Start")


def check_quote(quote):
    if not math.isfinite(quote) or not 0 < quote <= RATE_CAP:
        raise PreflightError(f"A positive approved rate at or below ${RATE_CAP:.2f}/hour is required")


def deployment_time(generation):
    try:
        started = datetime.fromisoformat(generation.replace("Z", "+00:00"))
    except (AttributeError, ValueError):
        raise PreflightError("Unknown deployment time") from None
    if started.tzinfo is None:
        raise PreflightError("Unknown deployment time")
    return started.timestamp()


def setup_deadline(began, seconds):
    return began + min(seconds, LIMIT - SHUTDOWN_MARGIN)


def check_setup_time(began, seconds=SETUP):
    elapsed = time.time() - began
    if not 0 <= elapsed < LIMIT - SHUTDOWN_MARGIN:
        raise PreflightError("Deployment deadline exceeded")
    if time.time() >= setup_deadline(began, seconds):
        prefix = "Five-minute setup" if seconds == SETUP else "Setup"
        raise PreflightError(f"{prefix} deadline passed for the original Pod start")


def attach_preflight(info, quote, expected=None, *, profile=L4_PROFILE, setup=False, setup_seconds=SETUP):
    compatible(info, profile)
    if not math.isfinite(quote) or not 0 < quote <= profile.rate_cap:
        raise PreflightError("Invalid approved rate ceiling")
    if info.get("status") != "RUNNING" or "stop" not in (info.get("actions") or []):
        raise PreflightError("Pod must be running and stoppable")
    raw_rate = info.get("cost")
    try:
        rate = float(raw_rate)
    except (TypeError, ValueError, OverflowError):
        raise PreflightError("Running rate could not be verified") from None
    if isinstance(raw_rate, bool) or not math.isfinite(rate) or rate <= 0:
        raise PreflightError("Running rate could not be verified")
    if rate > quote:
        raise PreflightError("Running rate exceeds approved quote")
    generation = info.get("startedAt")
    if expected is not None and generation != expected:
        raise PreflightError("Pod session changed")
    began = deployment_time(generation)
    if profile.started_at and began != deployment_time(profile.started_at):
        raise PreflightError("Running Pod does not match the supplied original startedAt")
    if setup:
        check_setup_time(began, setup_seconds)
    elif not 0 <= time.time() - began < LIMIT - SHUTDOWN_MARGIN:
        raise PreflightError("Deployment deadline exceeded")
    return generation, began, rate


def compute_rate(info):
    value = info.get("cost")
    try:
        rate = float(value)
    except (TypeError, ValueError, OverflowError):
        rate = 0
    if isinstance(value, bool) or not math.isfinite(rate) or rate <= 0:
        raise PreflightError("RunPod has no positive compute quote for this Pod")
    return rate


def total_rate(compute, storage):
    return float(Decimal(str(compute)) + Decimal(str(storage)))


def container_storage_rate(info):
    disk = info.get("disk")
    if type(disk) is not int or disk <= 0:
        raise PreflightError("RunPod did not report a valid container disk size. Refresh Pods to verify storage")
    # RunPod lists $0.10/GB/month for a running container disk. Budget for a
    # 28-day month and round up to the next $0.001/hour.
    # https://docs.runpod.io/accounts-billing/billing
    return float((disk * Decimal("0.10") / (28 * 24)).quantize(Decimal("0.001"), rounding=ROUND_CEILING))


@dataclass(frozen=True)
class Approval:
    profile: PodProfile
    status: str
    generation: str | None
    compute: float
    storage: float
    setup_seconds: int
    image: str | None = None
    disk: int | None = None

    @property
    def starting(self):
        return self.status in {"EXITED", "STOPPED"}

    @property
    def total(self):
        return total_rate(self.compute, self.storage)

    def check_storage(self, info):
        mounts = info["mounts"]
        if len(mounts["network"]) != 1 or mounts.get("persistent") or mounts.get("global"):
            raise PreflightError("Additional storage makes the total rate unverifiable")
        if self.disk is not None and (info.get("disk") != self.disk or container_storage_rate(info) > self.storage):
            raise PreflightError("Pod container storage changed. Refresh and confirm the current rate")

    def validate(self, info):
        compatible(info, self.profile)
        if self.image is not None and info.get("image") != self.image:
            raise PreflightError("Pod container image changed. Refresh before approving")
        self.check_storage(info)
        if self.total > self.profile.rate_cap or not 0 < self.storage < self.total:
            raise PreflightError("Total hourly rate exceeds the ceiling or storage rate is missing")
        if info.get("status") != self.status or info.get("startedAt") != self.generation:
            raise PreflightError("Pod session changed. Refresh and confirm the current Pod")
        if compute_rate(info) != self.compute:
            raise PreflightError("Compute quote changed. Refresh and confirm the current rate")
        if self.starting:
            if "start" not in (info.get("actions") or []) or "22/tcp" not in (info.get("ports") or []):
                raise PreflightError("Pod does not offer Start with a configured direct SSH port")
        else:
            attach_preflight(info, self.total, self.generation, profile=self.profile,
                             setup=True, setup_seconds=self.setup_seconds)

    def running(self, info, generation, setup=False):
        current, began, compute = attach_preflight(
            info, self.total, generation, profile=self.profile, setup=setup, setup_seconds=self.setup_seconds)
        if total_rate(compute, self.storage) > self.total:
            raise PreflightError("Running total rate exceeds the confirmed quote")
        if self.image is not None and info.get("image") != self.image:
            raise PreflightError("Pod container image changed")
        self.check_storage(info)
        return current, began, total_rate(compute, self.storage)

def verified_stop(key, generation=None, attempts=12, observe=None, *, pod=POD):
    for _ in range(attempts):
        try:
            info = api(key, pod=pod)
            if info["status"] not in STOPPED and (not generation or info.get("startedAt") != generation):
                raise RuntimeError("Pod session changed. Check the RunPod console")
            if observe:
                observe(info)
            if info["status"] in STOPPED:
                return True
            api(key, "stop", pod=pod)
        except (OSError, ValueError):
            pass
        time.sleep(2)
    return False


def old_watchdog_check(pod):
    script = r"""$ErrorActionPreference='Stop'
foreach($w in Get-CimInstance Win32_Process -Filter "Name LIKE 'python%.exe'") {
    if ($w.CommandLine -notmatch 'run_validation[.]py.*watchdog' -or $w.CommandLine -notmatch '(?<![A-Za-z0-9_-])__POD__(?![A-Za-z0-9_-])') {continue}
    exit 9
}"""
    script = script.replace("__POD__", re.escape(pod_id(pod)))
    result=subprocess.run(["powershell.exe", "-NoProfile", "-Command", script], timeout=12,
                   capture_output=True, creationflags=subprocess.CREATE_NO_WINDOW)
    if result.returncode:
        raise PreflightError("Legacy watchdog active or unreadable. Check the old watchdog before starting")


def deadline_reason(now, began, last_seen, test, setup_seconds=SETUP):
    if now >= began + LIMIT - SHUTDOWN_MARGIN:
        return "15-minute Pod deadline (15-second shutdown reserve)"
    if test and now - test >= TEST:
        return "60-second test deadline"
    if last_seen and now - last_seen >= GRACE:
        return "Browser heartbeat lost"
    if not last_seen and now >= setup_deadline(began, setup_seconds):
        return "Setup deadline before worker readiness"
    return ""


def record_shutdown(path, reason, verified):
    try:
        with path.open("a", encoding="utf-8") as log:
            outcome = "stop requested" if verified is None else ("verified stopped" if verified else "stop unverified")
            print(f"{datetime.now().astimezone().isoformat()} | {reason} | {outcome}", file=log)
    except OSError:
        pass


def publish_status(updates, info):
    status = info["status"]
    fields = {"pod_status": status, "observed": time.time()}
    if status == "RUNNING":
        try:
            rate = float(info["cost"])
            if math.isfinite(rate) and rate > 0:
                fields["actual_rate"] = rate
            fields["started_at"] = datetime.fromisoformat(info["startedAt"].replace("Z", "+00:00")).timestamp()
        except (AttributeError, KeyError, TypeError, ValueError):
            pass
    updates.put(("pod_status", fields))


def state_directory():
    return Path(os.environ["LOCALAPPDATA"]) / "SignToCap"


@contextmanager
def session_lock(directory):
    import msvcrt
    directory.mkdir(exist_ok=True)
    with (directory / "gpu-session.lock").open("a+b") as lock:
        lock.seek(0)
        try:
            msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
        except OSError:
            raise PreflightError("Another GPU session holds the Windows watchdog lock") from None
        yield


def pending_path(directory):
    return directory / "gpu-shutdown-pending.json"


def mark_pending(directory, pod, generation, **startup):
    path = pending_path(directory)
    temporary = path.with_suffix(".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump({"pod": pod, "generation": generation, "marked_at": time.time(), **startup}, handle)
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def pending_record(directory):
    try:
        data = json.loads(pending_path(directory).read_text(encoding="utf-8"))
    except FileNotFoundError:
        return None
    pod_id(data["pod"])
    return data


def pending_session(directory):
    record = pending_record(directory)
    return (record["pod"], record["generation"], record["marked_at"]) if record else None


def reconcile_pending(key, manual=False):
    directory = state_directory()
    with session_lock(directory):
        pending = pending_record(directory)
        if not pending:
            return True
        pod, generation, marked_at = (pending[field] for field in ("pod", "generation", "marked_at"))
        if generation:
            verified = verified_stop(key, generation, pod=pod)
        elif "previous_generation" in pending:
            info = api(key, pod=pod)
            generation = observed_start_generation(info, pending["previous_generation"], marked_at)
            if generation:
                mark_pending(directory, pod, generation)
                verified = verified_stop(key, generation, pod=pod)
            else:
                stopped = info.get("status") in STOPPED and info.get("startedAt") == pending["previous_generation"]
                rejected = pending.get("rejected") is True
                verified = stopped and (rejected or manual and time.time() >= marked_at + START_WAIT)
                if rejected and not stopped:
                    mark_pending(directory, pod, None, marked_at=marked_at, rejected=False,
                                 previous_generation=pending["previous_generation"], start_detail=pending.get("start_detail"))
        else:
            # A stopped snapshot alone cannot settle a lost, potentially delayed Start.
            verified = manual and time.time() >= marked_at + START_WAIT and api(key, pod=pod).get("status") in STOPPED
        record_shutdown(directory / "live-shutdown.log", f"{pod}: Shutdown reconciliation", verified)
        if verified:
            pending_path(directory).unlink()
        return verified


def parent_alive():
    from multiprocessing import parent_process
    parent = parent_process()
    return parent is None or parent.is_alive()


def observed_start_generation(info, previous, submitted):
    generation = info.get("startedAt")
    if generation and generation != previous:
        if submitted - 2 <= deployment_time(generation) <= min(time.time(), submitted + START_WAIT):
            return generation
        raise PreflightError("Pod generation does not match this Start request")
    return None


def wait_for_start(key, approval, updates, submitted, rejection, directory):
    pod = approval.profile.pod
    while time.time() < submitted + START_WAIT:
        try:
            info = api(key, pod=pod)
            publish_status(updates, info)
            generation = observed_start_generation(info, approval.generation, submitted)
            if generation:
                return info, generation
            stopped = info.get("status") in STOPPED
            if rejection and stopped and info.get("startedAt") == approval.generation:
                return info, None
            if rejection and not stopped:
                rejection = False
                record = pending_record(directory)
                if record and record.get("rejected") is True:
                    mark_pending(directory, pod, None, marked_at=submitted, rejected=False,
                                 previous_generation=approval.generation, start_detail=record["start_detail"])
        except OSError:
            pass
        time.sleep(1)
    return None, None


def other_active_pod(rows, selected):
    for info in rows:
        if not isinstance(info, dict):
            raise PreflightError("RunPod returned a malformed Pod list")
        if info.get("id") == selected or info.get("status") in STOPPED:
            continue
        mounts = info.get("mounts") or {}
        network = mounts.get("network") if isinstance(mounts, dict) else None
        if not isinstance(mounts, dict) or (network is not None and
                (not isinstance(network, list) or not all(isinstance(mount, dict) for mount in network))):
            raise PreflightError("RunPod returned malformed Pod mounts")
        if any(mount.get("volumeId") == VOLUME for mount in network or []):
            return pod_id(info.get("id"))
    return None


def run_approved_guard(key, approval, stop, heartbeat, test_since, updates, stop_reason):
    directory = state_directory()
    profile = approval.profile
    pod = profile.pod
    with session_lock(directory):
        if pending_session(directory):
            raise PreflightError("Resolve the previous unverified shutdown before connecting")
        before = api(key, pod=pod)
        approval.validate(before)
        old_watchdog_check(pod)
        if other_active_pod(list_pods(key), pod):
            raise PreflightError("Another active Pod uses the shared volume. Verify it is stopped in RunPod first")
        before = api(key, pod=pod)
        approval.validate(before)
        verified = False
        generation = None if approval.starting else approval.generation
        if approval.starting and (stop.is_set() or not parent_alive()):
            updates.put(("stopped", {"detail": "Start canceled before dispatch", "ended": time.time()}))
            return
        startup = {"previous_generation": approval.generation} if approval.starting else {}
        mark_pending(directory, pod, generation, **startup)
        reason = "Windows watchdog failed"
        try:
            if approval.starting:
                updates.put(("starting", {"detail": "Windows watchdog armed. Requesting one Start."}))
                if stop.is_set() or not parent_alive():
                    reason, verified = "Start canceled before dispatch", True
                    return
                submitted = time.time()
                deadline = submitted + START_WAIT
                rejection = None
                outcome = "RunPod accepted the request. Waiting for the Pod generation."
                try:
                    api(key, "start", pod=pod, allow_start=True)
                except ProviderError as error:
                    outcome = str(error)
                    rejection = error.rejected_start
                except OSError:
                    outcome = "RunPod Start reply missing"
                try:
                    mark_pending(directory, pod, None, marked_at=submitted, rejected=bool(rejection),
                                 previous_generation=approval.generation, start_detail=outcome)
                except OSError:
                    pass  # The pre-dispatch marker still blocks a competing watchdog.
                updates.put(("watching_delayed_start", {"detail": outcome, "start_deadline": deadline}))
                before, generation = wait_for_start(key, approval, updates, submitted, rejection, directory)
                if not generation:
                    verified = before is not None and bool(rejection)
                    reason = (outcome + ". RunPod verified the Pod is stopped. Choose another GPU."
                              if verified else outcome + ". Monitoring expired without a verified outcome.")
                    return
                mark_pending(directory, pod, generation)
                if before["status"] in STOPPED:
                    reason, verified = "The new Pod generation already stopped", True
                    return
                updates.put(("booting", {"began": deployment_time(generation), "generation": generation,
                                         "detail": "RunPod is starting the selected Pod"}))

            began = deployment_time(generation)
            info = before
            in_setup = False
            while True:
                app_alive = parent_alive()
                if stop.is_set() or not app_alive:
                    reason = stop_reason.value.decode() or (EMERGENCY if app_alive else "Desktop app closed")
                    break
                reason = deadline_reason(time.time(), began, heartbeat.value, test_since.value, approval.setup_seconds)
                if reason:
                    break
                if info["status"] in STOPPED:
                    reason, verified = "Pod stopped outside the watchdog", True
                    publish_status(updates, info)
                    break
                if info.get("startedAt") != generation:
                    raise PreflightError("Pod session changed")
                if info["status"] == "RUNNING":
                    _, _, rate = approval.running(info, generation, setup=not in_setup)
                    if not in_setup:
                        updates.put(("setup", {"began": began, "rate": rate, "generation": generation}))
                        in_setup = True
                elif in_setup or info["status"] not in {"PROVISIONING", "STARTING"}:
                    raise PreflightError("RunPod reported an unexpected state for the selected generation")
                publish_status(updates, info)
                stop.wait(1)
                info = api(key, pod=pod)
        except PreflightError as error:
            reason = str(error)
        except (OSError, ValueError, KeyError, TypeError):
            reason = "RunPod status or watchdog check failed"
        finally:
            if generation and not verified:
                updates.put(("stopping", {"detail": reason}))
                try:
                    verified = verified_stop(key, generation, observe=lambda info: publish_status(updates, info), pod=pod)
                except (RuntimeError, KeyError, TypeError):
                    reason += ". Pod state or generation could not be verified"
            record_shutdown(directory / "live-shutdown.log", f"{pod}: {reason}", verified)
            if verified:
                pending_path(directory).unlink()
            updates.put(("stopped" if verified else "shutdown_unverified", {"detail": reason, "ended": time.time()}))
            if not parent_alive() and hasattr(updates, "cancel_join_thread"):
                updates.cancel_join_thread()


def guard(key, quote, stop, heartbeat, test_since, updates, attach_generation, stop_reason, profile=L4_PROFILE,
          approval=None):
    if approval is not None:
        try:
            run_approved_guard(key, approval, stop, heartbeat, test_since, updates, stop_reason)
        except PreflightError as error:
            updates.put(("error", {"detail": str(error)}))
        except (OSError, ValueError, KeyError, TypeError):
            updates.put(("error", {"detail": "Windows watchdog could not complete its safety checks"}))
        return
    directory = state_directory()
    generation = None
    armed = False
    verified = False
    reason = ""
    with session_lock(directory):
        try:
            before = api(key, pod=profile.pod)
            if profile.started_at and attach_generation is None:
                raise PreflightError("The RTX 4090 profile requires an exact running generation")
            if attach_generation is not None:
                attach_preflight(before, quote, attach_generation, profile=profile, setup=True)
            else:
                preflight(before, quote)
            old_watchdog_check(profile.pod)
            before = api(key, pod=profile.pod)
            if attach_generation is not None:
                generation, began, rate = attach_preflight(before, quote, attach_generation, profile=profile, setup=True)
            else:
                preflight(before, quote)
                armed = True
                arm_deadline = time.time() + SETUP
                updates.put(("waiting", {"detail":"Watchdog armed. Start the approved Pod in RunPod now."}))
                while True:
                    # Continue observing a manual start in flight after an emergency request.
                    info = api(key, pod=profile.pod)
                    candidate = info.get("startedAt")
                    if info["status"] == "RUNNING" and candidate and candidate != before.get("startedAt"):
                        generation = candidate
                        _, began, rate = attach_preflight(info, quote, generation)
                        before = info
                        break
                    publish_status(updates, info)
                    if time.time() >= arm_deadline:
                        reason = stop_reason.value.decode() or "Manual startup window expired"
                        verified = info["status"] in STOPPED
                        return
                    time.sleep(1)
            updates.put(("setup", {"began": began, "rate": rate, "generation": generation,
                                   "ssh": (before.get("ssh") or {}).get("direct")}))
            publish_status(updates, before)
            while True:
                if stop.is_set():
                    reason = stop_reason.value.decode() or EMERGENCY
                    break
                reason = deadline_reason(time.time(), began, heartbeat.value, test_since.value)
                if reason:
                    break
                info = api(key, pod=profile.pod)
                if info["status"] in STOPPED:
                    reason = "Pod stopped outside the watchdog"
                    verified = True
                    publish_status(updates, info)
                    break
                attach_preflight(info, quote, generation, profile=profile)
                publish_status(updates, info)
                stop.wait(1)
        except (PreflightError, OSError) as error:
            reason = str(error) if isinstance(error, PreflightError) else "RunPod status or watchdog check failed"
            if not (armed or generation):
                updates.put(("error", {"detail":reason}))
        except Exception:
            reason = "Windows watchdog failed"
            if not (armed or generation):
                updates.put(("error", {"detail":reason}))
        finally:
            if armed or generation:
                if generation and not verified:
                    updates.put(("stopping", {"detail":reason}))
                    try:
                        verified = verified_stop(key, generation, observe=lambda info: publish_status(updates, info),
                                                 pod=profile.pod)
                    except RuntimeError:
                        if "Pod session changed" not in reason:
                            reason += ". Pod session changed"
                    except Exception:
                        pass
                record_shutdown(directory / "live-shutdown.log", reason, verified)
                updates.put(("stopped" if verified else "shutdown_unverified",
                             {"ended":time.time(), "detail":reason}))
