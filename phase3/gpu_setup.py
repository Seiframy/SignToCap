"""Obtain trusted SSH identity and stage the existing worker in an isolated directory."""
import base64
import binascii
import hashlib
import ipaddress
import os
from pathlib import Path
import re
import shlex
import subprocess

from live_guard import PreflightError, ProviderError, provider
from live_worker import POSE_WEIGHTS

HERE = Path(__file__).resolve().parent
WORKER_FILES = {
    "phase3": ("desktop.py", "live_guard.py", "live_worker.py", "install-cuda-pose.sh",
               "requirements-cuda-benchmark.txt", "warmup.jpg"),
    "phase2": ("captions.py", "pose_data.py", "unisign.py"),
}
POD_ENV = "unset RUNPOD_POD_ID; set -a; . /etc/rp_environment || exit 41; set +a; "
HOST_KEY = "/etc/ssh/ssh_host_ed25519_key.pub"
HOST_KEY_COMMAND = (
    POD_ENV + 'test -n "$RUNPOD_POD_ID" && printf "SignToCap Pod %s\\n" "$RUNPOD_POD_ID"'
    f" && cat {HOST_KEY} && ssh-keygen -E sha256 -lf {HOST_KEY}"
)


class HostKeyUnavailable(ValueError):
    pass


def endpoint(info):
    try:
        direct = info["ssh"]["direct"]
        if not direct:
            return None
        address = ipaddress.ip_address(direct["host"])
        port = direct["port"]
        if (not address.is_global or str(address).find("%") >= 0 or type(port) is not int
                or not 0 < port < 65536 or direct["username"] != "root"):
            raise ValueError
        return str(address), port
    except (KeyError, TypeError, ValueError):
        if not info.get("ssh"):
            return None
        raise PreflightError("RunPod returned an unsupported direct SSH endpoint") from None


def parse_host_key(output, pod, challenge=None):
    lines = output.strip().splitlines()
    if challenge:
        if not lines or lines.pop(0) != f"SignToCap session {challenge}":
            raise PreflightError("Copy the output of this session's command from the RunPod web terminal")
    if len(lines) != 3 or lines[0] != f"SignToCap Pod {pod}":
        raise PreflightError("Host key output does not identify the selected Pod")
    fields = lines[1].split()
    try:
        if len(fields) < 2 or fields[0] != "ssh-ed25519":
            raise ValueError
        encoded = fields[1]
        blob = base64.b64decode(encoded, validate=True)
        if len(blob) != 51 or not blob.startswith(b"\x00\x00\x00\x0bssh-ed25519\x00\x00\x00\x20"):
            raise ValueError
        fingerprint = "SHA256:" + base64.b64encode(hashlib.sha256(blob).digest()).decode().rstrip("=")
        if not re.fullmatch(r"256 " + re.escape(fingerprint) + r"(?: [^\r\n]*)? \(ED25519\)", lines[2]):
            raise ValueError
    except (ValueError, binascii.Error):
        raise PreflightError("SSH public key and fingerprint do not match") from None
    return encoded, fingerprint


def authenticated_host_key(key, pod):
    try:
        reply = provider(key, f"/{pod}/exec", "host-key lookup",
                         {"command": ["sh", "-c", HOST_KEY_COMMAND], "timeoutSeconds": 8}, timeout=12)
    except ProviderError as error:
        if error.code == 403 and error.category == "API key cannot run commands on Pods":
            raise HostKeyUnavailable("RunPod allows Pod control but restricts command execution for this API key") from None
        if error.code in {401, 403}:
            raise PreflightError(f"RunPod refused authenticated host-key lookup (HTTP {error.code})") from None
        raise HostKeyUnavailable("RunPod's authenticated host-key command is unavailable") from None
    except OSError:
        raise HostKeyUnavailable("RunPod's authenticated host-key command is unavailable") from None
    if (reply.get("exitCode") != 0 or reply.get("timedOut") is not False
            or reply.get("truncated") is not False or not isinstance(reply.get("stdout"), str)):
        raise HostKeyUnavailable("RunPod could not read the SSH host key through its command API")
    return parse_host_key(reply["stdout"], pod)


def pin_host(directory, direct, encoded):
    address, port = direct
    host = address if port == 22 else f"[{address}]:{port}"
    path = directory / "known_hosts"
    with path.open("x", encoding="ascii") as handle:
        handle.write(f"{host} ssh-ed25519 {encoded}\n")
    return path


def ssh_options(known_hosts):
    command = ["-F", "none"]
    for option in ("BatchMode=yes", "StrictHostKeyChecking=yes", "UpdateHostKeys=no",
                   "GlobalKnownHostsFile=none", f'UserKnownHostsFile="{known_hosts.as_posix()}"',
                   "HostKeyAlgorithms=ssh-ed25519", "ConnectTimeout=8", "ServerAliveInterval=5",
                   "ServerAliveCountMax=2", "ForwardAgent=no", "ForwardX11=no", "ClearAllForwardings=yes"):
        command.extend(("-o", option))
    if os.getenv("SIGNTOCAP_SSH_KEY"):
        command.extend(("-i", os.environ["SIGNTOCAP_SSH_KEY"]))
    return command


def stage_worker(directory, direct, pin, pod, session, remaining, progress):
    host, port = direct
    options = ssh_options(pin)
    ssh = ["ssh", *options, "-T", "-p", str(port), "root@" + host]
    scp_host = f"[{host}]" if ":" in host else host
    remote = f"/workspace/signgpt-live-{pod}-{session}"
    identity = POD_ENV + f'test "$RUNPOD_POD_ID" = {shlex.quote(pod)} || exit 42; '
    mount = "mountpoint -q /workspace || exit 43; "
    files = []
    hashes = []
    for folder, names in WORKER_FILES.items():
        for name in names:
            data = (HERE.parent / folder / name).read_bytes()
            (directory / name).write_bytes(data)
            files.append(name)
            hashes.append(f"{hashlib.sha256(data).hexdigest()}  {name}")
    manifest = "worker.sha256"
    (directory / manifest).write_bytes(("\n".join(hashes) + "\n").encode("ascii"))
    # Use our local manifest as the trusted digest list, independent of the upload itself.
    manifest_digest = hashlib.sha256((directory / manifest).read_bytes()).hexdigest()

    def run(command, error, *, cwd=directory, limit=30):
        try:
            timeout = min(limit, remaining())
            result = subprocess.run(command, cwd=cwd, capture_output=True, text=True,
                                    timeout=timeout, stdin=subprocess.DEVNULL,
                                    creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        except (OSError, subprocess.TimeoutExpired):
            raise PreflightError(error) from None
        if result.returncode == 42:
            raise PreflightError("SSH reported a different or missing RUNPOD_POD_ID")
        if result.returncode == 43:
            raise PreflightError("The approved /workspace volume is not mounted")
        if result.returncode == 255 and "Host key verification failed" in result.stderr:
            raise PreflightError("Direct SSH host key differs from the authenticated pin")
        if result.returncode:
            raise PreflightError(error)
        return result.stdout.strip()

    progress("Verifying pinned SSH identity and mounted volume")
    marker = run(ssh + [identity + mount + f"mkdir -m 700 -- {remote} && printf 'pod-verified\\n'"],
                 "SSH verification failed. Check direct SSH and the installed Windows SSH identity.")
    if marker != "pod-verified":
        raise PreflightError("SSH did not verify the selected Pod")
    progress("Staging and verifying worker files")
    run(["scp", *options, "-B", "-P", str(port), *files, manifest, f"root@{scp_host}:{remote}/"],
        "Worker staging failed. Check SSH file transfer and available volume space.")
    layout = " && ".join(f"mkdir {folder} && mv -- {' '.join(names)} {folder}/"
                         for folder, names in WORKER_FILES.items())
    command = (identity + mount + f"cd {remote} && "
               f"printf '%s  %s\\n' {manifest_digest} {manifest} | sha256sum -c --status && "
               f"sha256sum -c --status {manifest} && {layout} && printf 'worker-verified\\n'")
    if run(ssh + [command], "Staged worker checksum or layout verification failed") != "worker-verified":
        raise PreflightError("Staging did not report verified worker files")
    work = "/workspace/signgpt-validation"
    boot = (identity + mount + f'cd {remote}/phase3 || exit 44; '
            f'venv="$(cat {work}/venv-path.txt)" && test -x "$venv/bin/python" || exit 45; ')
    progress("Verifying the existing pose model cache")
    cache_status = run(ssh + [boot + 'exec "$venv/bin/python" -B live_worker.py probe-pose-cache'],
                       "Pod runtime check failed. The shared volume needs its bootstrapped Python environment.")
    cache_env = "unset SIGNTOCAP_POSE_CACHE; "
    if cache_status == "pose-cache-incomplete":
        progress("Restoring verified pose weights to this session")
        cache = Path(os.environ["LOCALAPPDATA"]) / "Temp/signgpt-phase2-stage/ort-cache/rtmlib/hub/checkpoints"
        for name, digest in POSE_WEIGHTS.items():
            remaining()
            try:
                with (cache / name).open("rb") as handle:
                    if hashlib.file_digest(handle, "sha256").hexdigest() != digest:
                        raise ValueError
            except (OSError, ValueError):
                raise PreflightError("Pose cache is incomplete and the verified local weights are unavailable") from None
        target = f"{remote}/pose-cache/hub/checkpoints"
        run(ssh + [identity + mount + f"mkdir -p -m 700 -- {target}"], "Cannot prepare the session pose cache")
        run(["scp", *options, "-B", "-P", str(port), *POSE_WEIGHTS, f"root@{scp_host}:{target}/"],
            "Pose weight transfer failed or exceeded the setup time", cwd=cache, limit=240)
        cache_env = f"export SIGNTOCAP_POSE_CACHE={remote}/pose-cache; "
    elif cache_status != "pose-cache-ready":
        raise PreflightError("Could not verify the shared pose model cache")
    return ssh + [boot + cache_env +
                  "printf 'pod-verified\\n'; exec \"$venv/bin/python\" -Bu live_worker.py"]
