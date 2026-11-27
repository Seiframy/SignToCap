"""Private SSH stdio inference service and detached Pod-side cost guard."""
import json
import hashlib
from io import BytesIO
import shutil
import tempfile
import os
from pathlib import Path
import select
import subprocess
import sys
import time

from live_guard import (GPU_NAMES, LIMIT, SETUP, SHUTDOWN_MARGIN, TEST, Approval, PodProfile, api,
                        check_setup_time, deployment_time, record_shutdown, verified_stop)

WORK = Path("/workspace/signgpt-validation")
POSE_WEIGHTS = {
    "yolox_tiny_8xb8-300e_humanart-6f3252f9.onnx": "ceb11c07298f95c50d7c5abeb906d03340c85f23aa79e3e66966e7fb6c307250",
    "rtmw-dw-l-m_simcc-cocktail14_270e-256x192_20231122.onnx": "68db326c5c6f3273456f803eae0faf5440478369f2c3ee67258d2f36512e8d76",
}


def monitor():
    config = json.loads(sys.stdin.buffer.readline())
    if os.environ.get("RUNPOD_POD_ID") != config["pod"]:
        raise ValueError("Wrong Pod")
    began = deployment_time(config["generation"])
    check_setup_time(began, config.get("setup_seconds", SETUP))
    deadline = began + LIMIT - SHUTDOWN_MARGIN
    print("guard-armed", flush=True)
    reason = "15-minute Pod deadline (15-second shutdown reserve)"
    while time.time() < deadline:
        readable, _, _ = select.select([0], [], [], max(0, deadline - time.time()))
        if not readable:
            break
        line = sys.stdin.buffer.readline()
        if not line:
            reason = "SSH worker disconnected"
            break
        if line.strip() == b"test" and time.time() + TEST < deadline:
            deadline = time.time() + TEST
            reason = "60-second test deadline"
    log = Path(__file__).with_name("live-shutdown.log")
    record_shutdown(log, reason, None)
    stopped = False
    try:
        stopped = verified_stop(config["key"], config["generation"], pod=config["pod"])
    except RuntimeError:
        reason += ". Pod session changed"
    finally:
        record_shutdown(log, reason, stopped)


def verify_pose_weight(path, expected):
    if not path.exists():
        pieces=sorted(path.with_name(path.name+".parts").glob("[0-9][0-9][0-9]"))
        if not pieces:
            raise ValueError("Missing staged pose model")
        with tempfile.NamedTemporaryFile(dir=path.parent,delete=False) as out:
            tmp=Path(out.name)
        try:
            with tmp.open("wb") as out:
                for piece in pieces:
                    with piece.open("rb") as src:
                        shutil.copyfileobj(src,out)
            with tmp.open("rb") as handle:
                if hashlib.file_digest(handle,"sha256").hexdigest()!=expected:
                    raise ValueError("Incomplete staged pose model")
            tmp.replace(path)
        finally:
            tmp.unlink(missing_ok=True)
    with path.open("rb") as handle:
        if hashlib.file_digest(handle,"sha256").hexdigest()!=expected:
            raise ValueError("Pose model checksum mismatch")


def verify_pose_cache(root):
    cache = root / "hub/checkpoints"
    for filename, expected in POSE_WEIGHTS.items():
        verify_pose_weight(cache / filename, expected)


def load_model(gpu_names):
    work = WORK
    if not os.path.ismount("/workspace"):
        raise ValueError("Network volume is not mounted")
    os.environ["XDG_CACHE_HOME"] = str(work / "cache")
    pose_cache = Path(os.environ.get("SIGNTOCAP_POSE_CACHE") or work / "cache/rtmlib")
    os.environ["TORCH_HOME"] = str(pose_cache)
    os.environ["HF_HUB_OFFLINE"] = "1"
    subprocess.run(["sha256sum", "--check", "staged.sha256"], cwd=work / "assets", check=True)
    verify_pose_cache(pose_cache)
    import torch
    if torch.cuda.device_count() != 1 or torch.cuda.get_device_name() not in gpu_names:
        raise ValueError("CUDA GPU differs from the approved configuration")
    here = Path(__file__).resolve().parent
    target = work / "cache/ort-cuda-1.20.2-py312"
    try:
        subprocess.run(["timeout", "--kill-after=5s", "85s", "bash", str(here / "install-cuda-pose.sh"),
                        sys.executable, str(target)], check=True, timeout=95,
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except (OSError, subprocess.SubprocessError):
        device, reason = "cpu", "Pinned CUDA wheel unavailable for the Pod runtime"
    else:
        try:
            subprocess.run(
                [sys.executable, "-c", "import onnxruntime as ort; "
                 "assert ort.__version__ == '1.20.2' and "
                 "'CUDAExecutionProvider' in ort.get_available_providers()"],
                env={**os.environ, "PYTHONPATH": str(target)}, check=True, timeout=12,
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            )
        except (OSError, subprocess.SubprocessError):
            device, reason = "cpu", "Pinned ONNX Runtime could not load with CUDA support"
        else:
            sys.path.insert(0, str(target))
            device, reason = "cuda", None
    from desktop import UniSignBackend
    return UniSignBackend(work, pose_device=device, cuda_reason=reason, warmup_path=here / "warmup.jpg")


def serve():
    output = os.fdopen(os.dup(1), "w", buffering=1)
    os.dup2(2, 1)  # Native libraries may write to stdout during initialization.
    config = json.loads(sys.stdin.readline())
    if os.environ.get("RUNPOD_POD_ID") != config["pod"]:
        raise ValueError("Wrong Pod")
    generation = config["generation"]
    began = deployment_time(generation)
    setup_seconds = config.get("setup_seconds", SETUP)
    check_setup_time(began, setup_seconds)
    integrated = "compute" in config
    gpu_names = tuple(config["aliases"]) if integrated else GPU_NAMES[config["gpu"]]
    if integrated:
        try:
            profile = PodProfile(config["pod"], config["gpu"], config["cap"], aliases=gpu_names)
            approval = Approval(profile, "RUNNING", generation, config["compute"], config["storage"],
                                setup_seconds, config["image"], config.get("disk"))
            approval.running(api(config["key"], pod=profile.pod), generation, setup=True)
        except (OSError, ValueError, KeyError, TypeError):
            print(json.dumps({"error": "generation"}), file=output)
            return
    guardian = subprocess.Popen([sys.executable, "-B", __file__, "guard"], stdin=subprocess.PIPE,
                                stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, start_new_session=True)
    guardian.stdin.write((json.dumps(config) + "\n").encode())
    guardian.stdin.flush()
    del config
    with guardian.stdout:
        if guardian.stdout.readline() != b"guard-armed\n":
            raise ValueError("Pod-side watchdog failed to arm")
    if integrated:
        print(json.dumps({"guard": True}), file=output)
    try:
        model = load_model(gpu_names)
        check_setup_time(began, setup_seconds)
    except Exception as exc:
        if integrated:
            import traceback
            import re
            message = str(exc)[:1500]
            message = re.sub(r"rpa_[A-Za-z0-9]+", "[REDACTED]", message)
            message = re.sub(r"(?i)Bearer\s+\S+", "Bearer [REDACTED]", message)
            frames = [
                f"{os.path.basename(f.filename)}:{f.lineno} in {f.name}"
                for f in traceback.extract_tb(exc.__traceback__)
            ]
            print(json.dumps({
                "error": "model",
                "diagnostic": {
                    "type": type(exc).__name__,
                    "message": message,
                    "frames": frames
                }
            }), file=output)
        return  # Closing stdin wakes the detached guard to stop the exact generation.
    if guardian.poll() is not None:
        raise ValueError("Pod-side watchdog exited")
    ready = {"ready": True}
    if integrated:
        ready.update(generation=generation, guard=True, runtime=model.runtime)
    print(json.dumps(ready), file=output)
    import base64
    for line in sys.stdin:
        if guardian.poll() is not None:
            raise ValueError("Pod-side watchdog exited")
        req = json.loads(line)
        action = req["action"]
        began_request = time.perf_counter()
        event = None
        if action == "start" and model.session is None:
            guardian.stdin.write(b"test\n")
            guardian.stdin.flush()
            model.start(req.get("mode", "live"), req.get("sample_fps"))
        elif action == "frame" and model.session:
            event = model.frame(base64.b64decode(req["jpeg"], validate=True), req["slot"], req.get("at_ms"))
        elif action == "frames" and model.session:
            from desktop import frame_requests

            print(json.dumps({"frames": model.frames(frame_requests(req))}), file=output)
            continue
        elif action == "prerecorded" and model.session is None:
            try:
                event = model.translate_recorded(BytesIO(base64.b64decode(req["poses"], validate=True)))
            except ValueError:
                print(json.dumps({"error": "The prerecorded pose window could not be verified"}), file=output)
                continue
        elif action == "commit" and model.session:
            event = model.commit()
        elif action == "stop" and model.session:
            event = model.stop()
        else:
            raise ValueError("Invalid inference state")
        result = {"event": event}
        if action == "start":
            result["runtime"] = model.runtime
            result["capture_fps"] = model.capture_fps
        if action == "frame":
            result.update(pose_ms=model.pose_ms, jpeg_ms=model.jpeg_ms, dropped=model.dropped,
                          worker_ms=(time.perf_counter() - began_request) * 1000)
        print(json.dumps(result), file=output)


if __name__ == "__main__":
    if sys.argv[1:] == ["guard"]:
        monitor()
    elif sys.argv[1:] == ["probe-pose-cache"]:
        try:
            verify_pose_cache(WORK / "cache/rtmlib")
        except (ValueError, OSError):
            print("pose-cache-incomplete")
        else:
            print("pose-cache-ready")
    else:
        serve()
