"""Run the frozen Sign GPT benchmark on a Pod, or stop it at a deadline from a second machine."""

import argparse
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import time
from urllib.request import Request, urlopen


API = "https://api.runpod.io/v2/pods"
MAX_BUDGET_USD = 10.0
NON_GPU_RESERVE_USD = 1.0
MAX_RUNTIME_SECONDS = 4 * 3600
BACKUP_SECONDS = 3 * 3600 + 30 * 60
MAX_RATE = 2.25


def request(pod, key, *, stop=False):
    url = f"{API}/{pod}"
    headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}
    req = Request(url + "/action", data=b'{"action":"stop"}', headers=headers) if stop else Request(url, headers=headers)
    with urlopen(req, timeout=12) as response:
        data = response.read()
    return json.loads(data) if data and not stop else None


def stop_pod(pod, key):
    for attempt in range(3):
        try:
            request(pod, key, stop=True)
            print("RunPod accepted the Stop request.", flush=True)
            return True
        except OSError as error:
            print(f"Stop attempt {attempt + 1} failed: {error}", flush=True)
            if attempt < 2:
                time.sleep(5)
    print("STOP FAILED. Use the RunPod console to stop this Pod.", flush=True)
    return False


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def verify_pod(pod, key):
    info = request(pod, key)
    if info.get("id") != pod:
        raise ValueError("RunPod returned a different Pod ID")
    return info


def spending_limits(info, started):
    rate = float(info["cost"])
    if not math.isfinite(rate) or not 0 < rate <= MAX_RATE:
        raise ValueError(f"RunPod rate {rate} USD/h is outside the approved range")
    created = info.get("createdAt")
    if created:
        started = min(started, datetime.fromisoformat(created.replace("Z", "+00:00")).timestamp())
    affordable_seconds = (MAX_BUDGET_USD - NON_GPU_RESERVE_USD) * 3600 / rate
    return rate, started + min(BACKUP_SECONDS, MAX_RUNTIME_SECONDS, affordable_seconds)


def watchdog(args, key, deadline):
    print(f"{utc_now()} Backup stop armed for {args.pod} at {datetime.fromtimestamp(deadline, timezone.utc)}", flush=True)
    time.sleep(max(0, deadline - time.time()))
    try:
        if verify_pod(args.pod, key).get("status") in {"EXITED", "TERMINATED", "STOPPED"}:
            print("Pod already stopped.", flush=True)
            return 0
    except OSError:
        print("Status lookup failed. Attempting the scheduled stop.", flush=True)
    return 0 if stop_pod(args.pod, key) else 2


def run_step(command, env, cwd, work_deadline, stdout=None):
    remaining = work_deadline - time.time()
    if remaining <= 0:
        raise TimeoutError("Validation time limit reached")
    print(f"{utc_now()} {Path(command[0]).name}: {' '.join(map(str, command[1:]))}", flush=True)
    with subprocess.Popen(command, cwd=cwd, env=env, stdout=stdout, start_new_session=True) as process:
        try:
            status = process.wait(timeout=remaining)
        except BaseException:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait()
            raise
        if status:
            raise subprocess.CalledProcessError(status, command)


def validate(source, inputs, work, output, work_deadline, *, dataset, continuous_manifest, official_continuous=False,
             followup=False):
    if followup:
        required = [source.parent / "phase3/followup.py", work / "assets/staged.sha256",
                    work / "prepared/phase3/pose-audit.json", work / "prepared/phase3/publisher.jsonl",
                    work / "prepared/phase3/fresh.jsonl", work / "prepared/continuous/continuous.jsonl"]
        missing = [str(path) for path in required if not path.is_file()]
        if missing:
            raise FileNotFoundError("Phase 3 staged files missing: " + ", ".join(missing))
    videos = sorted(inputs.glob("*.mp4")) if dataset == "webcam" else []
    if dataset == "webcam":
        if not 2 <= len(videos) <= 10 or not any(not video.stem.startswith("idle-") for video in videos):
            raise ValueError("Supply 2 to 10 MP4s, including at least one signing clip")
    excluded = {"RUNPOD_API_KEY", "PYTHONPATH", "PYTHONHOME"}
    env = {k: v for k, v in os.environ.items() if k not in excluded}
    env.update(PYTHONUNBUFFERED="1", XDG_CACHE_HOME=str(work / "cache"), SIGNGPT_PYTHON=sys.executable)

    # Reuse the Windows-uploaded scripts with Linux line endings.
    scripts = output / "setup"
    scripts.mkdir()
    for name in ("bootstrap.sh", "source-only.patch", "pose-imports.patch", "requirements.txt", "prepare_weights.py"):
        (scripts / name).write_bytes((source / name).read_bytes().replace(b"\r\n", b"\n"))
    run_step(["bash", scripts / "bootstrap.sh", work], env, work, work_deadline)
    python = Path((work / "venv-path.txt").read_text().strip()) / "bin/python"

    def step(script, *params, stdout=None):
        run_step([python, source / script, *params], env, work, work_deadline, stdout)

    for name in ("requirements-resolved.txt", "unisign-revision.txt", "assets.sha256", "gpu.txt"):
        (output / name).write_bytes((work / name).read_bytes())
    if followup:
        run_step([python, source.parent / "phase3/followup.py", work, output], env, work, work_deadline)
        return

    def evaluate(mode, label, manifest, settings, *, bleu=False):
        events = output / f"{label}-events.jsonl"
        step("benchmark.py", mode, manifest, events, *settings)
        with (output / f"{label}-summary.json").open("w") as summary:
            step("score.py", events, "--manifest", manifest, *(["--bleu"] if bleu else []), stdout=summary)

    oracle_settings = ["--parity", "--beams", "4", "--tokens", "100"]
    replay_decode = ["--beams", "1", "--tokens", "48"]
    replay_settings = [*replay_decode, "--policy", "stable-punctuation", "--buffer-s", "8", "--cadence-ms", "500"]
    if dataset == "how2sign":
        official_manifest = work / "prepared/how2sign-oracle/oracle.jsonl"
        if not official_manifest.is_file():
            validation = output / "how2sign"
            step("how2sign_assets.py", validation, "--sample-labels", work / "Uni-Sign/data/How2Sign/labels.test")
            official_manifest = validation / "oracle.jsonl"
        oracle_rows = [json.loads(line) for line in official_manifest.read_text(encoding="utf-8").splitlines()
                       if line.strip()]
        if len(oracle_rows) != 32 or any(not row.get("reference") for row in oracle_rows):
            raise ValueError("The official baseline requires 32 labeled test clips")
        evaluate("oracle", "how2sign-oracle", official_manifest, oracle_settings, bleu=True)
        if continuous_manifest is None and official_continuous:
            continuous_cache = work / "continuous-cache"
            step("prepare_continuous.py", continuous_cache, work / "Uni-Sign")
            continuous_manifest = continuous_cache / "prepared/continuous.jsonl"
        if continuous_manifest is not None:
            paired = output / "paired-oracle"
            step("pose_data.py", "split-oracle", continuous_manifest, paired)
            evaluate("oracle", "continuous-oracle", paired / "oracle.jsonl", replay_decode, bleu=True)
            evaluate("replay", "continuous-replay", continuous_manifest, replay_settings)
        else:
            print("Continuous replay skipped: no timestamped continuous source manifest supplied.", flush=True)
        return

    poses = output / "poses"
    poses.mkdir()
    continuous, completed = [], []
    for video in videos:
        pose = poses / f"{video.stem}.npz"
        with (output / f"pose-{video.stem}.json").open("w") as log:
            step("pose_data.py", "video", work / "Uni-Sign", video, pose, stdout=log)
        row = {"id": video.stem, "pose": pose.relative_to(output).as_posix()}
        if video.stem.startswith("idle-"):
            row["idle_only"] = True
        else:
            completed.append(row)
        continuous.append(row)

    for mode, rows, settings in (
        ("oracle", completed, oracle_settings),
        ("replay", continuous, replay_settings),
    ):
        manifest = output / f"{mode}-inputs.jsonl"
        manifest.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
        evaluate(mode, mode, manifest, settings)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["run", "watchdog"])
    parser.add_argument("--pod", default=os.environ.get("RUNPOD_POD_ID"))
    parser.add_argument("--started", type=int, required=True, help="Unix timestamp captured before Deploy")
    parser.add_argument("--inputs", type=Path, default=Path("/workspace/signgpt-input"))
    parser.add_argument("--work", type=Path, default=Path("/workspace/signgpt-validation"))
    parser.add_argument("--dataset", choices=["webcam", "how2sign"], default="webcam")
    parser.add_argument("--continuous-manifest", type=Path, help="How2Sign: verified continuous track manifest")
    parser.add_argument("--official-continuous", action="store_true", help="Prepare official original recordings if needed")
    parser.add_argument("--phase3-followup", action="store_true", help="Run the staged paired audit and controller comparison")
    args = parser.parse_args()
    if (args.continuous_manifest is not None or args.official_continuous) and args.dataset != "how2sign":
        parser.error("Official continuous inputs require --dataset how2sign")
    if args.phase3_followup and args.dataset != "how2sign":
        parser.error("Phase 3 requires --dataset how2sign")
    key = os.environ.pop("RUNPOD_API_KEY", "")
    if not key or not re.fullmatch(r"[a-zA-Z0-9_-]+", args.pod or ""):
        parser.error("Set RUNPOD_API_KEY and supply a valid Pod ID")
    if args.started > time.time() + 5:
        parser.error("Deployment timestamp is in the future")
    info = verify_pod(args.pod, key)
    if args.mode == "watchdog":
        try:
            _, deadline = spending_limits(info, args.started)
        except (KeyError, TypeError, ValueError) as error:
            print(f"Budget check failed: {error}. Requesting Stop now.", flush=True)
            return 1 if stop_pod(args.pod, key) else 2
        if info.get("locked") or deadline <= time.time():
            print("Pod is locked or already at its deadline. Check it in the console now.", flush=True)
            return 0 if stop_pod(args.pod, key) else 2
        return watchdog(args, key, deadline)
    if args.pod != os.environ.get("RUNPOD_POD_ID") or sys.platform != "linux":
        parser.error("Run mode must target the current Linux RunPod Pod")

    os.umask(0o077)
    work = args.work.resolve()
    output = work / "results" / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    status = {"pod": args.pod, "started": utc_now(), "outcome": "running"}
    result = 1
    try:
        output.mkdir(parents=True)
        print(f"Results: {output}", flush=True)
        rate, deadline = spending_limits(info, args.started)
        if args.phase3_followup:
            deadline = min(deadline, args.started + 55 * 60)
        if info.get("locked") or "stop" not in info.get("actions", []):
            raise ValueError("RunPod reports that this Pod cannot be stopped in its current state")
        status["hourly_usd"] = rate
        status["budget_usd"] = MAX_BUDGET_USD
        status["non_gpu_reserve_usd"] = NON_GPU_RESERVE_USD
        status["shutdown_deadline_utc"] = datetime.fromtimestamp(deadline, timezone.utc).isoformat()
        manifest = args.continuous_manifest.resolve() if args.continuous_manifest else None
        staged_continuous = work / "prepared/continuous/continuous.jsonl"
        if args.dataset == "how2sign" and manifest is None and staged_continuous.is_file():
            manifest = staged_continuous
        if manifest is not None and not manifest.is_file():
            raise ValueError(f"Continuous manifest does not exist: {manifest}")
        validate(Path(__file__).resolve().parent, args.inputs.resolve(), work, output, deadline - 120,
                 dataset=args.dataset, continuous_manifest=manifest, official_continuous=args.official_continuous,
                 followup=args.phase3_followup)
        status["outcome"] = "completed"
        result = 0
    except BaseException as error:
        status.update(outcome="failed", error=str(error))
        print(f"Validation failed: {error}", flush=True)
    finally:
        try:
            if output.is_dir():
                status["finished"] = utc_now()
                status["shutdown"] = "requesting_stop"
                (output / "status.json").write_text(json.dumps(status, indent=2) + "\n", encoding="utf-8")
            os.sync()
        finally:
            if not stop_pod(args.pod, key):
                result = 2
    return result


if __name__ == "__main__":
    sys.exit(main())
