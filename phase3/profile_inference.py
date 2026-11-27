"""Compare pose providers and a frozen Uni-Sign decode on the same complete video window."""
import argparse
from collections import Counter
import json
import os
from pathlib import Path
import statistics
import subprocess
from tempfile import TemporaryDirectory, TemporaryFile
from time import perf_counter
from types import SimpleNamespace

from desktop import UniSignBackend
from pose_data import PoseExtractor, extract_video, load_track
from unisign import Translator


def cuda_assignment(upstream, video, first_ms, directory):
    import cv2

    probed = PoseExtractor(upstream, device="cuda", profile_prefix=directory / "ort")
    capture = cv2.VideoCapture(str(video))
    try:
        fps = capture.get(cv2.CAP_PROP_FPS)
        if not capture.isOpened() or fps <= 0:
            raise ValueError("Could not read the CUDA probe video")
        capture.set(cv2.CAP_PROP_POS_FRAMES, int(first_ms * fps / 1000))
        ok, frame = capture.read()
    finally:
        capture.release()
    if not ok:
        raise ValueError("Could not decode the CUDA probe frame")
    probed.extract(frame)
    assignment = {}
    for kind, component in (("detection", probed.detector.det_model), ("pose", probed.detector.pose_model)):
        report = Path(component.session.end_profiling())
        events = json.loads(report.read_text(encoding="utf-8"))
        counts = Counter(row["args"]["provider"] for row in events if row.get("args", {}).get("provider"))
        if not counts["CUDAExecutionProvider"]:
            raise RuntimeError(f"{kind} ran without any CUDA operators")
        assignment[kind] = dict(counts)
    return assignment


def benchmark(args):
    import numpy as np
    import torch

    upstream = args.work / "Uni-Sign"
    assets = args.work / "assets"
    os.environ["TORCH_HOME"] = str(args.pose_cache or args.work / "cache/rtmlib")
    os.environ["HF_HUB_OFFLINE"] = "1"
    started = perf_counter()
    backend = UniSignBackend.__new__(UniSignBackend)
    backend.model = Translator(upstream, assets / "mt5-base", assets / "how2sign_pose_only_slt.pth")
    print(json.dumps({"kind": "checkpoint", "gpu": torch.cuda.get_device_name(),
                      "model_load_ms": 1000 * (perf_counter() - started)}), flush=True)
    baseline = None
    with TemporaryDirectory(prefix="SignToCap-profile-") as folder:
        directory = Path(folder)
        for case in args.cases:
            try:
                assignment = cuda_assignment(upstream, args.video, args.start_ms, directory) if case == "cuda" else None
                pose_started = perf_counter()
                extractor = PoseExtractor(upstream, ort_threads=2 if case == "cpu2" else 0,
                                          device="cuda" if case == "cuda" else "cpu")
                pose_load_ms = 1000 * (perf_counter() - pose_started)
            except (RuntimeError, ValueError) as error:
                if case != "cuda":
                    raise
                print(json.dumps({"kind": "cuda_unavailable", "reason": str(error)}), flush=True)
                continue
            output = directory / "poses.npz"
            with TemporaryFile(mode="w+t") as readings:
                monitor = subprocess.Popen(
                    ["nvidia-smi", "--query-gpu=utilization.gpu", "--format=csv,noheader,nounits", "-lms", "250"],
                    stdout=readings, stderr=subprocess.DEVNULL,
                )
                try:
                    started = perf_counter()
                    pose_stats = extract_video(SimpleNamespace(
                        upstream=upstream, video=args.video, output=output, start_ms=args.start_ms,
                        end_ms=args.end_ms, verify_cadence=True), extractor=extractor)
                    extraction_ms = 1000 * (perf_counter() - started)
                    first = backend.translate_recorded(output)
                    total_ms = 1000 * (perf_counter() - started)
                    warm = backend.translate_recorded(output)
                finally:
                    monitor.terminate()
                    monitor.wait(timeout=5)
                readings.seek(0)
                usage = [int(line) for raw in readings if (line := raw.strip()).isdigit()]
            track = load_track(output)
            if baseline is None:
                baseline = track
            delta = [float(np.max(np.abs(old - new))) for old, new in zip(baseline[:2], track[:2])]
            print(json.dumps({"kind": "comparison", "case": case, "pose": pose_stats, "ort_operators": assignment,
                              "pose_load_ms": pose_load_ms, "video_to_poses_ms": extraction_ms,
                              "video_to_caption_ms": total_ms, "first": first, "warm": warm,
                              "delta_xy": delta[0], "delta_scores": delta[1],
                              "gpu_samples": len(usage), "gpu_percent_p50": statistics.median(usage) if usage else None,
                              "gpu_percent_max": max(usage) if usage else None}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("work", type=Path)
    parser.add_argument("video", type=Path)
    parser.add_argument("start_ms", type=float)
    parser.add_argument("end_ms", type=float)
    parser.add_argument("--pose-cache", type=Path)
    parser.add_argument("--cases", choices=["baseline", "cpu2", "cuda"], nargs="+",
                        default=["baseline", "cpu2", "cuda"])
    benchmark(parser.parse_args())
