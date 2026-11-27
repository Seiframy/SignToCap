"""Prepare model-compatible, replayable pose tracks. Run in the Python 3.11 Pod environment."""

import argparse
import gzip
import json
import math
import pickle
import sys
from pathlib import Path
from time import perf_counter
from zipfile import ZipFile


def save_track(path, keypoints, scores, fps):
    import numpy as np

    keypoints = np.asarray(keypoints, dtype=np.float32)
    scores = np.asarray(scores, dtype=np.float32)
    if (
        keypoints.ndim != 4
        or keypoints.shape[1:] != (1, 133, 2)
        or scores.shape != keypoints.shape[:-1]
        or not len(scores)
        or (fps is not None and (not math.isfinite(fps) or fps <= 0))
        or not np.isfinite(keypoints).all()
        or not np.isfinite(scores).all()
    ):
        raise ValueError(f"Unexpected landmark data for {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    metadata = {} if fps is None else {"fps": float(fps)}
    np.savez_compressed(path, keypoints=keypoints, scores=scores, **metadata)


def load_track(path):
    import numpy as np

    with np.load(path, allow_pickle=False) as data:
        keypoints, scores = data["keypoints"], data["scores"]
        fps = float(data["fps"]) if "fps" in data else None
    if (
        keypoints.ndim != 4
        or keypoints.shape[1:] != (1, 133, 2)
        or scores.shape != keypoints.shape[:-1]
        or not len(keypoints)
        or (fps is not None and (not np.isfinite(fps) or fps <= 0))
        or not np.isfinite(keypoints).all()
        or not np.isfinite(scores).all()
    ):
        raise ValueError(f"Invalid COCO-WholeBody pose track: {path}")
    return keypoints, scores, fps


class PoseExtractor:
    """Pinned RTMW whole-body extraction shared by video audit and live capture."""

    def __init__(self, upstream, *, ort_threads=0, device="cpu", profile_prefix=None):
        sys.path.insert(0, str(upstream.resolve() / "demo" / "rtmlib-main"))
        from rtmlib import Wholebody

        self.detector = (self.configured_detector(Wholebody, ort_threads, device, profile_prefix)
                         if ort_threads or device == "cuda" or profile_prefix else
                         Wholebody(to_openpose=False, mode="lightweight", backend="onnxruntime", device="cpu"))
        self.previous_center = None

    @staticmethod
    def configured_detector(wholebody, threads, device, profile_prefix):
        import onnxruntime as ort
        from rtmlib import RTMPose, YOLOX
        from rtmlib.tools.base import BaseTool
        from rtmlib.tools.file import download_checkpoint
        provider = "CUDAExecutionProvider" if device == "cuda" else "CPUExecutionProvider"
        if provider not in ort.get_available_providers():
            raise RuntimeError(f"{provider} is unavailable")
        providers = [provider, "CPUExecutionProvider"] if device == "cuda" else [provider]

        class ConfiguredSession(BaseTool):
            def __init__(self, onnx_model, model_input_size, mean=None, std=None,
                         backend="onnxruntime", device="cpu"):
                if not Path(onnx_model).is_file():
                    onnx_model = download_checkpoint(onnx_model)
                options = ort.SessionOptions()
                options.intra_op_num_threads = threads
                if threads:
                    options.add_session_config_entry("session.intra_op.allow_spinning", "0")
                if profile_prefix:
                    options.enable_profiling = True
                    options.profile_file_prefix = str(profile_prefix) + "-" + type(self).__name__
                self.session = ort.InferenceSession(onnx_model, sess_options=options,
                                                    providers=providers)
                if self.session.get_providers()[0] != provider:
                    raise RuntimeError(f"{provider} did not initialize")
                if device == "cuda":
                    self.session.disable_fallback()
                self.onnx_model, self.model_input_size = onnx_model, model_input_size
                self.mean, self.std = mean, std
                self.backend, self.device = backend, device

        class Detector(YOLOX, ConfiguredSession):
            pass

        class Pose(RTMPose, ConfiguredSession):
            pass

        class ConfiguredWholebody(wholebody):
            def __init__(self):
                config = self.MODE["lightweight"]
                self.det_model = Detector(config["det"], model_input_size=config["det_input_size"], device=device)
                self.pose_model = Pose(config["pose"], model_input_size=config["pose_input_size"], device=device)

        return ConfiguredWholebody()

    def cuda_operators(self):
        from collections import Counter

        result = {}
        for name, model in (("detector", self.detector.det_model), ("pose", self.detector.pose_model)):
            if model.session.get_providers()[0] != "CUDAExecutionProvider":
                raise RuntimeError(f"{name} CUDA provider did not initialize")
            events = json.loads(Path(model.session.end_profiling()).read_text(encoding="utf-8"))
            counts = Counter(event["args"]["provider"] for event in events
                             if event.get("cat") == "Node" and event.get("args", {}).get("provider"))
            if not counts["CUDAExecutionProvider"]:
                raise RuntimeError(f"{name} executed no CUDA operators")
            result[name] = dict(counts)
        return result

    def extract(self, frame):
        import numpy as np

        height, width = frame.shape[:2]
        joints, confidence = self.detector(frame)
        joints = np.asarray(joints, dtype=np.float32).reshape(-1, 133, 2)
        confidence = np.asarray(confidence, dtype=np.float32).reshape(-1, 133)
        if not len(joints):
            return np.zeros((1, 133, 2), dtype=np.float32), np.zeros((1, 133), dtype=np.float32)
        joints = joints / np.array([width, height], dtype=np.float32)
        upper_body = confidence[:, :11].mean(axis=1)
        centers = joints[:, [5, 6]].mean(axis=1)
        if self.previous_center is not None and len(joints) > 1:
            upper_body -= 2 * np.linalg.norm(centers - self.previous_center, axis=1)
        choice = int(upper_body.argmax())
        self.previous_center = centers[choice]
        return joints[choice:choice + 1], confidence[choice:choice + 1]


def extract_video(args, *, extractor=None):
    import cv2
    import numpy as np

    if extractor is None:
        extractor = PoseExtractor(args.upstream, ort_threads=getattr(args, "ort_threads", 0))
    capture = cv2.VideoCapture(str(args.video))
    try:
        fps = capture.get(cv2.CAP_PROP_FPS)
        if not capture.isOpened() or not np.isfinite(fps) or fps <= 0:
            raise ValueError(f"Cannot open a constant-frame-rate video: {args.video}")
        first = math.floor(args.start_ms * fps / 1000)
        last = None if args.end_ms is None else math.ceil(args.end_ms * fps / 1000)
        if first < 0 or (last is not None and last <= first):
            raise ValueError("Invalid video crop")
        if first and (not capture.set(cv2.CAP_PROP_POS_FRAMES, first)
                      or round(capture.get(cv2.CAP_PROP_POS_FRAMES)) != first):
            raise ValueError("Could not seek to the requested source frame")
        keypoints, scores, times = [], [], []
        cadence_error = 0.0
        while last is None or first + len(times) < last:
            ok, frame = capture.read()
            if not ok:
                break
            if args.verify_cadence:
                observed = capture.get(cv2.CAP_PROP_POS_MSEC)
                expected = (first + len(times)) * 1000 / fps
                if not math.isfinite(observed):
                    raise ValueError("Missing decoded media timestamp")
                cadence_error = max(cadence_error, abs(observed - expected))
            started = perf_counter()
            joints, confidence = extractor.extract(frame)
            times.append((perf_counter() - started) * 1000)
            keypoints.append(joints)
            scores.append(confidence)
    finally:
        capture.release()
    if not times:
        raise ValueError(f"No decoded frames: {args.video}")
    if last is not None and len(times) != last - first:
        raise ValueError(f"Incomplete video crop: {args.video}")
    if args.verify_cadence and cadence_error > 2:
        raise ValueError(f"Decoded timestamps differ from fixed cadence by {cadence_error:.2f} ms")
    save_track(args.output, keypoints, scores, fps)
    warm = times[min(5, len(times) - 1) :]
    metrics = {
        "frames": len(times), "source_fps": fps, "pose_first_ms": times[0],
        "pose_p50_ms": float(np.median(warm)),
        "pose_p95_ms": float(np.percentile(warm, 95)),
        "pose_work_fps": 1000 / float(np.mean(warm)),
        "source_start_ms": first * 1000 / fps,
        "cadence_max_error_ms": cadence_error if args.verify_cadence else None,
    }
    print(json.dumps(metrics), flush=True)
    return metrics


def convert_oracle(args):
    if args.pose_root.is_dir():
        convert_oracle_members(
            args, (path.relative_to(args.pose_root).as_posix() for path in args.pose_root.rglob("*.pkl")),
            lambda name: (args.pose_root / name).open("rb"),
        )
    else:
        with ZipFile(args.pose_root) as archive:
            convert_oracle_members(args, archive.namelist(), archive.open)


def convert_oracle_members(args, names, open_pose):
    import numpy as np

    # The pinned author-supplied labels and pose files are trusted pickle inputs.
    with gzip.open(args.labels, "rb") as source:
        annotations = pickle.load(source)
    by_name = {}
    paths = set()
    for name in names:
        if name.endswith(".pkl"):
            paths.add(name)
            by_name.setdefault(name.rsplit("/", 1)[-1], []).append(name)
    missing = []
    written = 0
    ordered = sorted(annotations, key=str)
    np.random.default_rng(42).shuffle(ordered)
    args.output.mkdir(parents=True, exist_ok=True)
    manifest = args.output / "oracle.jsonl"
    temporary = manifest.with_suffix(".tmp")
    with temporary.open("w", encoding="utf-8") as out:
        for key in ordered[: args.limit]:
            row = annotations[key]
            relative = Path(row["video_path"].replace("\\", "/")).with_suffix(".pkl").as_posix()
            matches = by_name.get(relative.rsplit("/", 1)[-1], [])
            path = relative if relative in paths else matches[0] if len(matches) == 1 else None
            if path is None:
                missing.append(str(relative))
                continue
            with open_pose(path) as f:
                data = pickle.load(f)
            start, end = int(data.get("start", 0)), int(data.get("end", len(data["scores"])))
            if not 0 <= start < end <= min(len(data["scores"]), len(data["keypoints"])):
                raise ValueError(f"Invalid pose range: {path}")
            pose = f"clip-{written:04d}.npz"
            save_track(args.output / pose, data["keypoints"][start:end], data["scores"][start:end], args.fps)
            print(json.dumps({"id": str(key), "pose": pose, "reference": row["text"]}), file=out)
            written += 1
    if missing or written != args.limit:
        raise ValueError(f"Prepared {written}/{args.limit}. Missing or ambiguous paths: {missing[:10]}")
    temporary.replace(manifest)
    print(manifest)


def split_oracle(args):
    rows = [json.loads(line) for line in args.manifest.read_text(encoding="utf-8").splitlines() if line.strip()]
    args.output.mkdir(parents=True, exist_ok=True)
    manifest = args.output / "oracle.jsonl"
    temporary = manifest.with_suffix(".tmp")
    ids = set()
    with temporary.open("w", encoding="utf-8") as out:
        for row in rows:
            keypoints, scores, fps = load_track(args.manifest.parent / row["pose"])
            if fps is None:
                raise ValueError(f"Missing source FPS for {row['id']}")
            for segment in row.get("segments", []):
                key = row["id"], segment["item"]
                start = float(segment["start_ms"])
                end = float(segment["end_ms"])
                if key in ids or not 0 <= start < end <= len(scores) / fps * 1000:
                    raise ValueError(f"Invalid or duplicate oracle segment: {key}")
                left, right = round(start * fps / 1000), round(end * fps / 1000)
                pose = f"clip-{len(ids):04d}.npz"
                save_track(args.output / pose, keypoints[left:right], scores[left:right], fps)
                ids.add(key)
                print(json.dumps({
                    "id": f"{row['id']}/{segment['item']}", "pose": pose, "reference": segment["reference"],
                }), file=out)
    if not ids:
        raise ValueError("Continuous manifest includes no oracle segments")
    temporary.replace(manifest)
    print(manifest)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(required=True)
    video = commands.add_parser("video")
    video.add_argument("upstream", type=Path)
    video.add_argument("video", type=Path)
    video.add_argument("output", type=Path)
    video.add_argument("--start-ms", type=float, default=0)
    video.add_argument("--end-ms", type=float)
    video.add_argument("--verify-cadence", action="store_true")
    video.add_argument("--ort-threads", type=int, choices=range(0, 9), default=0,
                       help="ONNX Runtime CPU threads per model. 0 uses the runtime defaults")
    video.set_defaults(run=extract_video)
    oracle = commands.add_parser("oracle")
    oracle.add_argument("labels", type=Path)
    oracle.add_argument("pose_root", type=Path)
    oracle.add_argument("output", type=Path)
    oracle.add_argument("--fps", type=float, help="Supply only if verified from source metadata")
    oracle.add_argument("--limit", type=int, default=32)
    oracle.set_defaults(run=convert_oracle)
    split = commands.add_parser("split-oracle")
    split.add_argument("manifest", type=Path)
    split.add_argument("output", type=Path)
    split.set_defaults(run=split_oracle)
    args = parser.parse_args()
    args.run(args)


if __name__ == "__main__":
    main()
