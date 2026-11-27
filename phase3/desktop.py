"""Single-signer loopback caption prototype. Default mode plays measured Phase 2 events."""

import argparse
import base64
import json
import math
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import os
import secrets
import subprocess
import time
from threading import Lock
from tempfile import TemporaryDirectory
from pathlib import Path
import re
import sys
from types import SimpleNamespace


HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "phase2"))
from captions import AdaptiveStreamingSession, activity_features, repeated_phrase

BATCH_FRAMES = 3
SENTENCE_FPS = 6
SENTENCE_FRAMES = 48
SENTENCE_RATES = (6, 12, 24)
UPLOAD_LIMIT = 64 * 1024**2
MAX_JPEG_TEXT = 250_000
SLOT_EPSILON_MS = .001


def pose_stage():
    return Path(os.environ.get("SIGNTOCAP_POSE_STAGE", Path(os.environ.get("LOCALAPPDATA", "."))
                               / "Temp/signgpt-phase2-stage"))


def video_tool(*args, timeout=180):
    local = pose_stage()
    python = local / "cpu-venv/Scripts/python.exe"
    if not python.is_file():
        raise ValueError("Install the local SignToCap pose runtime to inspect or translate uploads")
    env = dict(os.environ, TORCH_HOME=str(local / "ort-cache/rtmlib"))
    env.pop("PYTHONPATH", None)
    env.pop("PYTHONHOME", None)
    result = subprocess.run([str(python), "-B", str(HERE.parent / "phase2/pose_data.py"), *map(str, args)],
                            env=env, capture_output=True, timeout=timeout)
    if result.returncode:
        raise ValueError("Video decoding or pose extraction failed. Use a complete, constant-frame-rate clip with one visible signer.")
    return json.loads(result.stdout.splitlines()[-1])


def frame_requests(body):
    rows = body.get("frames")
    if not isinstance(rows, list) or not 1 <= len(rows) <= BATCH_FRAMES:
        raise ValueError(f"Expected 1 to {BATCH_FRAMES} frames")
    last_slot, last_ms = -1, -1
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError("Invalid frame")
        slot, at_ms, jpeg = row.get("slot"), row.get("at_ms"), row.get("jpeg")
        if (type(slot) is not int or not last_slot < slot <= 1_000_000
                or type(at_ms) not in (int, float) or not 0 <= at_ms <= 60_000 or at_ms <= last_ms
                or not isinstance(jpeg, str) or not 0 < len(jpeg) <= MAX_JPEG_TEXT):
            raise ValueError("Expected ordered frames with valid capture timestamps")
        last_slot, last_ms = slot, at_ms
    return rows


class UniSignBackend:
    fps = 3  # Deliberate laptop/CPU demo sampling rate. Its translation quality needs validation.

    def __init__(self, work, *, pose_device="cpu", cuda_reason=None, warmup_path=None):
        from pose_data import PoseExtractor
        from unisign import Translator

        began = time.perf_counter()
        verification = None
        warm_frame = None
        if warmup_path is not None:
            import cv2
            warm_frame = cv2.imread(str(warmup_path))
            if warm_frame is None:
                raise ValueError("Could not read the warm-up frame")
        if pose_device == "cuda":
            try:
                with TemporaryDirectory(prefix="SignToCap-ort-") as folder:
                    self.pose = PoseExtractor(work / "Uni-Sign", device="cuda",
                                              profile_prefix=Path(folder) / "verify")
                    self.pose.extract(warm_frame)
                    verification = self.pose.cuda_operators()
            except (OSError, ValueError, RuntimeError, ImportError):
                cuda_reason = "CUDA operator verification failed"
                pose_device = "cpu"
        if pose_device == "cpu":
            self.pose = PoseExtractor(work / "Uni-Sign", ort_threads=2)
        pose_loaded = time.perf_counter()
        self.model = Translator(work / "Uni-Sign", work / "assets/mt5-base",
                                work / "assets/how2sign_pose_only_slt.pth")
        model_loaded = time.perf_counter()
        self.runtime = {"pose_provider": "CUDAExecutionProvider" if verification else "CPUExecutionProvider",
                        "pose_operators": verification, "pose_fallback": cuda_reason,
                        "batch_frames": BATCH_FRAMES, "sentence_fps": SENTENCE_FPS,
                        "sentence_frames": SENTENCE_FRAMES,
                        "sentence_rates": SENTENCE_RATES,
                        "pose_load_ms": (pose_loaded - began) * 1000,
                        "model_load_ms": (model_loaded - pose_loaded) * 1000}
        if warm_frame is not None:
            self.pose.previous_center = None
            points, scores = self.pose.extract(warm_frame)
            self.model.translate(points[None], scores[None], tokens=8)
            self.pose.previous_center = None
            self.runtime["warmup_ms"] = round((time.perf_counter() - model_loaded) * 1000, 1)
        self.session = None

    def start(self, mode="live", sample_fps=None):
        if mode not in {"live", "sentence"} or (sample_fps is not None and
                (mode != "sentence" or type(sample_fps) is not int or sample_fps not in SENTENCE_RATES)):
            raise ValueError("Unknown capture mode")
        options = SimpleNamespace(buffer_s=10, decode_interval_ms=900, partial_tokens=24,
                                  beams=1, tokens=48, motion=.7, policy="explicit")
        self.capture_mode = mode
        self.capture_fps = (sample_fps or SENTENCE_FPS) if mode == "sentence" else self.fps
        self.session = AdaptiveStreamingSession(options, self.model, self.capture_fps)
        self.pose.previous_center = None
        self.last_slot = -1
        self.last_frame_ms = -1
        self.dropped = 0
        self.sentence_pose_missing = False
        self.sentence_points, self.sentence_scores = [], []

    def frame(self, jpeg, slot, at_ms=None):
        import cv2
        import numpy as np

        if (type(slot) is not int or not self.last_slot < slot <= 1_000_000
                or (at_ms is not None and (type(at_ms) not in (int, float)
                                          or not 0 <= at_ms <= 60_000 or at_ms <= self.last_frame_ms
                                          or math.floor((at_ms + SLOT_EPSILON_MS) * self.capture_fps / 1000) != slot))):
            raise ValueError("Frame timestamps must increase")
        if self.capture_mode == "sentence" and (at_ms is None or slot >= 8 * self.capture_fps):
            raise ValueError("Sentence capture exceeds the supported window")
        before_jpeg = time.perf_counter()
        image = cv2.imdecode(np.frombuffer(jpeg, dtype=np.uint8), cv2.IMREAD_COLOR)
        if image is None or max(image.shape[:2]) > 1280:
            raise ValueError("Expected a valid image up to 1280 pixels")
        began = time.perf_counter()
        self.jpeg_ms = (began - before_jpeg) * 1000
        points, scores = self.pose.extract(image)
        self.pose_ms = (time.perf_counter() - began) * 1000
        # Advance the source clock without feeding synthetic observations to Uni-Sign.
        missing = slot - self.last_slot - 1
        self.dropped += missing
        self.last_slot = slot
        if at_ms is not None:
            self.last_frame_ms = at_ms
        if self.capture_mode == "sentence":
            if not np.any(scores > 0):
                self.sentence_pose_missing = True
            if not self.dropped:
                if self.sentence_pose_missing:
                    return {"kind": "uncertain", "reason": "insufficient_pose"}
                self.sentence_points.append(points)
                self.sentence_scores.append(scores)
                return None
            return {"kind": "uncertain", "reason": "capture_gap"}
        if missing:
            self.session.reset_caption()
            self.session.seen += missing
            self.session.keypoints = self.session.scores = self.session.motion = self.session.visible = None
            self.session.push(points[None], scores[None], decode=False)
            return self.session.event("uncertain", reason="capture_gap")
        return self.session.push(points[None], scores[None])

    def frames(self, rows):
        results = []
        for row in rows:
            began = time.perf_counter()
            event = self.frame(base64.b64decode(row["jpeg"], validate=True), row["slot"], row["at_ms"])
            results.append(dict(slot=row["slot"], at_ms=row["at_ms"], event=event,
                                pose_ms=self.pose_ms, jpeg_ms=self.jpeg_ms, dropped=self.dropped,
                                worker_ms=(time.perf_counter() - began) * 1000))
        return results

    def translate_recorded(self, track):
        from pose_data import load_track

        points, scores, fps = load_track(track)
        if fps is None or not 0 < fps <= 60 or not max(2, round(.5 * fps)) <= len(points) <= 256:
            raise ValueError("The prerecorded sentence must be complete and fit the Uni-Sign window")
        return self.translate_window(points, scores, fps)

    def translate_window(self, points, scores, fps):
        _, visible = activity_features(points, scores, fps)
        if visible.mean() < .2:
            return {"kind": "uncertain", "reason": "insufficient_pose", "frames": len(points), "source_fps": fps}
        text, _, timing = self.model.translate(points, scores)
        if not text or repeated_phrase(text):
            return {"kind": "uncertain", "reason": "empty_or_repeated_text", "frames": len(points), "source_fps": fps, **timing}
        return {"kind": "final", "text": text, "frames": len(points), "source_fps": fps, **timing}

    def commit(self):
        if self.capture_mode == "sentence":
            import numpy as np

            if self.dropped:
                return {"kind": "uncertain", "reason": "capture_gap"}
            if self.sentence_pose_missing:
                return {"kind": "uncertain", "reason": "insufficient_pose"}
            if len(self.sentence_points) < max(2, round(.5 * self.capture_fps)):
                return {"kind": "uncertain", "reason": "no_complete_window"}
            return self.translate_window(np.asarray(self.sentence_points), np.asarray(self.sentence_scores),
                                         self.capture_fps)
        return self.session.commit()

    def stop(self):
        event = self.session.finish()
        self.session = None
        self.sentence_points.clear()
        self.sentence_scores.clear()
        return event


class PreviewServer(ThreadingHTTPServer):
    def __init__(self, port, event_path, backend=None, manager=None):
        super().__init__(("127.0.0.1", port), Handler)
        self.backend = backend
        self.manager = manager
        self.serial = Lock()
        self.recorded_job = Lock()
        self.token = secrets.token_urlsafe(32)
        self.catalog = {}
        self.media = {}
        local = Path(os.environ.get("LOCALAPPDATA", "."))
        directory = Path(os.environ.get("SIGNTOCAP_MEDIA", str(local / "SignToCap/media")))
        manifest = HERE.parent / "phase2-analysis/continuous-inputs.jsonl"
        if manifest.exists():
            for line in manifest.read_text(encoding="utf-8").splitlines():
                row = json.loads(line)
                name = row["id"]
                video = directory / (name + ".mp4")
                if not video.is_file() and not os.environ.get("SIGNTOCAP_MEDIA"):
                    video = local / "Temp/signgpt-phase2-stage/continuous" / video.name
                self.catalog[name] = dict(source=row["source"], segments=row["segments"], available=video.is_file())
                if video.is_file():
                    self.media[name] = video
        self.tracks = {}
        for line in event_path.read_text(encoding="utf-8").splitlines():
            row = json.loads(line)
            if row["kind"] in ("partial", "final", "uncertain"):
                self.tracks.setdefault(row["id"], []).append(row)


class Handler(BaseHTTPRequestHandler):
    def send(self, status, data, content_type="application/json"):
        if content_type == "application/json":
            data = json.dumps(data).encode()
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.end_headers()
        self.wfile.write(data)

    def local_host(self):
        return self.headers.get("Host") in {f"localhost:{self.server.server_port}",
                                            f"127.0.0.1:{self.server.server_port}"}

    def upload_video(self, origin):
        if (origin != "http://" + self.headers["Host"] or self.headers.get("X-Control") != self.server.token
                or not self.headers.get("X-Session")):
            return self.send(403, {"error": "Local session authorization required"})
        size = int(self.headers.get("Content-Length", "0"))
        if not 0 < size <= UPLOAD_LIMIT:
            return self.send(413, {"error": "Select a video no larger than 64 MiB"})
        name = self.headers.get("X-Video-Extension", "").lower()
        if not re.fullmatch(r"\.[a-z0-9]{1,8}", name):
            return self.send(400, {"error": "Choose a video file with a valid file extension"})
        translate = self.path == "/api/upload-translate"
        owner = self.headers["X-Session"]
        manager = self.server.manager
        rate = None
        if translate:
            rate = int(self.headers.get("X-Sample-Fps", "0"))
            if rate not in SENTENCE_RATES:
                raise ValueError("Choose 6, 12, or 24 FPS")
            if not manager or manager.owner != owner or not manager.status(owner)["ready"] or manager.live:
                raise ValueError("Attach and prepare the GPU in this tab before translating")
            generation = manager.generation
        if not self.server.recorded_job.acquire(blocking=False):
            return self.send(409, {"error": "A video job is already in progress"})
        streaming = False
        started = time.perf_counter()

        def emit(**data):
            self.wfile.write(json.dumps(data).encode() + b"\n")
            self.wfile.flush()

        try:
            with TemporaryDirectory(prefix="SignToCap-upload-") as folder:
                video = Path(folder) / ("video" + name)
                timeout = self.connection.gettimeout()
                self.connection.settimeout(10)
                try:
                    with video.open("wb") as out:
                        remaining = size
                        while remaining:
                            block = self.rfile.read1(min(remaining, 256 * 1024))
                            if not block or time.perf_counter() - started > 30:
                                raise ValueError("Video upload was incomplete or timed out")
                            out.write(block)
                            remaining -= len(block)
                finally:
                    self.connection.settimeout(timeout)
                metadata = video_tool("probe", video, timeout=15)
                if not translate:
                    return self.send(200, {"video": metadata})
                fps = metadata["source_fps"]
                if rate > fps + .05:
                    raise ValueError("Choose a sampling rate no higher than the source FPS")
                if metadata["duration_ms"] > 60_000 or math.ceil(metadata["source_frames"] * min(rate, fps) / fps) > 256:
                    raise ValueError("Choose a shorter complete sentence or a lower rate. Limit: 256 sampled frames and 60 seconds.")
                self.send_response(200)
                self.send_header("Content-Type", "application/x-ndjson")
                self.send_header("Cache-Control", "no-store")
                self.send_header("X-Content-Type-Options", "nosniff")
                self.end_headers()
                streaming = True
                emit(stage="extracting", video=metadata)
                pose = Path(folder) / "sentence.npz"
                metrics = video_tool("video", pose_stage() / "Uni-Sign", video, pose, "--sample-fps", rate,
                                     "--verify-cadence", "--whole-sentence", "--ort-threads", 2)
                emit(stage="translating")
                payload = base64.b64encode(pose.read_bytes()).decode("ascii")
                result = manager.infer(owner, "prerecorded", {"poses": payload, "generation": generation})
                if result.get("error"):
                    raise ValueError(result["error"])
                if not isinstance(result.get("event"), dict):
                    raise ValueError("The guarded worker returned no translation result")
                emit(stage="complete", event=result["event"], pose=metrics,
                     elapsed_ms=(time.perf_counter() - started) * 1000)
        except (BrokenPipeError, ConnectionResetError):
            return
        except (ValueError, OSError, subprocess.SubprocessError, RuntimeError) as error:
            detail = str(error) if isinstance(error, ValueError) else "Video processing failed or timed out"
            if streaming:
                try:
                    emit(error=detail)
                except ConnectionError:
                    pass
            else:
                return self.send(400, {"error": detail})
        finally:
            self.server.recorded_job.release()

    def translate_video(self, owner, body):
        manager = self.server.manager
        gpu = manager.status(owner)
        if not gpu["ready"] or gpu["active"]:
            raise ValueError("Prepare the GPU before translating, and end any active webcam test first")
        generation = manager.generation
        track, index = body.get("track"), body.get("segment")
        if not isinstance(track, str) or track not in self.server.media or type(index) is not int:
            raise ValueError("Select an available prerecorded sentence")
        entry = self.server.catalog[track]
        if not 0 <= index < len(entry["segments"]):
            raise ValueError("Select an annotated sentence")
        source, segment = entry["source"], entry["segments"][index]
        start = source["source_start_ms"] + segment["start_ms"]
        end = source["source_start_ms"] + segment["end_ms"]
        fps = source["source_fps"]
        if math.ceil(end * fps / 1000) - math.floor(start * fps / 1000) > 256:
            raise ValueError("Choose a complete shorter sentence. Uni-Sign accepts up to 256 source frames")
        if not self.server.recorded_job.acquire(blocking=False):
            raise ValueError("A prerecorded sentence is already being processed")
        try:
            with TemporaryDirectory(prefix="SignToCap-poses-") as folder:
                poses = Path(folder) / "sentence.npz"
                try:
                    pose_metrics = video_tool("video", pose_stage() / "Uni-Sign", self.server.media[track], poses,
                                              "--start-ms", start, "--end-ms", end, "--verify-cadence", "--ort-threads", 2)
                except (subprocess.SubprocessError, OSError):
                    raise ValueError("Could not extract a complete, time-verified sentence from the local video") from None
                payload = base64.b64encode(poses.read_bytes()).decode("ascii")
            translated = manager.infer(owner, "prerecorded", {"poses": payload, "generation": generation})
            return dict(translated, pose=pose_metrics, reference=segment["reference"])
        finally:
            self.server.recorded_job.release()

    def do_GET(self):
        if not self.local_host():
            return self.send(403, {"error": "Use the local address"})
        static = {"/": ("index.html", "text/html; charset=utf-8"), "/app.js": ("app.js", "text/javascript"),
                  "/style.css": ("style.css", "text/css")}
        static["/evidence.js"] = ("evidence.js","text/javascript")
        if self.path.startswith("/media/") and self.path[7:] in self.server.media:
            return self.video(self.server.media[self.path[7:]])
        if self.path == "/api/catalog":
            return self.send(200, self.server.catalog)
        if self.path in static:
            name, mime = static[self.path]
            return self.send(200, (HERE / "ui" / name).read_bytes(), mime)
        if self.path == "/api/status":
            manager = self.server.manager
            owner = self.headers.get("X-Session") if self.headers.get("X-Control") == self.server.token else None
            gpu = manager.status(owner) if manager else None
            return self.send(200, {"live": gpu["ready"] if gpu else self.server.backend is not None,
                                   "fps":3, "gpu":gpu, "token":self.server.token})
        if self.path == "/api/replay":
            return self.send(200, self.server.tracks)
        return self.send(404, {"error": "Unknown endpoint"})

    def video(self, path):
        size = path.stat().st_size
        start, end = 0, size-1
        requested = self.headers.get("Range")
        if requested:
            match = re.fullmatch(r"bytes=(\d+)-(\d*)", requested)
            if not match:
                return self.send(416, {"error":"Unsupported range"})
            start = int(match[1])
            end = min(end, int(match[2])) if match[2] else end
            if start > end:
                return self.send(416, {"error":"Range outside video"})
        self.send_response(206 if requested else 200)
        self.send_header("Content-Type","video/mp4")
        self.send_header("Accept-Ranges","bytes")
        self.send_header("Content-Length",str(end-start+1))
        if requested:
            self.send_header("Content-Range",f"bytes {start}-{end}/{size}")
        self.end_headers()
        with path.open("rb") as stream:
            stream.seek(start)
            remaining = end-start+1
            try:
                while remaining:
                    block = stream.read(min(remaining,65536))
                    if not block:
                        break
                    self.wfile.write(block)
                    remaining -= len(block)
            except (ConnectionError, OSError):
                pass

    def do_POST(self):
        origin = self.headers.get("Origin")
        if not self.local_host() or (origin and origin != "http://" + self.headers["Host"]):
            return self.send(403, {"error": "Use the local app origin"})
        backend = self.server.backend
        manager = self.server.manager
        if self.path in {"/api/upload-preview", "/api/upload-translate"}:
            try:
                return self.upload_video(origin)
            except (ValueError, TypeError) as error:
                return self.send(400, {"error": str(error)})
        if not backend and not manager:
            return self.send(409, {"error": "Live backend unavailable. Recorded replay is available."})
        try:
            size = int(self.headers.get("Content-Length", "0"))
            if not 0 <= size <= 1_000_000:
                return self.send(413, {"error": "Frame too large"})
            body = json.loads(self.rfile.read(size) or b"{}")
            if not isinstance(body, dict):
                return self.send(400, {"error": "Expected a JSON object"})
            if manager:
                if (origin != "http://" + self.headers["Host"] or
                        self.headers.get("X-Control") != self.server.token):
                    return self.send(403, {"error":"Local session authorization required"})
                owner = self.headers.get("X-Session","")
                if self.path == "/api/gpu/credentials":
                    manager.credentials(body)
                elif self.path == "/api/gpu/discover":
                    manager.discover(body)
                elif self.path == "/api/gpu/start":
                    if body.get("status") != "RUNNING":
                        from live_guard import PreflightError
                        raise PreflightError("Start the Pod in the RunPod console, then attach its running generation here")
                    manager.begin(owner, body)
                elif self.path == "/api/gpu/pin":
                    manager.submission(owner, body)
                elif self.path == "/api/gpu/stop":
                    reason = {"test_deadline": "60-second browser test deadline",
                              "connection": "Browser lost the desktop connection",
                              "frame_failed": "Camera frame or inference request failed",
                              "finish_failed": "Live test finish did not respond"}.get(body.get("reason"))
                    options = {"manual": True} if body.get("console_confirmed") is True else {}
                    manager.stop(reason, **options) if reason else manager.stop(**options)
                elif self.path in {"/api/start","/api/stop","/api/frame","/api/frames","/api/commit"}:
                    if self.path == "/api/frames":
                        frame_requests(body)
                    return self.send(200,manager.infer(owner,self.path[5:],body))
                elif self.path == "/api/translate-video":
                    return self.send(200, self.translate_video(owner, body))
                else:
                    return self.send(404, {"error":"Unknown endpoint"})
                return self.send(200,manager.status(owner))
            if self.path == "/api/start":
                self.server.backend.start()
                event = None
            elif not backend.session:
                return self.send(409, {"error": "Start a live session first"})
            elif self.path == "/api/frame":
                event = backend.frame(base64.b64decode(body["jpeg"], validate=True), int(body["slot"]))
            elif self.path == "/api/commit":
                event = backend.commit()
            elif self.path == "/api/stop":
                event = backend.stop()
            else:
                return self.send(404, {"error": "Unknown endpoint"})
            return self.send(200, {"event": event})
        except (KeyError, ValueError, TypeError, OSError, RuntimeError) as error:
            if manager:
                from live_guard import PreflightError, ProviderError
                if self.path.startswith("/api/gpu/") and not isinstance(error, (PreflightError, ProviderError)):
                    return self.send(400, {"error": "GPU request failed. Check the local app and Windows credentials"})
            return self.send(400, {"error": str(error)})

    def log_message(self, fmt, *args):
        pass  # Live frames, transcripts, and request bodies remain in memory.


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8765)
    replay = HERE.parent / "signgpt-results/20261127T062446887049Z/continuous-replay-events.jsonl"
    parser.add_argument("--replay", type=Path, default=replay if replay.is_file() else HERE / "demo-replay.jsonl")
    parser.add_argument("--live", type=Path, help="Path to bootstrapped Phase 2 work directory on a GPU Pod")
    parser.add_argument("--manage", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--open", action="store_true", help="Open SignToCap in the default browser")
    parser.add_argument("--setup-minutes", type=int, choices=range(1, 15), default=10,
                        metavar="1–14", help="GPU setup time from the actual startup (default 10)")
    args = parser.parse_args()
    if args.live:
        parser.error("Use the GPU Control Center in the Windows app for live testing")
    manager = None
    if os.name == "nt":
        from gpu_control import GPUControlCenter
        manager = GPUControlCenter(args.setup_minutes * 60)
    with PreviewServer(args.port, args.replay, manager=manager) as server:
        url = f"http://localhost:{server.server_port}"
        print(f"SignToCap at {url}", flush=True)
        if args.open:
            import webbrowser
            webbrowser.open(url)
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            if manager and manager.working():
                manager.stop("Desktop app closed")


if __name__ == "__main__":
    main()
