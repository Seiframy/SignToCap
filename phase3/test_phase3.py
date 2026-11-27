"""CPU-only behavior tests, including matched traces with a stub decoder."""

import io
import json
from pathlib import Path
import shutil
import sys
from tempfile import TemporaryDirectory
from threading import Thread
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import numpy as np

from followup import Settings
import benchmark
import followup
from captions import AdaptiveStreamingSession, repeated_phrase
from pose_data import load_track, save_track
from desktop import UniSignBackend, frame_requests
from pose_data import PoseExtractor


class ControllerTests(unittest.TestCase):
    def setUp(self):
        self.points = np.zeros((5, 1, 133, 2), dtype=np.float32)
        self.scores = np.ones((5, 1, 133), dtype=np.float32)
        self.calls = []

    def translate(self, points, scores, *, beams, tokens):
        self.calls.append(tokens)
        return "The school is open.", [], {"work_ms": 20, "prep_ms": 1, "generation_ms": 18, "pose_encode_ms": 1}

    def test_stable_preview_and_explicit_commit(self):
        options = Settings(controller="adaptive", policy="explicit")
        session = AdaptiveStreamingSession(options, SimpleNamespace(translate=self.translate), 10)
        with patch("captions.activity_features", side_effect=lambda p, c, f: (np.ones(len(p)), np.ones(len(p)))):
            self.assertEqual(session.push(self.points, self.scores)["kind"], "telemetry")
            self.assertIsNone(session.push(self.points, self.scores))
            self.assertEqual(session.push(self.points, self.scores)["kind"], "partial")
            self.assertEqual(session.commit()["kind"], "final")
            self.assertIsNone(session.finish())
            self.assertEqual(session.commit()["kind"], "uncertain")
        self.assertEqual(self.calls, [24, 24, 48])
        self.assertTrue(repeated_phrase("very versatile, very versatile, very versatile."))

    def test_repeated_final_is_rejected(self):
        options = Settings(controller="adaptive", policy="explicit")
        bad = SimpleNamespace(translate=lambda *a, **k: ("yes yes yes", [], {"work_ms": 1}))
        session = AdaptiveStreamingSession(options, bad, 10)
        with patch("captions.activity_features", return_value=(np.ones(5), np.ones(5))):
            self.assertEqual(session.push(self.points, self.scores)["reason"], "repetitive_preview")
        self.assertEqual(session.commit()["kind"], "uncertain")

    def test_live_gap_does_not_add_synthetic_poses(self):
        import cv2

        backend = UniSignBackend.__new__(UniSignBackend)
        backend.model = SimpleNamespace(translate=self.translate)
        backend.pose = SimpleNamespace(extract=lambda _: (self.points[0], self.scores[0]))
        backend.start()
        jpeg = cv2.imencode(".jpg", np.zeros((8, 8, 3), dtype=np.uint8))[1].tobytes()
        backend.frame(jpeg, 0)
        event = backend.frame(jpeg, 2)
        self.assertEqual(event["reason"], "capture_gap")
        self.assertEqual(backend.session.seen, 3)
        self.assertEqual(len(backend.session.keypoints), 1)
        self.assertEqual(backend.commit()["reason"], "no_complete_window")
        self.assertEqual(self.calls, [])

    def test_prerecorded_uses_complete_window_and_rejects_missing_pose(self):
        backend = UniSignBackend.__new__(UniSignBackend)
        translate = Mock(return_value=("The school is open.", [], {"work_ms": 20}))
        backend.model = SimpleNamespace(translate=translate)
        with TemporaryDirectory() as tmp:
            track = Path(tmp) / "clip.npz"
            points, scores = np.tile(self.points, (4, 1, 1, 1)), np.tile(self.scores, (4, 1, 1))
            save_track(track, points, scores, 24)
            result = backend.translate_recorded(track)
            self.assertEqual((result["kind"], result["frames"], result["source_fps"]), ("final", 20, 24))
            np.testing.assert_array_equal(translate.call_args.args[0], points)
            save_track(track, points, scores * 0, 24)
            self.assertEqual(backend.translate_recorded(track)["reason"], "insufficient_pose")
            translate.assert_called_once()

    def test_sentence_batch_preserves_times_and_rejects_missing_frames(self):
        import base64
        import cv2

        backend = UniSignBackend.__new__(UniSignBackend)
        translate = Mock(return_value=("The school is open.", [], {"work_ms": 20}))
        backend.model = SimpleNamespace(translate=translate)
        backend.pose = SimpleNamespace(extract=lambda _: (self.points[0], self.scores[0]))
        jpeg = base64.b64encode(cv2.imencode(".jpg", np.zeros((8, 8, 3), dtype=np.uint8))[1]).decode()
        timings = [{"slot":slot,"at_ms":ms,"jpeg":jpeg} for slot,ms in enumerate((3,167,335))]
        with self.assertRaisesRegex(ValueError, "ordered"):
            frame_requests({"frames":timings[::-1]})
        backend.start("sentence")
        result = backend.frames(frame_requests({"frames":timings}))
        self.assertEqual([row["at_ms"] for row in result], [3,167,335])
        self.assertTrue(all(row["event"] is None for row in result))
        self.assertEqual(backend.commit()["text"], "The school is open.")
        self.assertEqual(len(translate.call_args.args[0]), 3)
        backend.start("sentence")
        backend.frames(frame_requests({"frames":[timings[0],timings[2]]}))
        self.assertEqual(backend.commit()["reason"], "capture_gap")
        backend.start("sentence")
        backend.pose.extract = lambda _: (self.points[0], self.scores[0] * 0)
        backend.frames(frame_requests({"frames":timings}))
        self.assertEqual(backend.sentence_points, [])
        self.assertEqual(backend.commit()["reason"], "insufficient_pose")
        translate.assert_called_once()

    def test_cuda_requires_real_operators_on_both_pose_models(self):
        def session(trace):
            return SimpleNamespace(get_providers=lambda: ["CUDAExecutionProvider"],
                                   end_profiling=lambda: str(trace))

        with TemporaryDirectory() as tmp:
            det, pose = (Path(tmp) / (name + ".json") for name in ("detector", "pose"))
            for path in (det, pose):
                path.write_text('[{"cat":"Node","args":{"provider":"CUDAExecutionProvider"}}]', encoding="utf-8")
            extractor = PoseExtractor.__new__(PoseExtractor)
            extractor.detector = SimpleNamespace(
                det_model=SimpleNamespace(session=session(det)),
                pose_model=SimpleNamespace(session=session(pose)))
            self.assertEqual(extractor.cuda_operators()["detector"]["CUDAExecutionProvider"], 1)
            pose.write_text('[{"cat":"Node","args":{"provider":"CPUExecutionProvider"}}]', encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "pose executed no CUDA operators"):
                extractor.cuda_operators()

    def test_controller_comparison_on_recorded_pose_tracks(self):
        staged = Path(sys.argv[1]) if len(sys.argv) > 1 else None
        if staged is None or not staged.is_file():
            self.skipTest("Pass the two-track Phase 2 continuous manifest")
        totals = {}
        for controller in ("baseline", "adaptive"):
            totals[controller] = {"calls": 0, "partials": 0, "finals": 0}
            for row in benchmark.read_rows(staged):
                now = [0.]
                def wait(seconds):
                    now[0] += seconds
                stream = io.StringIO()
                with patch.object(benchmark, "perf_counter", lambda: now[0]), patch.object(benchmark, "sleep", wait):
                    benchmark.replay(Settings(controller=controller), stream,
                                     SimpleNamespace(translate=self.translate), row, load_track(staged.parent / row["pose"]), None)
                rows = [json.loads(line) for line in stream.getvalue().splitlines()]
                totals[controller]["calls"] += rows[-1]["model_calls"]
                totals[controller]["partials"] += sum(r["kind"] == "partial" for r in rows)
                totals[controller]["finals"] += sum(r["kind"] == "final" for r in rows)
        self.assertLess(totals["adaptive"]["calls"], totals["baseline"]["calls"])
        self.assertLess(totals["adaptive"]["partials"], totals["baseline"]["partials"])
        print("Stub decoder, actual pose tracks:", json.dumps(totals))

    def test_loopback_replay_and_live_guard(self):
        from desktop import PreviewServer
        path = Path(__file__).resolve().with_name("demo-replay.jsonl")
        server = PreviewServer(0, path)
        worker = Thread(target=server.serve_forever, daemon=True)
        worker.start()
        try:
            base = f"http://127.0.0.1:{server.server_port}"
            with urlopen(base + "/api/replay", timeout=3) as response:
                tracks = json.load(response)
            self.assertEqual(set(tracks), {"Illustrative caption replay"})
            self.assertEqual([e["kind"] for e in next(iter(tracks.values()))],
                             ["partial", "final", "partial", "final"])
            with self.assertRaises(HTTPError) as caught:
                urlopen(Request(base + "/api/start", data=b"{}"), timeout=3)
            self.assertEqual(caught.exception.code, 409)
        finally:
            server.shutdown()
            server.server_close()
            worker.join()


    def test_followup_plans_on_staged_inputs(self):
        staged = Path(sys.argv[1]) if len(sys.argv) > 1 else None
        audit = Path(__file__).resolve().parent / "audit-data"
        if staged is None or not staged.is_file() or not (audit / "pose-audit.json").is_file():
            self.skipTest("Requires the real audit pairs and historical continuous manifest")
        now = [0.]
        def wait(seconds):
            now[0] += seconds
        fake = SimpleNamespace(np=np, translate=self.translate, memory=lambda: {},
                               torch=SimpleNamespace(cuda=Mock()))
        with TemporaryDirectory() as tmp:
            work = Path(tmp)
            output = work / "out"
            output.mkdir()
            shutil.copytree(audit, work / "prepared/phase3")
            shutil.copytree(staged.parent, work / "prepared/continuous")
            with patch.object(followup, "Translator", return_value=fake) as load, \
                 patch.object(followup, "summarize") as summarize, \
                 patch.object(benchmark, "perf_counter", lambda: now[0]), patch.object(benchmark, "sleep", wait):
                followup.main(work, output)
            load.assert_called_once()
            self.assertEqual(summarize.call_count, 7)
            full = benchmark.read_rows(output / "heldout-fresh-events.jsonl")
            sampled = benchmark.read_rows(output / "heldout-fresh-3fps-events.jsonl")
            self.assertEqual(sampled[0]["sampling_fps"], 3)
            self.assertEqual([event["id"] for event in full[1:-1]], [event["id"] for event in sampled[1:-1]])


if __name__ == "__main__":
    unittest.main(argv=sys.argv[:1])
