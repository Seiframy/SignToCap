"""Local upload, sampling and pose-selection regression tests."""
import base64
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from threading import Thread
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import cv2
import numpy as np

from desktop import HERE, PreviewServer, UniSignBackend
from pose_data import PoseExtractor, extract_video, load_track


class Phase4Tests(unittest.TestCase):
    def test_real_video_decoder_sampling_preserves_original_frame_times(self):
        with TemporaryDirectory() as folder:
            video, output = (Path(folder) / name for name in ("sample.avi", "track.npz"))
            writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"MJPG"), 24, (32, 24))
            self.assertTrue(writer.isOpened())
            try:
                for frame in range(24):
                    writer.write(np.full((24, 32, 3), frame, dtype=np.uint8))
            finally:
                writer.release()
            for fps in (6, 12, 24):
                with self.subTest(fps=fps):
                    points = np.zeros((1, 133, 2), dtype=np.float32)
                    scores = np.ones((1, 133), dtype=np.float32)
                    extractor = SimpleNamespace(extract=Mock(return_value=(points, scores)))
                    stats = extract_video(SimpleNamespace(video=video, output=output, sample_fps=fps,
                                          start_ms=0, end_ms=None, verify_cadence=True, whole_sentence=True),
                                          extractor=extractor)
                    self.assertEqual((stats["source_frames"], stats["frames"], stats["decoded_frames"]), (24, fps, 24))
                    self.assertEqual(extractor.extract.call_count, fps)
                    np.testing.assert_allclose(stats["frame_ms"], np.arange(fps) * 1000 / fps, atol=.001)
                    self.assertEqual(load_track(output)[2], fps)

    def test_invalid_detection_cannot_replace_valid_signer_or_tracking_center(self):
        tracking = PoseExtractor.__new__(PoseExtractor)
        tracking.filter_invalid = True
        tracking.previous_center = None
        joints = np.tile([40., 20.], (2, 133, 1))
        scores = np.full((2, 133), .7)
        joints[0, 5] = np.nan
        scores[0] = 1
        tracking.detector = Mock(return_value=(joints, scores))
        observed, confident = tracking.extract(np.zeros((40, 160, 3)))
        np.testing.assert_allclose(observed[0, 0], [.25, .50])
        np.testing.assert_allclose(tracking.previous_center, [.25, .50])
        self.assertAlmostEqual(float(confident[0, 91]), .7)
        tracking.detector.return_value = (joints[:1], scores[:1])
        _, absent = tracking.extract(np.zeros((40, 160, 3)))
        self.assertEqual(int(np.count_nonzero(absent)), 0)
        np.testing.assert_allclose(tracking.previous_center, [.25, .50])

    def test_sentence_rates_reject_duplicates_and_missing_slots(self):
        backend = UniSignBackend.__new__(UniSignBackend)
        backend.model = Mock()
        points, scores = np.zeros((1, 133, 2)), np.ones((1, 133))
        backend.pose = SimpleNamespace(extract=lambda _: (points, scores))
        jpg = cv2.imencode(".jpg", np.zeros((16, 16, 3), dtype=np.uint8))[1].tobytes()
        for fps in (6, 12, 24):
            backend.start("sentence", fps)
            backend.frame(jpg, 0, 0)
            backend.frame(jpg, 1, 1000 / fps + .1)
            with self.assertRaises(ValueError):
                backend.frame(jpg, 1, 2000 / fps + .1)
            event = backend.frame(jpg, 3, 3000 / fps + .1)
            self.assertEqual(event["reason"], "capture_gap")
            self.assertEqual(backend.commit()["reason"], "capture_gap")
        with self.assertRaises(ValueError):
            backend.start("sentence", 30)
        backend.model.translate.assert_not_called()

    def test_upload_keeps_raw_video_local_and_requires_guard_owner(self):
        manager = SimpleNamespace(owner="viewer", generation="exact", live=False,
                                  status=lambda _: {"ready": True},
                                  infer=Mock(return_value={"event": {"kind": "final", "text": "Hello."}}))
        clip = b"fixture bytes"

        def video_tool(command, *args, **kwargs):
            if command == "probe":
                self.assertEqual(args[0].read_bytes(), clip)
                return {"source_fps": 24, "source_frames": 24, "duration_ms": 1000}
            args[2].write_bytes(b"only poses")
            return {"frames": 24}

        with PreviewServer(0, HERE / "demo-replay.jsonl", manager=manager) as server:
            thread = Thread(target=server.serve_forever, daemon=True)
            thread.start()
            base = f"http://localhost:{server.server_port}"

            def request(mode, owner="viewer", token=server.token):
                return urlopen(Request(base + "/api/upload-" + mode, data=clip,
                               headers={"Origin": base, "X-Control": token, "X-Session":owner,
                                        "X-Video-Extension": ".mp4", "X-Sample-Fps": "24"}), timeout=5)
            try:
                with patch("desktop.video_tool", side_effect=video_tool):
                    with request("preview") as response:
                        self.assertEqual(json.load(response)["video"]["source_fps"], 24)
                    with self.assertRaises(HTTPError) as wrong:
                        request("preview", token="wrong")
                    self.assertEqual(wrong.exception.code, 403)
                    with self.assertRaises(HTTPError):
                        request("translate", owner="different")
                    manager.infer.assert_not_called()
                    with request("translate") as response:
                        output = [json.loads(line) for line in response]
                    self.assertEqual([row["stage"] for row in output], ["extracting", "translating", "complete"])
                    manager.infer.assert_called_once_with("viewer", "prerecorded",
                        {"poses": base64.b64encode(b"only poses").decode(), "generation": "exact"})
            finally:
                server.shutdown()
