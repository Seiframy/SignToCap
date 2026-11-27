"""Offline integration tests for the universal GPU Control Center."""
import base64
from contextlib import ExitStack
from datetime import datetime, timezone
import hashlib
import io
import json
import os
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
import threading
import time
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import gpu_control as control
import gpu_setup as setup
import live_guard as guard
from desktop import PreviewServer


PODS = {
    "1jce4o05fmadxn": ("NVIDIA L4", .69, .70),
    "hs902er146ibel": ("NVIDIA GeForce RTX 4090", .89, .90),
    "xvhui1jmr4urgg": ("NVIDIA L40S", 1.09, 1.10),
}
L4, RTX, L40S = PODS
KEY = "fixture-private-key"
BLOB = b"\0\0\0\x0bssh-ed25519\0\0\0\x20" + bytes(range(32))
PUBLIC = base64.b64encode(BLOB).decode()
FINGERPRINT = "SHA256:" + base64.b64encode(hashlib.sha256(BLOB).digest()).decode().rstrip("=")


def host_output(pod, challenge=None):
    line = f"SignToCap session {challenge}\n" if challenge else ""
    return line + f"SignToCap Pod {pod}\nssh-ed25519 {PUBLIC} fixture\n256 {FINGERPRINT} fixture (ED25519)\n"


def clock_stamp():
    return datetime.now(timezone.utc).isoformat()


def fixture_pod(pod, model, rate, *, running=False):
    return {"id": pod, "gpu": {"id": model, "count": 1}, "dataCenterId": "EUR-IS-1",
            "mounts": {"network": [{"path": "/workspace", "volumeId": guard.VOLUME}]},
            "disk": 50,
            "startedAt": clock_stamp() if running else None, "status": "RUNNING" if running else "EXITED",
            "actions": ["stop"] if running else ["start"], "cost": rate, "ports": ["22/tcp"], "image": "fixture/cuda",
            "ssh": {"direct": {"username": "root", "host": "195.26.233.3", "port": 23022}}}


class FakeProvider:
    def __init__(self):
        self.rows = {pod: fixture_pod(pod, model, rate) for pod, (model, rate, _) in PODS.items()}
        self.actions = []
        self.capacity_pod = None
        self.delay_pod = None
        self.ambiguous_pod = None
        self.publish_delayed = threading.Event()

    def snapshot(self, _key):
        return [row.copy() for row in self.rows.values()]

    def __call__(self, key, action=None, *, pod, allow_start=False):
        assert key == KEY
        row = self.rows[pod]
        if not action:
            if pod == self.delay_pod and self.publish_delayed.is_set():
                self.delay_pod = None
                row.update(status="RUNNING", actions=["stop"], startedAt=clock_stamp())
            return row.copy()
        self.actions.append((action, pod))
        if action == "start":
            assert allow_start and row["status"] == "EXITED"
            if pod == self.capacity_pod:
                error = HTTPError("https://api.runpod.io", 400, "Bad Request", {},
                                  io.BytesIO(b'{"message":"No available GPUs for this Pod"}'))
                raise guard.request_error(error, key, "start")
            if pod == self.ambiguous_pod:
                raise guard.ProviderError(400, "response did not match a known error category", "start")
            if pod == self.delay_pod:
                raise TimeoutError()
            row.update(status="RUNNING", actions=["stop"], startedAt=clock_stamp())
        elif action == "stop":
            row.update(status="EXITED", actions=["start"])
        else:
            raise AssertionError(action)


class ThreadProcess(threading.Thread):
    """Run the real guard logic with inherited mocks for integrated HTTP tests."""
    def __init__(self, **kwargs):
        super().__init__(daemon=True, **kwargs)


def until(check, timeout=12):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if check():
            return
        time.sleep(.02)
    raise AssertionError("Timed out waiting for fixture state")


class CenterTests(unittest.TestCase):
    def setUp(self):
        stack = self.enterContext(ExitStack())
        self.temp = Path(stack.enter_context(TemporaryDirectory()))
        self.fixture = FakeProvider()
        stack.enter_context(patch.dict(os.environ, LOCALAPPDATA=str(self.temp)))
        self.save = stack.enter_context(patch.object(control.gpu_credentials, "save_key"))
        stack.enter_context(patch.object(control.gpu_credentials, "load_key", return_value=None))
        stack.enter_context(patch.object(control.gpu_credentials, "forget_key"))
        stack.enter_context(patch.object(control, "api", self.fixture))
        stack.enter_context(patch.object(guard, "api", self.fixture))
        stack.enter_context(patch.object(control, "list_pods", self.fixture.snapshot))
        stack.enter_context(patch.object(guard, "list_pods", self.fixture.snapshot))
        stack.enter_context(patch.object(guard, "old_watchdog_check"))
        stack.enter_context(patch("live_manager.mp.Process", ThreadProcess))
        self.exec_key = stack.enter_context(patch.object(setup, "provider", side_effect=self.provider_exec))
        self.ssh_calls = []
        self.fail_stage = None
        self.fail_worker = None
        self.ssh = stack.enter_context(patch.object(setup.subprocess, "run", side_effect=self.run_ssh))
        self.worker = stack.enter_context(patch.object(control.subprocess, "Popen", side_effect=self.spawn_worker))
        self.manager = control.GPUControlCenter()
        self.addCleanup(self.stop_fixture)

    def stop_fixture(self):
        if self.manager.process and self.manager.process.is_alive():
            self.manager.stop_request.set()
            self.fixture.publish_delayed.set()
            self.manager.process.join(6)

    def provider_exec(self, _key, path, _, body, **kwargs):
        self.assertTrue(path.endswith("/exec"))
        self.assertEqual(body["command"], ["sh", "-c", setup.HOST_KEY_COMMAND])
        return {"exitCode": 0, "timedOut": False, "truncated": False,
                "stdout": host_output(path.split("/")[1])}

    def run_ssh(self, command, *, cwd, **kwargs):
        self.ssh_calls.append(command)
        self.assertIn("StrictHostKeyChecking=yes", command)
        self.assertIn("GlobalKnownHostsFile=none", command)
        self.assertIn("UpdateHostKeys=no", command)
        self.assertNotIn(KEY, " ".join(command))
        pin_option = next(option for option in command if option.startswith("UserKnownHostsFile="))
        self.assertIn(PUBLIC, Path(pin_option.split('"')[1]).read_text())
        if self.fail_stage == "host":
            return SimpleNamespace(returncode=255, stdout="", stderr="Host key verification failed")
        if command[0] == "scp":
            if self.fail_stage == "transfer":
                return SimpleNamespace(returncode=1, stdout="", stderr="")
            if "worker.sha256" in command:
                for line in (cwd / "worker.sha256").read_text().splitlines():
                    digest, name = line.split()
                    self.assertEqual(hashlib.sha256((cwd / name).read_bytes()).hexdigest(), digest)
            else:
                self.assertIn("/pose-cache/hub/checkpoints/", command[-1])
            return SimpleNamespace(returncode=0, stdout="", stderr="")
        self.assertIn("unset RUNPOD_POD_ID; set -a; . /etc/rp_environment", command[-1])
        self.assertIn("mountpoint -q /workspace", command[-1])
        if "mkdir -m 700" in command[-1]:
            failure = {"pod": 42, "volume": 43}.get(self.fail_stage, 0)
            return SimpleNamespace(returncode=failure, stdout="pod-verified", stderr="")
        if "sha256sum" in command[-1]:
            return SimpleNamespace(returncode=int(self.fail_stage == "checksum"), stdout="worker-verified", stderr="")
        if self.fail_stage == "pose":
            status = "pose-cache-incomplete" if "probe-pose-cache" in command[-1] else ""
            return SimpleNamespace(returncode=0, stdout=status, stderr="")
        return SimpleNamespace(returncode=0, stdout="pose-cache-ready", stderr="")

    def spawn_worker(self, command, **kwargs):
        self.assertNotIn(KEY, " ".join(command))
        generation = self.fixture.rows[self.manager.profile.pod]["startedAt"]
        if self.fail_worker == "generation":
            generation = "incorrect"
        fixture = Mock()
        fixture.poll.return_value = None
        entries = ["pod-verified", '{"guard":true}', json.dumps({"ready": True, "guard": True,
                                                               "generation": generation})]
        entries.extend(['{"event":null}', '{"event":{"kind":"final","text":"fixture caption"}}'])
        fixture.stdout = io.StringIO("\n".join(entries) + "\n")
        return fixture

    def authenticate(self):
        self.manager.credentials({"key": KEY})
        self.manager.discover({})

    def body(self, pod, ceiling=None):
        candidate = self.manager.candidates[pod]
        compute = candidate["rate"]
        if ceiling is None:
            ceiling = PODS.get(pod, ("", 0, .90))[2]
        return {"confirmed": True, "pod": pod, "compute": compute, "storage": .008, "ceiling": ceiling,
                "total": guard.total_rate(compute, .008), "status": candidate["status"],
                "generation": candidate["generation"]}

    def wait_state(self, state):
        until(lambda: self.manager.state == state and (state != "stopped" or not self.manager.working()))

    def test_three_profiles_switch_and_new_pod_after_capacity_rejection(self):
        self.fixture.capacity_pod = L4
        new_pod = "newly-discovered_pod"
        self.fixture.rows[new_pod] = fixture_pod(new_pod, "RTX 4090", .88)
        self.authenticate()
        self.assertEqual(set(self.manager.candidates), {L4, RTX, L40S, new_pod})
        for pod, will_ready in ((L4, False), (RTX, True), (L40S, True), (new_pod, True)):
            with self.subTest(pod=pod):
                self.manager.begin("tab", self.body(pod))
                if will_ready:
                    self.wait_state("ready")
                    status = self.manager.status("tab")
                    self.assertTrue(status["ready"] and status["windows_guard"] and status["remote_guard"])
                    self.assertEqual(status["began"], guard.deployment_time(self.fixture.rows[pod]["startedAt"]))
                    self.assertEqual(status["setup_seconds"], 600)
                    with self.assertRaisesRegex(ValueError, "already in progress"):
                        self.manager.begin("tab", {"pod": L40S})
                    self.assertIsNone(self.manager.infer("tab", "start", {})["event"])
                    with self.assertRaisesRegex(ValueError, "Duplicate start"):
                        self.manager.infer("tab", "start", {})
                    self.assertEqual(self.manager.infer("tab", "commit", {})["event"]["text"], "fixture caption")
                    self.manager.stop()
                self.wait_state("stopped")
                self.assertEqual(self.fixture.rows[pod]["status"], "EXITED")
                self.assertIsNone(self.manager.pending())
                if not will_ready:
                    self.assertIn("capacity", self.manager.detail)
                    self.assertEqual(self.fixture.actions, [("start", L4)])
        self.assertEqual([pod for action, pod in self.fixture.actions if action == "start"], [L4, RTX, L40S, new_pod])

    def test_a100_requires_manual_start_and_explicit_ceiling(self):
        pod = "a100-fixture"
        self.fixture.rows[pod] = fixture_pod(pod, guard.A100_SXM_80GB, 2.20)
        self.fixture.rows["a100-pcie"] = fixture_pod("a100-pcie", "NVIDIA A100 80GB PCIe", 2.20)
        self.authenticate()
        self.assertNotIn("a100-pcie", self.manager.candidates)
        self.assertIsNone(self.manager.candidates[pod]["cap"])
        with self.assertRaisesRegex(ValueError, "manually"):
            self.manager.begin("tab", self.body(pod, 2.25))
        generation = datetime.fromtimestamp(time.time() - 30, timezone.utc).isoformat()
        self.fixture.rows[pod].update(status="RUNNING", actions=["stop"], startedAt=generation)
        self.manager.discover({})
        for overrides, reason in (({"confirmed": False}, "Confirm"), ({"ceiling": None}, "ceiling"),
                                  ({"ceiling": 2.20}, "exceeds")):
            with self.assertRaisesRegex(ValueError, reason):
                self.manager.begin("tab", self.body(pod, 2.25) | overrides)
        self.assertEqual(self.fixture.actions, [])
        self.manager.begin("tab", self.body(pod, 2.25))
        self.wait_state("ready")
        self.assertEqual(self.manager.profile.rate_cap, 2.25)
        self.assertLessEqual(self.manager.status("tab")["remaining"], 870)
        self.manager.stop()
        self.wait_state("stopped")
        self.assertEqual(self.fixture.actions, [("stop", pod)])

    def test_attach_keeps_original_deadline_and_web_terminal_rejects_wrong_key(self):
        generation = datetime.fromtimestamp(time.time() - 30, timezone.utc).isoformat()
        self.fixture.rows[RTX].update(status="RUNNING", actions=["stop"], startedAt=generation)
        self.exec_key.side_effect = guard.ProviderError(404, "not found", "exec")
        self.authenticate()
        self.manager.begin("tab", self.body(RTX))
        self.wait_state("host_key")
        status = self.manager.status("tab")
        self.assertGreater(status["setup_remaining"], 560)
        self.assertLessEqual(status["setup_remaining"], 570)
        self.assertIsNone(self.manager.status("other")["pin_command"])
        challenge = self.manager.pin_request[1]
        with self.assertRaisesRegex(ValueError, "selected Pod"):
            self.manager.submission("tab", {"trusted": True, "output": host_output(L4, challenge)})
        with self.assertRaisesRegex(ValueError, "fingerprint"):
            self.manager.submission("tab", {"trusted": True,
                                           "output": host_output(RTX, challenge).replace(FINGERPRINT, "SHA256:wrong")})
        self.manager.submission("tab", {"trusted": True, "output": host_output(RTX, challenge)})
        self.wait_state("ready")
        heartbeat = self.manager.heartbeat.value
        self.manager.status("other")
        self.assertEqual(self.manager.heartbeat.value, heartbeat)
        self.manager.stop()
        self.wait_state("stopped")
        self.assertEqual(self.fixture.actions, [("stop", RTX)])

    def test_stop_reconciles_delayed_start_and_does_not_dispatch_again(self):
        self.fixture.delay_pod = RTX
        self.authenticate()
        approval = self.body(RTX)
        self.manager.begin("tab", approval)
        self.wait_state("watching_delayed_start")
        self.assertGreater(self.manager.status("tab")["start_remaining"], 0)
        self.manager.stop()
        self.assertTrue(self.manager.status("tab")["cancel_requested"])
        self.assertIn("Start reply missing", self.manager.detail)
        with self.assertRaisesRegex(ValueError, "already in progress"):
            self.manager.begin("tab", approval)
        self.fixture.publish_delayed.set()
        self.wait_state("stopped")
        self.assertEqual(self.fixture.actions, [("start", RTX), ("stop", RTX)])
        self.worker.assert_not_called()

    def test_unclassified_400_retains_reason_and_blocks_switch_after_timeout(self):
        self.fixture.ambiguous_pod = L4
        self.authenticate()
        with patch.object(guard, "START_WAIT", 2):
            self.manager.begin("tab", self.body(L4))
            self.wait_state("watching_delayed_start")
            self.manager.stop()
            self.wait_state("shutdown_unverified")
        self.assertIn("HTTP 400", self.manager.detail)
        self.assertFalse(self.manager.status("tab")["available"])
        self.assertFalse(guard.reconcile_pending(KEY))
        self.assertEqual(self.fixture.actions, [("start", L4)])

    def test_capacity_and_price_checks_prevent_billing_wrong_target(self):
        self.authenticate()
        with self.assertRaisesRegex(ValueError, "exceeds"):
            self.manager.begin("tab", self.body(L40S, .90))
        with self.assertRaisesRegex(ValueError, "exceeds"):
            self.manager.begin("tab", self.body(RTX) | {"storage": .02, "total": .91})
        price = self.body(RTX)
        self.fixture.rows[RTX]["cost"] = .895
        self.manager.begin("tab", price)
        self.wait_state("error")
        self.assertIn("quote changed", self.manager.detail)
        self.manager.process.join(5)
        self.manager.discover({})
        self.fixture.rows[L4].update(status="STARTING", actions=["stop"])
        self.manager.begin("tab", self.body(L40S))
        self.wait_state("error")
        self.manager.process.join(5)
        self.assertIn("Another active Pod", self.manager.detail)
        self.assertEqual(self.fixture.actions, [])

    def test_discovery_explains_empty_lists_missing_quotes_and_storage(self):
        self.authenticate()
        row = self.manager.candidates[L4]
        self.assertEqual((row["status"], row["rate"], row["disk"], row["storage"], row["issue"]),
                         ("EXITED", .69, 50, .008, ""))
        with self.assertRaisesRegex(ValueError, "below the published container disk rate"):
            self.manager.begin("tab", self.body(L4) | {"storage": .007, "total": .697})
        self.fixture.rows[L4]["cost"] = 0
        self.fixture.rows[RTX].pop("disk")
        self.manager.discover({})
        self.assertIn("Start offer", self.manager.candidates[L4]["issue"])
        self.assertIn("container disk size", self.manager.candidates[RTX]["issue"])
        self.fixture.rows = {L40S: self.fixture.rows[L40S] | {"dataCenterId": "US-KS-2"}}
        self.manager.discover({})
        self.assertEqual(self.manager.status("tab")["candidates"], [])
        self.assertIn("data center", self.manager.status("tab")["discovery_detail"])
        self.fixture.rows.clear()
        self.manager.discover({})
        self.assertIn("key’s workspace", self.manager.status("tab")["discovery_detail"])

    def test_disk_change_prevents_paid_start(self):
        self.authenticate()
        quote = self.body(L4)
        self.fixture.rows[L4]["disk"] = 60
        self.manager.begin("tab", quote)
        self.wait_state("error")
        self.assertIn("container storage changed", self.manager.detail)
        self.assertEqual(self.fixture.actions, [])

    def test_ssh_and_staging_fail_closed(self):
        for failure in ("host", "pod", "volume", "transfer", "checksum"):
            with self.subTest(failure=failure):
                self.fail_stage = failure
                self.authenticate()
                self.manager.begin("tab", self.body(L4))
                self.wait_state("stopped")
                self.assertEqual(self.fixture.actions[-2:], [("start", L4), ("stop", L4)])
                self.worker.assert_not_called()
                self.assertIn({"host":"host key", "pod":"POD_ID", "volume":"volume",
                               "transfer":"staging", "checksum":"checksum"}[failure], self.manager.detail)

    def test_wrong_worker_generation_stops_before_live(self):
        self.fail_worker = "generation"
        self.authenticate()
        self.manager.begin("tab", self.body(L40S))
        self.wait_state("stopped")
        self.assertIn("generation", self.manager.detail)
        self.assertFalse(self.manager.status("tab")["ready"])
        self.assertEqual(self.fixture.actions, [("start", L40S), ("stop", L40S)])

    def test_incomplete_pose_cache_is_restored_from_verified_local_files(self):
        self.fail_stage = "pose"
        cache = self.temp / "Temp/signgpt-phase2-stage/ort-cache/rtmlib/hub/checkpoints"
        cache.mkdir(parents=True)
        (cache / "fixture.onnx").write_bytes(b"fixture weights")
        weights = {"fixture.onnx": hashlib.sha256(b"fixture weights").hexdigest()}
        self.authenticate()
        with patch.object(setup, "POSE_WEIGHTS", weights):
            self.manager.begin("tab", self.body(RTX))
            self.wait_state("ready")
        self.assertEqual(sum(call[0] == "scp" for call in self.ssh_calls), 2)
        self.assertIn("export SIGNTOCAP_POSE_CACHE=/workspace/signgpt-live-", self.worker.call_args.args[0][-1])
        self.manager.stop()
        self.wait_state("stopped")

    def test_stop_without_confirmation_blocks_gpu_switch_until_reconciled(self):
        self.authenticate()
        self.manager.begin("tab", self.body(L40S))
        self.wait_state("ready")
        with patch.object(guard, "verified_stop", return_value=False):
            self.manager.stop()
            self.wait_state("shutdown_unverified")
            self.manager.process.join(5)
        self.assertEqual(self.manager.pending()[:2], (L40S, self.manager.generation))
        with self.assertRaisesRegex(ValueError, "unverified shutdown"):
            self.manager.begin("tab", self.body(RTX))
        self.manager.stop()
        self.wait_state("stopped")
        self.assertEqual(self.fixture.rows[L40S]["status"], "EXITED")

    def test_credentials_and_managed_loopback(self):
        self.manager.credentials({"key": KEY})
        self.save.assert_not_called()
        self.manager.credentials({"key": KEY, "remember": True})
        self.save.assert_called_once_with(KEY)
        path = Path(__file__).resolve().parents[1] / "signgpt-results/20261127T062446887049Z/continuous-replay-events.jsonl"
        with PreviewServer(0, path, manager=self.manager) as server:
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            try:
                base = f"http://127.0.0.1:{server.server_port}"
                with urlopen(base + "/api/status") as response:
                    token = json.load(response)["token"]
                with urlopen(base + "/api/replay") as response:
                    self.assertTrue(json.load(response))
                data = json.dumps({}).encode()
                headers = {"Origin": base, "X-Session": "tab", "X-Control": token, "Content-Type": "application/json"}
                with self.assertRaises(HTTPError) as denied:
                    urlopen(Request(base + "/api/gpu/discover", data=data))
                with denied.exception:
                    self.assertEqual(denied.exception.code, 403)
                with urlopen(Request(base + "/api/gpu/discover", data=data, headers=headers)) as response:
                    status = response.read().decode()
                    self.assertNotIn(KEY, status)
                    self.assertIn(L40S, status)
                data = json.dumps(self.body(L40S)).encode()
                with self.assertRaises(HTTPError) as stopped:
                    urlopen(Request(base + "/api/gpu/start", data=data, headers=headers))
                self.assertIn(b"Start the Pod in the RunPod console", stopped.exception.read())
                stopped.exception.close()
                self.assertEqual(self.fixture.actions, [])
                self.fixture.rows[L40S].update(status="RUNNING", startedAt=clock_stamp(), actions=["stop"])
                self.manager.discover({})
                data = json.dumps(self.body(L40S)).encode()
                with urlopen(Request(base + "/api/gpu/start", data=data, headers=headers)) as response:
                    self.assertNotIn(KEY, response.read().decode())
                self.wait_state("ready")
                with urlopen(base + "/api/status") as response:
                    self.assertFalse(json.load(response)["live"])
                with urlopen(Request(base + "/api/status", headers=headers)) as response:
                    self.assertTrue(json.load(response)["live"])
                with urlopen(Request(base + "/api/gpu/stop", data=b"{}", headers=headers)) as response:
                    self.assertEqual(response.status, 200)
                self.wait_state("stopped")
                for file in self.temp.rglob("*"):
                    if file.is_file():
                        self.assertNotIn(KEY.encode(), file.read_bytes())
            finally:
                server.shutdown()
                thread.join(5)


class GuardIntegrationTests(unittest.TestCase):
    def test_runpod_v2_shapes_and_transition_are_bound_to_exact_id(self):
        def response(body):
            return io.BytesIO(json.dumps(body).encode())
        with patch.object(guard, "urlopen", side_effect=[
            response({"pods": [{"id": L4}], "pagination": {"hasNextPage": True, "nextCursor": "a/b+?"}}),
            response({"pods": [{"id": RTX}, {"id": L40S}], "pagination": {"hasNextPage": False}}),
            response({"id": RTX}),
            response({"id": RTX}),
            response({"id": L4}),
        ]) as request:
            self.assertEqual([row["id"] for row in guard.list_pods(KEY)], [L4, RTX, L40S])
            self.assertEqual(guard.api(KEY, pod=RTX)["id"], RTX)
            guard.api(KEY, "start", pod=RTX, allow_start=True)
            with self.assertRaisesRegex(ValueError, "identity"):
                guard.api(KEY, pod=RTX)
        calls = [call.args[0] for call in request.call_args_list]
        self.assertTrue(calls[1].full_url.endswith("?cursor=a%2Fb%2B%3F"))
        self.assertTrue(calls[3].full_url.endswith(f"/pods/{RTX}/action"))
        self.assertEqual(json.loads(calls[3].data), {"action": "start"})
        self.assertIn(KEY, calls[3].get_header("Authorization"))
        self.assertNotIn(KEY, calls[3].full_url)

    def test_setup_budget_and_no_stop_on_replacement_generation(self):
        self.assertEqual(guard.setup_deadline(100, 999), 985)
        self.assertFalse(guard.deadline_reason(699, 100, 0, 0, 600))
        self.assertIn("Setup", guard.deadline_reason(700, 100, 0, 0, 600))
        self.assertIn("heartbeat", guard.deadline_reason(140, 100, 120, 0, 600))
        with patch.object(guard, "api", return_value={"startedAt": "other", "status": "RUNNING"}) as status:
            with self.assertRaisesRegex(RuntimeError, "session changed"):
                guard.verified_stop(KEY, clock_stamp(), pod=L40S)
            status.assert_called_once_with(KEY, pod=L40S)

    def test_pending_unknown_blocks_new_start_until_manual_reconciliation(self):
        with TemporaryDirectory() as tmp, patch.dict(os.environ, LOCALAPPDATA=tmp):
            directory = guard.state_directory()
            directory.mkdir()
            with patch.object(guard.time, "time", return_value=100):
                guard.mark_pending(directory, RTX, None)
                self.assertFalse(guard.reconcile_pending(KEY, manual=True))
            with patch.object(guard, "api", return_value={"status": "EXITED"}):
                self.assertFalse(guard.reconcile_pending(KEY))
                self.assertTrue(guard.reconcile_pending(KEY, manual=True))
            self.assertIsNone(guard.pending_session(directory))

    def test_persisted_rejection_recovers_only_after_exact_stopped_read(self):
        with TemporaryDirectory() as tmp, patch.dict(os.environ, LOCALAPPDATA=tmp):
            directory = guard.state_directory()
            directory.mkdir()
            generation = clock_stamp()
            guard.mark_pending(directory, RTX, None,
                               previous_generation=generation, rejected=True, start_detail="fixture rejection")
            with patch.object(guard, "api", side_effect=TimeoutError):
                with self.assertRaises(TimeoutError):
                    guard.reconcile_pending(KEY)
            with patch.object(guard, "api", return_value={"status": "EXITED", "startedAt": generation}) as read:
                self.assertTrue(guard.reconcile_pending(KEY))
                read.assert_called_once_with(KEY, pod=RTX)
            self.assertIsNone(guard.pending_session(directory))

    def test_windows_watchdog_survives_abrupt_app_exit(self):
        with TemporaryDirectory() as tmp:
            directory = Path(tmp)
            app = subprocess.Popen([sys.executable, "-B", __file__, "--crash-parent", tmp],
                                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            try:
                until(lambda: (directory / "start-sent").is_file())
                app.kill()
                app.wait(5)
                log = directory / "SignToCap/live-shutdown.log"
                until(lambda: log.is_file() and "verified stopped" in log.read_text()
                      and not guard.pending_path(log.parent).exists())
                self.assertIn("Desktop app closed", log.read_text())
                self.assertTrue((directory / "stop-sent").is_file())
                self.assertFalse(guard.pending_path(log.parent).exists())
            finally:
                if app.poll() is None:
                    app.kill()
                    app.wait(5)


def crash_guard(directory, stop, heartbeat, device_test, updates, reason):
    fake = FakeProvider()

    def api(key, action=None, **kwargs):
        result = fake(key, action, **kwargs)
        if action:
            (Path(directory) / (action + "-sent")).touch()
        return result

    profile = guard.PodProfile(L40S, "NVIDIA L40S", 1.1, aliases=("NVIDIA L40S",))
    approval = guard.Approval(profile, "EXITED", None, 1.09, .007, 600)
    with patch.object(guard, "api", api), patch.object(guard, "list_pods", fake.snapshot), \
            patch.object(guard, "old_watchdog_check"):
        guard.guard(KEY, approval.total, stop, heartbeat, device_test, updates, None, reason, profile, approval)


if __name__ == "__main__":
    if sys.argv[1:2] == ["--crash-parent"]:
        import multiprocessing as mp
        os.environ["LOCALAPPDATA"] = sys.argv[2]
        mp.set_executable(str(Path(sys.executable).with_name("pythonw.exe")))
        primitives = (mp.Event(), mp.Value("d", 0), mp.Value("d", 0), mp.Queue(), mp.Array("c", 160))
        child = mp.Process(target=crash_guard, args=(sys.argv[2], *primitives))
        child.start()
        child.join(20)
    else:
        unittest.main()
