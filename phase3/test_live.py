"""No-network rehearsal of the production guard and session broker."""
import os
import io
import hashlib
import json
import multiprocessing as mp
from datetime import datetime, timezone
from pathlib import Path
from queue import Queue
from types import SimpleNamespace
from tempfile import TemporaryDirectory
import threading
import unittest
from unittest.mock import Mock, patch

import live_guard as g
from live_manager import LiveManager
import live_worker

STOP = dict(id=g.POD,gpu={"id":"NVIDIA L4","count":1},dataCenterId="EUR-IS-1",
            mounts={"network":[{"path":"/workspace","volumeId":g.VOLUME}]},status="EXITED",actions=["start"],startedAt="old",cost=0)
RUN = dict(STOP,status="RUNNING",actions=["stop"],cost=0.70,startedAt=datetime.now(timezone.utc).isoformat(),
           ssh={"direct":{"host":"127.0.0.1","port":2222,"username":"root"}})
RTX_POD = "fixture4090xx"
RTX_RUN = dict(RUN, id=RTX_POD, gpu={"id":"NVIDIA GeForce RTX 4090","count":1}, cost=.81)


def child_fixture(ready,stop,queue,profile,running):
    state=[False]
    def api(key,action=None,*,pod=None):
        if pod != profile.pod:
            raise ValueError("Wrong fixture Pod")
        if action=="stop":
            state[0] = True
        if not ready.is_set():
            return STOP
        return dict(running,status="EXITED") if state[0] else running
    with patch.object(g,"old_watchdog_check"),patch.object(g,"api",api):
        g.guard("fixture",profile.rate_cap,stop,SimpleNamespace(value=0),SimpleNamespace(value=0),queue,
                profile.started_at,SimpleNamespace(value=b"Emergency stop requested"),profile)


class GuardTests(unittest.TestCase):
    def test_independent_windows_process(self):
        self.check_independent_process(g.L4_PROFILE, RUN)

    def test_independent_windows_process_arms_running_4090(self):
        self.check_independent_process(g.PodProfile.rtx4090(RTX_POD,.82,RTX_RUN["startedAt"]), RTX_RUN)

    def check_independent_process(self, profile, pod):
        import sys
        mp.set_executable(str(Path(sys.executable).with_name("pythonw.exe")))
        ready,stop,queue=mp.Event(),mp.Event(),mp.Queue()
        with TemporaryDirectory() as tmp,patch.dict(os.environ,LOCALAPPDATA=tmp):
            if profile.started_at:
                ready.set()
            proc=mp.Process(target=child_fixture,args=(ready,stop,queue,profile,pod))
            proc.start()
            try:
                if not profile.started_at:
                    self.assertEqual(queue.get(timeout=10)[0],"waiting")
                    ready.set()
                state, fields = queue.get(timeout=10)
                while state == "pod_status":
                    state, fields = queue.get(timeout=10)
                self.assertEqual(state,"setup",fields)
                self.assertEqual(fields["generation"],pod["startedAt"])
                self.assertEqual(fields["began"],g.deployment_time(pod["startedAt"]))
                self.assertNotEqual(proc.pid,os.getpid())
                stop.set()
                proc.join(10)
                self.assertEqual(proc.exitcode,0)
                states=[]
                while "stopped" not in states:
                    states.append(queue.get(timeout=2)[0])
            finally:
                if proc.is_alive():proc.terminate();proc.join()

    def test_remote_guard_timer_and_ssh_eof(self):
        started_at = datetime.fromtimestamp(100,timezone.utc).isoformat()
        for command in (b"test\n",b""):
            message=json.dumps(dict(key="fixture",pod=RTX_POD,generation=started_at,deadline=2000)).encode()+b"\n"+command
            with patch.object(live_worker.sys,"stdin",SimpleNamespace(buffer=io.BytesIO(message))), \
                    patch.dict(os.environ,RUNPOD_POD_ID=RTX_POD),patch("builtins.print") as output, \
                    patch.object(live_worker.time,"time",return_value=100), \
                    patch.object(live_worker.select,"select",side_effect=[([0],[],[]),([],[],[])]) as wait, \
                    patch.object(live_worker,"record_shutdown") as log, \
                    patch.object(live_worker,"verified_stop") as stop:
                live_worker.monitor()
                output.assert_called_once_with("guard-armed",flush=True)
                stop.assert_called_once_with("fixture",started_at,pod=RTX_POD)
                self.assertEqual(wait.call_args.args[3],60 if command else 885)
                self.assertIn("60-second" if command else "disconnected",log.call_args_list[0].args[1])
                self.assertIsNone(log.call_args_list[0].args[2])

    def test_worker_verifies_assembled_weight(self):
        with TemporaryDirectory() as tmp:
            path=Path(tmp)/"pose.onnx"
            parts=path.with_name(path.name+".parts")
            parts.mkdir()
            (parts/"000").write_bytes(b"ab")
            digest=hashlib.sha256(b"abcd").hexdigest()
            with self.assertRaises(ValueError):live_worker.verify_pose_weight(path,digest)
            self.assertFalse(path.exists())
            (parts/"001").write_bytes(b"cd")
            live_worker.verify_pose_weight(path,digest)
            self.assertEqual(path.read_bytes(),b"abcd")
    def test_quote_and_confirmation(self):
        for rate in (0,0.700001,2.25,float("nan")):
            with self.assertRaises(ValueError): g.preflight(STOP,rate)
        g.preflight(STOP,0.70)
        with patch.object(g,"urlopen") as request:
            with self.assertRaises(ValueError): g.api("fixture","start")
            request.assert_not_called()
        with patch("live_manager.api",return_value=STOP):
            manager=LiveManager("fixture")
            with self.assertRaises(ValueError):manager.begin("tab",{"confirmed":False})
            self.assertIsNone(manager.process)
            with self.assertRaises(ValueError): manager.infer("second-tab","frame",{})

    def test_4090_verifies_original_start_price_hardware_and_mount(self):
        generation = RTX_RUN["startedAt"]
        profile = g.PodProfile.rtx4090(RTX_POD, .81, generation)
        with patch("live_manager.api",return_value=RTX_RUN) as api:
            manager = LiveManager("fixture", profile=profile)
            status = manager.status(None)
            self.assertEqual((status["pod"],status["gpu"],status["rate_cap"]), (RTX_POD,g.RTX_4090,.81))
            self.assertFalse(status["can_stop"])
            with self.assertRaisesRegex(ValueError,"Watchdog has not verified"):
                manager.stop()
            with patch.object(manager,"launch_guard") as launch:
                manager.arm()
                launch.assert_called_once_with(None,generation)
            api.assert_called_once_with("fixture",pod=RTX_POD)
        for changes in (
            dict(id=g.POD), dict(gpu={"id":"RTX 4090 D","count":1}), dict(gpu={"id":"RTX 4090","count":2}),
            dict(dataCenterId="OTHER"), dict(mounts={"network":[{"path":"/workspace","volumeId":"other"}]}),
            dict(status="EXITED"), dict(cost=.810001), dict(cost=float("nan")), dict(cost=True),
            dict(startedAt=datetime.fromtimestamp(g.deployment_time(generation)-1,timezone.utc).isoformat()),
        ):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                g.attach_preflight(dict(RTX_RUN,**changes),.81,profile=profile,setup=True)
        with patch.object(g.time,"time",return_value=g.deployment_time(generation)+g.SETUP):
            with self.assertRaisesRegex(ValueError,"Five-minute setup deadline"):
                g.PodProfile.rtx4090(RTX_POD,.81,generation)
            with TemporaryDirectory() as tmp, patch.dict(os.environ,LOCALAPPDATA=tmp), \
                    patch.object(g,"api",return_value=RTX_RUN) as api:
                queue=Queue()
                g.guard("fixture",.81,threading.Event(),SimpleNamespace(value=0),SimpleNamespace(value=0),queue,
                        generation,SimpleNamespace(value=b""),profile)
                api.assert_called_once_with("fixture",pod=RTX_POD)
                state, fields = queue.get()
                self.assertEqual(state,"error")
                self.assertIn("Five-minute setup deadline",fields["detail"])

    def test_4090_guard_refuses_generation_change_during_arming(self):
        generation = RTX_RUN["startedAt"]
        profile = g.PodProfile.rtx4090(RTX_POD,.81,generation)
        replacement = dict(RTX_RUN,startedAt=datetime.now(timezone.utc).isoformat())
        with TemporaryDirectory() as tmp, patch.dict(os.environ,LOCALAPPDATA=tmp), \
                patch.object(g,"old_watchdog_check") as check, \
                patch.object(g,"api",side_effect=[RTX_RUN,replacement]) as api:
            queue = Queue()
            g.guard("fixture",.81,threading.Event(),SimpleNamespace(value=0),SimpleNamespace(value=0),queue,
                    generation,SimpleNamespace(value=b""),profile)
            state, fields = queue.get()
            self.assertEqual(state,"error")
            self.assertIn("session changed",fields["detail"])
            check.assert_called_once_with(RTX_POD)
            self.assertEqual([call.args for call in api.call_args_list],[("fixture",),("fixture",)])
    def test_stop_needs_confirmation_and_generation(self):
        with patch.object(g,"api",side_effect=[RUN,None,RUN,None]),patch.object(g.time,"sleep"):
            self.assertFalse(g.verified_stop("fixture",RUN["startedAt"],2))
        with patch.object(g,"api",return_value=RUN) as api:
            with self.assertRaises(RuntimeError):g.verified_stop("fixture","other",1)
            self.assertEqual(api.call_count,1)
    def test_emergency_during_manual_start_stops_the_delayed_generation(self):
        stop=threading.Event()
        queue=Queue()
        actions=[]
        pending=iter([STOP,STOP,STOP,RUN])
        current=[None]
        def api(key,action=None,*,pod=g.POD):
            if action:
                actions.append(action)
                current[0]=dict(RUN,status="EXITED")
            else:
                current[0]=next(pending,current[0])
                if current[0] is STOP and queue.qsize() >= 1:
                    stop.set()
            return current[0]
        with TemporaryDirectory() as tmp, patch.dict(os.environ,LOCALAPPDATA=tmp),patch.object(g,"api",api), \
                patch.object(g,"old_watchdog_check"),patch.object(g.time,"sleep"):
            g.guard("fixture",0.70,stop,SimpleNamespace(value=0),SimpleNamespace(value=0),queue,
                    None,SimpleNamespace(value=b"Emergency stop requested"))
            self.assertIn("Emergency stop requested",(Path(tmp)/"SignToCap/live-shutdown.log").read_text())
        self.assertEqual(actions,["stop"])
        self.assertEqual(list(queue.queue)[-1][0],"stopped")
    def test_stale_guard_blocks_charge(self):
        with TemporaryDirectory() as tmp, patch.dict(os.environ,LOCALAPPDATA=tmp),patch.object(g,"api",return_value=STOP) as api, \
                patch.object(g,"old_watchdog_check",side_effect=g.PreflightError("stale")):
            queue=Queue()
            g.guard("fixture",0.70,threading.Event(),SimpleNamespace(value=0),SimpleNamespace(value=0),queue,
                    None,SimpleNamespace(value=b""))
            api.assert_called_once_with("fixture",pod=g.POD)
            self.assertEqual(queue.get()[0],"error")
    def test_broken_ssh_requests_shutdown(self):
        with patch("live_manager.api",return_value=STOP):
            m=LiveManager("fixture")
        m.owner="tab";m.state="ready";m.live=True;m.stop_request=threading.Event()
        m.stop_reason=SimpleNamespace(value=b"")
        m.process=Mock();m.process.is_alive.return_value=True
        m.ssh=Mock();m.ssh.stdin.write.side_effect=BrokenPipeError
        with self.assertRaises(ValueError):m.infer("tab","commit",{})
        self.assertTrue(m.stop_request.is_set())

    def test_attach_uses_the_armed_generation_without_launching_another_watchdog(self):
        generation = RTX_RUN["startedAt"]
        profile = g.PodProfile.rtx4090(RTX_POD,.81,generation)
        body=dict(confirmed=True,attach=True,pod=RTX_POD)
        with patch("live_manager.api",return_value=RTX_RUN) as api:
            m=LiveManager("fixture",profile=profile)
            with self.assertRaisesRegex(ValueError,"session or shutdown"):
                m.begin("tab",body)
            m.state,m.generation="setup",generation
            m.stop_request=threading.Event()
            m.process=Mock()
            with patch.object(m,"ssh_command",return_value=["ssh"]), patch.object(m,"launch_guard") as launch, \
                    patch.object(m,"connect"):
                m.begin("tab",body)
                self.assertEqual(m.state,"loading")
                launch.assert_not_called()
                with self.assertRaises(ValueError): m.begin("tab",body)
                m.state="setup"
                api.return_value=dict(RTX_RUN,startedAt="different")
                with self.assertRaisesRegex(ValueError,"session changed"): m.begin("tab",body)

    def test_worker_identity_and_heartbeat_after_readiness(self):
        for marker in ("pod-verified\n",""):
            with self.subTest(marker=bool(marker)), patch("live_manager.api",return_value=RUN):
                m=LiveManager("fixture")
                m.owner,m.state,m.stop_request="tab","loading",threading.Event()
                m.began=g.deployment_time(RUN["startedAt"])
                m.heartbeat=SimpleNamespace(value=0)
                m.stop_reason=SimpleNamespace(value=b"")
                m.process=Mock()
                m.status("tab")
                self.assertEqual(m.heartbeat.value,0)
                with patch("live_manager.subprocess.Popen") as spawn:
                    proc=spawn.return_value
                    proc.wait.return_value=42
                    proc.stdout.readline.side_effect=[marker,'{"ready":true}\n']
                    m.connect(dict(generation=RUN["startedAt"]),["ssh"],m.stop_request)
                if marker:
                    self.assertTrue(m.status("tab")["ready"])
                    self.assertGreater(m.heartbeat.value,0)
                    last_seen = m.heartbeat.value
                    self.assertFalse(m.status("other-tab")["ready"])
                    self.assertEqual(m.heartbeat.value,last_seen)
                    proc.stdin.write.assert_called_once()
                else:
                    proc.stdin.write.assert_not_called()
                    self.assertTrue(m.stop_request.is_set())
                    self.assertIn(b"RUNPOD_POD_ID",m.stop_reason.value)

    def test_guard_lifecycle_with_virtual_clock(self):
        for scenario, reason in {"live_timeout":"60-second","full_timeout":"15-minute",
                                 "disconnected":"Browser heartbeat","setup_timeout":"Setup deadline",
                                 "price_changed":"rate exceeds","wrong_gpu":"GPU or data center",
                                 "changed_generation":"session changed"}.items():
            with self.subTest(scenario=scenario):
                clock=[1000.]
                heartbeat=SimpleNamespace(value=0 if scenario=="setup_timeout" else clock[0])
                test=SimpleNamespace(value=0)
                actions=[]
                generation=datetime.fromtimestamp(clock[0],timezone.utc).isoformat()
                running=dict(RTX_RUN,startedAt=generation)
                profile=g.PodProfile(RTX_POD,g.RTX_4090,.81,generation)
                def api(key, action=None, *, pod=g.POD):
                    if pod != RTX_POD:
                        raise ValueError("Wrong fixture Pod")
                    if action == "stop":
                        actions.append(action)
                        running["status"]="EXITED"
                    elif clock[0] > 1000:
                        if scenario=="price_changed": running["cost"]=2.8
                        if scenario=="wrong_gpu": running["gpu"]={"id":"A100","count":1}
                        if scenario=="changed_generation": running["startedAt"]="replacement"
                    return dict(running)
                class Tick:
                    def wait(self,seconds):
                        clock[0]+=seconds
                        if scenario not in {"disconnected","setup_timeout"}: heartbeat.value=clock[0]
                        if scenario=="live_timeout" and clock[0]>=1005 and not test.value:
                            test.value=clock[0]
                        return False
                    def is_set(self):return False
                with TemporaryDirectory() as tmp, patch.dict(os.environ,LOCALAPPDATA=tmp), \
                        patch.object(g,"old_watchdog_check"),patch.object(g,"api",api), \
                        patch.object(g.time,"time",lambda:clock[0]),patch.object(g.time,"sleep"):
                    queue=Queue()
                    g.guard("fixture",.81,Tick(),heartbeat,test,queue,generation,SimpleNamespace(value=b""),profile)
                final, fields = queue.queue[-1]
                if scenario=="changed_generation":
                    self.assertEqual(actions,[])
                    self.assertEqual(final,"shutdown_unverified")
                else:
                    self.assertEqual(actions,["stop"])
                    self.assertEqual(final,"stopped")
                self.assertIn(reason,fields["detail"])
                if scenario=="live_timeout": self.assertEqual(clock[0]-test.value,60)
                if scenario=="full_timeout": self.assertEqual(clock[0]-1000,885)
                if scenario=="disconnected": self.assertEqual(clock[0]-1000,20)
                if scenario=="setup_timeout": self.assertEqual(clock[0]-1000,300)


if __name__=="__main__":
    unittest.main()
