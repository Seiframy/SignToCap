# Phase 3

The Windows desktop broker and UI live here. [Quickstart](../QUICKSTART.md) covers replay and live attachment. [LIVE-TEST.md](LIVE-TEST.md) documents the GPU safeguards. [INFERENCE-BENCHMARK.md](INFERENCE-BENCHMARK.md) contains historical measurements and the guarded pilot procedure.

`desktop.py` is the entry point. `gpu_control.py`, `gpu_setup.py`, `live_manager.py`, `live_guard.py`, and `live_worker.py` handle the GPU lifecycle and inference. `ui/` contains the loopback client and local evidence exporter. The pilot uses `accuracy_plan.py`, `train_pose_adapter.py`, and `evaluate_external.py`. Obtain restricted assets through the private handoff described in [Project context](../PROJECT_CONTEXT.md).
