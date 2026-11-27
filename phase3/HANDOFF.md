# Paused on November 27, 2026

Work is saved in the workspace. The short live profile uses the Windows app, a separate Windows watchdog, a private SSH inference worker with a Pod-side deadline, and the existing frozen Uni-Sign model. The benchmark retains its separate entry point. This session issued no RunPod start, restart, deployment, or training request. The last user-confirmed Pod state was stopped. A new authenticated status read was not performed at handoff.

## Test record

| Check | Last result |
|---|---|
| Existing Phase 3 CPU/controller and replay regression suite | 5 passed using the two real prerecorded pose tracks. |
| Short-profile safety and integration suite | 10 passed locally. Includes a real separate Windows process with a mock Pod, paid-start gates, duplicate/inactive requests, stale-watchdog rejection, timing, SSH failure, emergency stop, verified vs unverified shutdown, and pose-weight assembly integrity. Provider transitions and deadline progression were simulated. |
| Edge end-to-end | Earlier passes covered original video, seeking, clipboard, synthetic-camera capture, mock inference and commitment, recording and playback after reload, paid-start cancel/confirm and emergency-stop warning. The latest run timed out waiting for camera videoWidth after reload. A camera-release change in the app and test was saved afterward. Its rerun is pending. |
| Offline media | Both original How2Sign videos were copied into the persistent local media folder and hash-matched to the Phase 2 originals. |
| Stale watchdog inspection | Found a legacy Phase 2 watchdog for the existing Pod, PID 18084 at inspection, scheduled for 09:45:44 UTC on November 27. Its current state is unknown. It was left alone. The new guard checks again at start. |
| S3 staging | First multipart transfer failed signature validation. A direct PUT succeeded for the small detector and failed for the larger RTMW model. The resumable upload was stopped at the user's request. Logs confirm chunks 000–012 of 16. Later chunks and the latest worker code were not verified remotely. |

## Resume locally

Use Windows PowerShell from the workspace. These commands run tests or stage files on the existing volume. They do not start a Pod.

```powershell
# Run from the SignToCap repository root.
$cache = Join-Path $env:LOCALAPPDATA 'Temp\signgpt-phase2-stage'
& "$cache\cpu-venv\Scripts\python.exe" -B .\phase3\test_phase3.py "$cache\continuous\prepared\continuous.jsonl"
python -B .\phase3\test_live.py
# Install Playwright locally before running the optional UI test.
node .\phase3\test_ui.cjs
```

After the browser test passes, finish the resumable upload. Check its exit code and success line. Compare the staged worker files and reassembled RTMW bytes to the local copies before approving compute.

```powershell
.\phase3\stage-live.ps1
```

Launch the current app with `python -B .\phase3\desktop.py --open`. The [GPU Control Center rehearsal guide](LIVE-TEST.md) covers replay, Pod discovery, rate confirmation, preparation, and the recorded live test.

## Outstanding hardware validation

A real run is still needed to test current L40S availability and price, scoped REST start permissions, direct SSH and host-key authentication, persistent Python/CUDA environment, worker initialization, genuine model outputs and end-to-end latency, and actual provider shutdown and billed duration. The source capture and recording tests used a synthetic camera, and timed safety tests used mock transitions. Keep LoRA on hold pending the separate paired benchmark and ASL-qualified semantic review.
