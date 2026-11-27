# Quickstart

Ask the repository owner for access to the private GitHub repository. On Windows, install Git and Python 3.12 and open PowerShell.

```powershell
git clone https://github.com/Seiframy/SignToCap.git
Set-Location SignToCap
git switch handover/signtocap-2026-11-28
python -B .\phase3\desktop.py --open
```

If Git asks you to sign in, complete Git Credential Manager's browser sign-in yourself, then retry. Use `http://localhost:8765` if SignToCap does not open automatically.

## 1. Offline demonstration

Select **Recorded replay**, choose **Illustrative caption replay**, and click **Play**. This demonstrates caption timing and controls with a fixture. No model executes. For an authorized archived replay, launch with `--replay <path-to-events.jsonl>`. To show matching video in the app, restore the authorized catalog to `phase2-analysis/continuous-inputs.jsonl` and set `SIGNTOCAP_MEDIA` to the directory with the matching clips.

## 2. Webcam translation with the pretrained checkpoint

1. Ask the maintainer for access to the existing private RunPod network volume and its prepared `/workspace/signgpt-validation` runtime. Obtain an authorized real warm-up frame at `phase3/warmup.jpg` through the private asset handoff. Windows needs the OpenSSH client, an authorized SSH identity, and Python with `pythonw.exe`. In SignToCap, set a scoped RunPod Pod read/write key through **RunPod credential**, as needed. Use Windows Credential Manager if saving it.
2. Approve the exact all-in quote and manually start one supported single-GPU Pod with the matching region, volume and mount. In SignToCap choose **Sign & Translate**, click **Refresh Pods**, select that running Pod, enter the approved ceiling, and click **Attach & Prepare GPU**. Confirm the Pod and quote. Follow the authenticated host-key dialog if prompted.
3. Wait for both watchdogs and **Model warmed**. The app shows **Pose CUDAExecutionProvider** when both pose models execute CUDA operators, or reports a CPU fallback and its reason. Select **6 FPS** for the known fallback, or **12 FPS** or **24 FPS** for experimental capture. Click **Start Signing**, sign one short complete sentence, then click **Translate**. Review the caption. Recording and evidence export are optional and remain local.
4. Confirm shutdown in SignToCap and **Exited** in RunPod. If needed, use **Stop GPU** and reconcile the displayed Pod. See [the GPU runbook](phase3/LIVE-TEST.md) for setup and shutdown details.

## 3. Reproduce the H100 pilot in the RunPod web terminal

Obtain access to the existing private volume with the pinned runtime and weights in `/workspace/signgpt-validation` and pilot code and authorized data staged under `/workspace/signgpt-pilot`. The pilot needs official training annotations and pose tracks, the two matched held-out pose manifests and tracks, and the archived evaluation manifests used to exclude prior source recordings. If restaging, use the offline preparation and S3 steps in [the pilot runbook](phase3/INFERENCE-BENCHMARK.md).

Approve the actual H100 SXM quote before manually starting the Pod. Attach through SignToCap and wait for both watchdogs. Keep SignToCap open, leave the webcam test idle, and start only with at least nine minutes left on the original Pod deadline. In the authenticated terminal for that exact Pod, run:

```bash
cd /workspace/signgpt-pilot
plan="$PWD/pilot-$(date -u +%Y%m%dT%H%M%SZ)"
bash phase3/run-accuracy-pilot.sh check "$PWD/pilot-data" "$plan" &&
  bash phase3/run-accuracy-pilot.sh run "$PWD/pilot-data" "$plan"
```

Read the paired `fresh-score.json` and `publisher-score.json` under `"$plan/run"`. The job has an eight-minute limit. When it finishes or fails, use **Stop GPU** and verify **Exited**. The original checkpoint is preserved and adapter adoption requires review.

## Upload a video on the Phase 4 branch

Select **Upload Video & Translate** and choose one complete ASL sentence in an MP4 or another file readable by the local OpenCV decoder. **Inspect video** reports source duration and FPS without a GPU or model. The browser previews formats it can play. The decoder needs the existing Windows pose runtime at `%LOCALAPPDATA%\Temp\signgpt-phase2-stage`, or the directory named by `SIGNTOCAP_POSE_STAGE`.

With a guarded worker ready in the same tab, select 6, 12, or 24 FPS and click **Translate video**. The app shows local pose extraction, model translation, and timing. Use **Copy captions** to copy the result. Limits are 64 MiB, 60 seconds, and 256 sampled frames. Use constant-frame-rate footage and a sampling rate at or below source FPS. Temporary uploads are deleted after processing. Only poses go to the worker. Use **Stop GPU** after the last upload and verify **Exited**.

### Next guarded GPU validation

1. Restore the authorized warm-up frame and prepared volume. For automated scores, stage the Phase 4 `phase2` and `phase3` code and authorized catalog and source clips in the same folder layout on the private volume. Approve the all-in rate, manually start one supported Pod, attach it in SignToCap, and wait for both watchdogs.
2. To collect chrF and BLEU, run this command from the staged root in the attached Pod's authenticated terminal with the webcam idle. Set `CATALOG`, `MEDIA`, and `CLIP` to the authorized files and annotated item ID. Allow enough time for shutdown and only begin the job with at least seven minutes remaining.

```bash
work=/workspace/signgpt-validation
timeout 6m "$work/.venv-py312/bin/python" -B phase3/phase4_benchmark.py \
  "$CATALOG" "$MEDIA" "$PWD/phase4-validation" --clip "$CLIP" --work "$work" --device cuda
```

   The command uses CPU pose extraction and CUDA Uni-Sign. Keep both reports private because the JSON contains the reference and generated captions.

3. With time remaining, use **Upload Video & Translate** on the same short annotated 24 FPS clip at **6**, **12**, and **24 FPS**. Record sample counts, timing, and captions. If time remains, test **Sign & Translate** at 12 FPS and export evidence locally to measure CUDA pose and transport. The webcam sentence requests shutdown. If you skip it, click **Stop GPU** after the final upload or benchmark. Verify **Exited** in RunPod.

### Offline verification

From the Phase 4 checkout with the Windows pose runtime available, run:

```powershell
$posePython = Join-Path $env:LOCALAPPDATA 'Temp\signgpt-phase2-stage\cpu-venv\Scripts\python.exe'
$env:PYTHONPATH = (Resolve-Path .\phase3).Path
& $posePython -B -m unittest test_phase4 test_phase3.ControllerTests.test_loopback_replay_and_live_guard
if ($LASTEXITCODE) { throw 'Python offline tests failed' }
node .\phase3\test_capture.cjs
```
