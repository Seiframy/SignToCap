# SignToCap inference benchmark

## Translation evidence

The selected complete sentence is the third annotated sentence in the How2Sign Roller Derby recording, `fzueHkUDyhE-8-rgb_front`. The English reference is “There are approximately 270 leagues around the world right now.”

The archived L40S decode of the annotated window returned “There's about two hundred and eighty leagues around the world now.” Its recorded model time was 328.5 ms. A fresh Windows CPU run decoded 81 time-verified source frames at 24 FPS from 22,540 to 25,840 ms, with outward rounding to frame boundaries. The frozen checkpoint returned the same text. It captured the general meaning and got the number wrong. Single-sentence evidence establishes caption generation and leaves translation accuracy open. [Archived L40S output](../signgpt-results/20261127T062446887049Z/continuous-oracle-events.jsonl)

The new **Prerecorded translation** mode takes one complete annotated sentence from the local catalog. It extracts poses at source cadence and sends a single full-window decode to the attached model. The path rejects incomplete decodes, variable cadence outside tolerance, windows over 256 frames, insufficient pose, and empty or repetitive output. It retains the English reference separately. The webcam mode still uses explicit sentence commitments and its 60-second watchdog.

## Before and after

The reported 60-second webcam session processed 37 frames, dropped 139, and produced no captions. It recorded 26 capture gaps and three incomplete-window events. The median pose stage was 908 ms and the median round trip was 1,117 ms against a 333 ms frame budget. A live commit requires at least two frames in the active window, sufficient visible pose, and a valid non-repetitive decode. At 3 FPS the controller caps a turn at 30 slots. The app puts detection and whole-body pose on the CPU. Uni-Sign's pose encoder and text generation use CUDA on the Pod. The webcam export contains no GPU utilization samples.

The first controlled local measurement kept the same 81 decoded source frames, the same two ONNX weights, and the same preprocessing. The tuned case uses two ONNX Runtime CPU threads per model and disables intra-op spinning.

| Metric | Runtime defaults | Two threads |
| --- | ---: | ---: |
| Video-to-poses wall time | 31.08 s | 14.66 s |
| First pose frame | 189.6 ms | 140.2 ms |
| Warm pose median | 272.2 ms | 138.2 ms |
| Warm pose p95 | 849.9 ms | 258.8 ms |
| Warm pose work rate | 2.72 FPS | 5.86 FPS |
| Maximum coordinate or confidence change | — | 0.0 |

An end-to-end comparison sent the same annotated sentence through the prerecorded HTTP handler with a verified local CPU checkpoint. With the model already loaded, video-to-caption wall time dropped from **36.79 s to 18.01 s**. Both responses used 81 real frames and returned identical text. This comparison includes video decode, pose extraction, model decode and the local HTTP request.

The fresh CPU checkpoint load took 27.53 s. On the same full pose window, the first decode took 2.94 s and the repeat took 3.40 s. These observations leave any wall-time benefit from model warm-up unproven. The webcam evidence now records pose and model initialization, JPEG decode, pose, worker processing, round trip, queue time, and queue overflow.

The browser now captures into a three-frame queue independently of the serial model request. The commit action preserves a boundary between pre-commit frames and the next turn. Missing slots invalidate the active window and advance the source clock without generating any replacement poses.

A deterministic 60-second, 3-FPS scheduling comparison isolates the queue policy. With a sustained 1,117 ms request time, completed frames rose from 45 to 53. Both schedules still had frequent gaps. With a 170 ms request time and every ninth request taking 650 ms, the blocking policy completed 162 frames with 17 gaps. The queue completed 179 frames with zero gaps and one in flight at the deadline. These are modeled scheduling results. The recorded webcam export provides the actual remote measurements above.

## L40S comparison

`profile_inference.py` benchmarks a complete source window with the original CPU defaults, the two-thread CPU configuration, and the experimental CUDA execution provider. The CUDA probe verifies that CUDA operators actually run in both the detector and the pose model. The benchmark reports pose and video-to-caption timings, repeated decode, pose differences, and GPU utilization sampled with `nvidia-smi`. CUDA stays outside the production default pending a measured comparison and a caption quality check. The separate [benchmark requirement](requirements-cuda-benchmark.txt) pins the available Linux x86-64 Python 3.12 wheel and its published hash for installation into an isolated target with dependency changes disabled. [ONNX Runtime CUDA compatibility](https://onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html) · [PyPI release files](https://pypi.org/project/onnxruntime-gpu/1.20.2/#files)

## RTX 4090 session on November 27, 2026

The user manually started RTX 4090 Pod because the L40S was unavailable. SignToCap verified the generation, pinned SSH through the authenticated RunPod terminal, and armed both watchdogs. The actual app returned the same caption for all 81 frames at 24 FPS. Pose extraction on Windows measured a 156.6 ms median and 495.7 ms p95. The first model decode on the RTX 4090 took 3,293.9 ms, including 2,071.2 ms for pose encoding and 1,150.2 ms for generation. The caption still said 280 where the reference says 270.

The isolated CUDA installation failed because the Pod's Python could not resolve `onnxruntime-gpu==1.20.1`. The watchdog subsequently stopped the Pod after a RunPod status check failed, before the stand-alone comparison produced results. CUDA operator assignment, sampled utilization, and a same-clip GPU warm-up comparison remain unmeasured. RunPod verified that the exact RTX generation was Exited at 20:54:13 Cairo time with no pending shutdown. SignToCap estimated 470 billable seconds and $0.117 in compute and container storage. The L40S remained Exited.

## Current worker

Attachment now stages the hash-pinned ONNX Runtime GPU 1.20.2 wheel installer and a real warm-up frame. The worker checks both pose models for executed CUDA operators and warms Uni-Sign before declaring readiness. The GPU Control Center displays the measured provider and warm-up time. A CPU fallback includes its cause. Evidence records batching, queue time, and per-frame pose time. A new Pod run is still needed to measure throughput and GPU utilization.

## Accuracy workflow

Local CPU inference compared the three fresh held-out clips that fit the 256-frame native window. The 24 FPS input preserved more of the reference meaning on two clips, and neither rate recovered the third. Adding real neighboring footage changed two of three diagnostic captions. Both changed cases had more motion in the added footage than in the annotated clip. The one low-motion case added three 6 FPS frames and produced the same caption. This does not establish the effect of seconds of idle footage. The app keeps 6 FPS and lets the signer end capture immediately. Raw comparisons are in `sampling-results.jsonl`.

### One guarded pilot

The prepared `pilot-data` contains 20 publisher training pose tracks from five source recordings. The local preflight assigned 16 clips to training and four to validation, with six separate held-out clips. It checks the annotation match, source isolation, pose format, official window duration, and duplicate pose content.

With the existing `signgpt-volume` AWS CLI profile, run this from the Windows project root. These commands check and stage files while GPUs are stopped.

```powershell
# Run from the SignToCap repository root.
$posePython = Join-Path $env:LOCALAPPDATA 'Temp/signgpt-phase2-stage/cpu-venv/Scripts/python.exe'
if (-not (Test-Path .\pilot-data\train.jsonl)) {
  & $posePython -B .\phase2\how2sign_assets.py .\pilot-data --training-annotations .\pilot-data\how2sign_realigned_train.csv
  if ($LASTEXITCODE) { throw 'Training subset preparation failed' }
}
$plan = 'pilot-check-' + (Get-Date -Format 'yyyyMMddHHmmss')
& $posePython -B .\phase3\accuracy_plan.py $plan --official-train .\pilot-data\train.jsonl --official-train-tsv .\pilot-data\how2sign_realigned_train.csv
if ($LASTEXITCODE) { throw 'Pilot data check failed' }
& $posePython -B .\phase3\train_pose_adapter.py $plan
if ($LASTEXITCODE) { throw 'Pilot pose or split check failed' }
$env:AWS_REQUEST_CHECKSUM_CALCULATION = 'when_required'
$env:AWS_RESPONSE_CHECKSUM_VALIDATION = 'when_required'
$s3 = @('--profile','signgpt-volume','--region','EUR-IS-1','--endpoint-url','https://s3api-eur-is-1.runpod.io/')
$dest = 's3://4q8k7ibb17/signgpt-pilot'
$files = @(
  'phase2/benchmark.py','phase2/captions.py','phase2/pose_data.py','phase2/unisign.py',
  'phase3/run-accuracy-pilot.sh','phase3/accuracy_plan.py','phase3/accuracy-settings.json',
  'phase3/train_pose_adapter.py','phase3/evaluate_external.py',
  'signgpt-results/20261127T062446887049Z/how2sign-oracle-events.jsonl',
  'signgpt-results/20261127T062446887049Z/continuous-oracle-events.jsonl'
)
foreach ($file in $files) {
  aws s3 cp $file "$dest/$file" @s3
  if ($LASTEXITCODE) { throw "Upload failed for $file" }
}
aws s3 sync .\phase3\audit-data "$dest/phase3/audit-data/" @s3
if ($LASTEXITCODE) { throw 'Held-out upload failed' }
aws s3 sync .\pilot-data "$dest/pilot-data/" @s3
if ($LASTEXITCODE) { throw 'Training upload failed' }
```

For the A100 pilot, verify the total compute plus container rate and approve a new hourly ceiling before manually starting one **NVIDIA A100 SXM 80 GB** on the existing volume. In SignToCap, click **Refresh Pods**, select that Pod, enter the approved **Spending ceiling ($/hour)**, and click **Attach & Prepare GPU**. Confirm the exact Pod, quote, and ceiling. Wait for both watchdogs and keep SignToCap open. Leave the webcam test idle. Proceed only with at least nine minutes left on the original 15-minute deadline. In that Pod’s authenticated RunPod web terminal run:

```bash
cd /workspace/signgpt-pilot
plan="$PWD/pilot-$(date -u +%Y%m%dT%H%M%SZ)"
bash phase3/run-accuracy-pilot.sh check "$PWD/pilot-data" "$plan" &&
  bash phase3/run-accuracy-pilot.sh run "$PWD/pilot-data" "$plan"
```

The runner caps training and paired inference at eight minutes combined. It trains only the pose projection for up to eight optimizer steps and writes a separate adapter. The original checkpoint is read-only to this workflow. The two score reports compare the adapter with its base at 6 FPS on fresh poses and at native sampling on publisher poses. When the command completes or fails, click **Stop GPU** in SignToCap and verify **Exited** in RunPod.

Keep the base for deployment unless fresh-pose chrF improves, publisher-pose chrF holds or improves, and a paired review finds more meaning-preserving fresh captions with no new errors in numbers, names, or negation. The six shared references make this a pilot. General webcam accuracy remains a separate question. The 15-minute compute and container cost ceiling is one quarter of the approved hourly ceiling. Shared storage is billed separately. A100 timing and peak GPU memory are unverified.
