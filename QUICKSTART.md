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
3. Wait for both watchdogs and **Model warmed**. The app shows **Pose CUDAExecutionProvider** when both pose models execute CUDA operators, or reports a CPU fallback and its reason. Click **Start Signing**, sign one short complete sentence, then click **Translate**. Review the caption. Recording and evidence export are optional and remain local.
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
