# SignToCap Phase 3

## Local preview and competition fallback

From the workspace in PowerShell:

```powershell
python -B phase3/desktop.py
```

Open [the local preview](http://localhost:8765). Select **Recorded replay** and a track, then press **F8** to start or stop. This just plays measured Phase 2 output on the original emission clock. In these recordings the model committed zero captions, so the replay final pane stays empty. **Copy captions** labels copied replay text. In **Live camera** mode, **Enable camera** shows your local preview when you grant camera permission. The default server leaves live translation disabled. The keyboard shortcuts work while this window has focus.

For the competition, use the labeled replay if translation quality or live transport cannot be independently verified in time. Share the window in a meeting. If OBS is already installed, you can add the browser window as a Window Capture source and click **Start Virtual Camera** in OBS. OBS and native virtual-camera output have not been tested here.

## Pose audit and decision

`audit-data/pose-audit.json` gives matching publisher RGB and pose filenames, RGB checksums, per-frame counts, decoded fresh-video cadence, confidence and missingness, and normalized same-index joint distances. The fresh extractor matches the package's configured COCO-WholeBody layout with x divided by image width and y divided by height. Published pose files carry no timestamp stream. Matching sample names and counts support direct comparison but do not independently prove the publisher's timestamps. Both kinds enter the same pinned Uni-Sign body, face and hand preprocessing.

All 10 audited clips matched frame-for-frame by count, totaling 2,352 frames per arm. Nine videos decoded at 24 fps and one at 60 fps. Across per-clip medians for joints with both scores above 0.3, the normalized distance was 0.31–0.43% for body, 0.20–0.27% for face and 0.47–0.82% for hands. Swapping the fresh hand groups increased pooled median error from about 0.6% to 28%. Neither arm produced all-zero frames or zero joint scores on these clips. Confidence values can exceed 1 in both arms. The low-score fraction at or below 0.3 was 21.4% publisher versus 21.7% fresh for body, and 0.59% versus 0.11% for the left hand. These measurements support matching layout and normalization on the sampled front-camera clips. Hand errors still need paired translation, and webcam lighting, framing and coverage remain open.

The independent sample selection excludes the previous 32 clip groups and both continuous source groups, selects one clip per remaining source group, and uses a fixed hash order with a 0.10–3.00 MB RGB-clip eligibility limit. The resulting six short clips are a readiness pilot. Source grouping cannot establish disjoint signer identities, and the sample does not cover webcam conditions. The historical continuous recordings are controller diagnostics only. Keep the held-out clips and all Phase 2 evaluation clips out of training.

**Decision:** no-go for LoRA until fresh and publisher gold-bounded outputs are acceptable on independent material. Phase 2 already showed semantic errors on publisher poses, so controller changes cannot remove the existing quality floor. Use the follow-up paired decode and an ASL-qualified reviewer to determine how much additional error fresh poses introduce. If pose differences lose useful content, fix the extraction path first. If oracle meaning passes and replay still fails, reconsider streaming adaptation. If both pose arms fail semantically, compare **one** alternative, Sign-Speak, conditional on approved access, rates, data terms, and separate consent before sending any video. This task neither calls the vendor nor estimates its quality. To score a later approved export, use `python phase3/evaluate_external.py phase3/audit-data/fresh.jsonl predictions.jsonl` in the benchmark environment. Provide one `{"id":"…","text":"…"}` per held-out clip. Use the same ASL review rubric in both arms.

For this small readiness pilot, have an ASL-qualified reviewer view each clip and rate blinded outputs as **preserved**, **partial**, **wrong**, or **missing**. Mark reversals, unsupported claims, and missing core actor/action details. Use BLEU and the English references for automated triage. Before rehearsing live, aim for preserved meaning on at least five of the six full-rate fresh clips and confirm that the 3-fps arm remains intelligible, then check useful-caption latency and explicit commitments on short consented webcam turns. Use replay if these checks cannot be completed before the competition.

The exact review clips are in `$env:LOCALAPPDATA\Temp\signgpt-phase3-audit` on this laptop, named by manifest ID. If the temporary cache is gone, regenerate it from the pinned release using the existing CPU environment. The audit reads individual ranges from the official archives and caches only the ten selected clips.

```powershell
$cache = Join-Path $env:LOCALAPPDATA 'Temp\signgpt-phase2-stage'
$rgb = Join-Path $env:LOCALAPPDATA 'Temp\signgpt-phase3-audit'
$env:XDG_CACHE_HOME = Join-Path $cache 'ort-cache'
& "$cache\cpu-venv\Scripts\python.exe" -B .\phase3\pose_audit.py `
  "$cache\labels.test" "$cache\prepared\how2sign-oracle\oracle.jsonl" "$cache\Uni-Sign" $rgb .\phase3\audit-data
```

## Targeted GPU experiment

The mirror on the existing volume was read back and verified by SHA-256 for all 33 Phase 3 objects, including the updated Phase 2 modules. If you edit the local files, restage them from the workspace. Otherwise proceed after explicit deployment approval. The staged Phase 2 weights and continuous tracks remain in use.

```powershell
# Run from the SignToCap repository root.
.\phase3\stage.ps1
```

Before clicking **Deploy**, in the same PowerShell window:

```powershell
$started = [DateTimeOffset]::UtcNow.ToUnixTimeSeconds()
$started
```

In **Pods → Deploy**, choose **L40S 48 GB**, **1 GPU**, **On-demand**, **EUR-IS-1** and the existing volume at `/workspace`. Verify the displayed rate at or below **$2.25/hour**. Use an SSH-enabled Linux image with Python 3.11 or 3.12 and venv support. After the approved deployment, copy its Pod ID and run the watchdog in the same PowerShell 5.1 window from the workspace. Use the separate RunPod REST key with Pod read and stop access at the hidden prompt.

```powershell
$pod = '<Pod ID from RunPod>'
$secret = Read-Host 'RunPod API key' -AsSecureString
$buffer = [Runtime.InteropServices.Marshal]::SecureStringToBSTR($secret)
try {
    $env:RUNPOD_API_KEY = [Runtime.InteropServices.Marshal]::PtrToStringBSTR($buffer)
    $watch = Start-Process -FilePath (Get-Command python).Source -WindowStyle Hidden -PassThru `
        -ArgumentList @('-B','-u','phase2/run_validation.py','watchdog','--pod',$pod,'--started',"$started") `
        -RedirectStandardOutput "$PWD\runpod-stop.log" -RedirectStandardError "$PWD\runpod-stop-error.log"
} finally {
    Remove-Item Env:RUNPOD_API_KEY -ErrorAction SilentlyContinue
    [Runtime.InteropServices.Marshal]::ZeroFreeBSTR($buffer)
    $secret.Dispose()
}
Get-Content .\runpod-stop.log -Tail 5
```

Confirm **Backup stop armed**. If it is missing, check `runpod-stop-error.log` immediately and stop the Pod from the console if you cannot arm the watchdog. Keep Windows awake and online. The total budget is **$10**, runtime cap **four hours**, and backup deadline **3.5 hours**. The follow-up has a shorter **55-minute** runner deadline and stops on completion. At the quoted **$1.10/hour**, a 30–60 minute allocation is **$0.55–$1.10** compute plus storage and tax.

At the Pod's SSH prompt, replace `DEPLOY_EPOCH` with the number printed in PowerShell:

```bash
started=DEPLOY_EPOCH
pod_python=$(command -v python3.12 || command -v python3.11)
findmnt /workspace
nvidia-smi -L
test -s /workspace/signgpt-validation/prepared/phase3/pose-audit.json
df -h /workspace
umask 077
read -rsp 'RunPod API key: ' RUNPOD_API_KEY; echo
export RUNPOD_API_KEY
nohup "$pod_python" -I -u /workspace/phase2/run_validation.py run \
  --started "$started" --dataset how2sign --phase3-followup \
  </dev/null >/workspace/signgpt-validation/phase3-runner.log 2>&1 &
unset RUNPOD_API_KEY
sleep 3
tail -12 /workspace/signgpt-validation/phase3-runner.log
```

One warm model process runs publisher/fresh oracle arms with identical four-beam settings. It keeps the diagnostic and independent splits separate. A separate run decodes the held-out fresh tracks at 3 fps to measure how the prototype capture cadence affects completed-sentence quality. It compares the historical controller and adaptive controller on the same two original continuous recordings. Adaptive uses a minimum 900 ms between preview decodes and a 24-token preview, plus stable prefix display and a full 48-token decode on commit. This tests the combined controller settings. No explicit human commit is fabricated during the timed replay.

Confirm RunPod shows **Stopped/Exited** after completion. Download the new timestamped result directory using the Phase 2 S3 commands. Review the paired references and predictions with an ASL-qualified reviewer before approving a live claim. The new follow-up and GPU shutdown were not exercised against a newly rented Pod during Phase 3 preparation.

## Short live test

Use the [60-second live test instructions](LIVE-TEST.md) for the Windows-managed SSH connection, optional recording and verified Pod shutdown. The 55-minute benchmark command above is for scientific evaluation.
