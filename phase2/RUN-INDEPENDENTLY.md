# Run the frozen validation without Codex

Use one fresh on-demand **L40S 48 GB** in **EUR-IS-1**. At the quoted **$1.10/hour**, 3.5 hours costs **$3.85** in GPU time. The approved ceilings are **$2.25/hour**, **$10 total**, and **four hours**. Both modes check the Pod's API rate and share the earlier of the 3.5-hour backup cutoff, the four-hour maximum, and the time affordable with $9 in compute.

The runner requests Stop on completion, error or timeout. The Windows backup requests Stop at **3 hours 30 minutes from immediately before deployment**, including startup and SSH setup. Keep the laptop awake and online until you confirm **Stopped/Exited**. Both stops depend on the API. If they fail, use the console **Stop** button before four hours or $10, whichever comes first. [RunPod v2 stop operation](https://docs.runpod.io/api-reference-v2/pods/trigger-a-pod-state-transition) · [Billing](https://docs.runpod.io/pods/pricing)

## How2Sign without webcam recordings

Use the updated `phase2` folder. Skip the webcam MP4 upload. In the remote launch command below, append `--dataset how2sign --official-continuous` after `--started "$started"`. The runner uses staged models and sample manifests when present. For the 32-clip baseline it checks decoder parity and writes `how2sign-oracle-events.jsonl` and `how2sign-oracle-summary.json` with reference text and SacreBLEU. If no sample is staged, it retrieves the same test members by exact byte range from the pinned archive and checks each ZIP CRC. Unknown clip cadence leaves input duration blank. The optional full downloader verifies SHA-256 for all four shards and needs about 17 GB for shards plus the joined ZIP. [Author pose files](https://huggingface.co/ZechengLi19/Uni-Sign/tree/main) · [Pinned test labels](https://github.com/ZechengLi19/Uni-Sign/blob/eed438bcb49e30405cd6ccdfcccca330c134e830/data/How2Sign/labels.test)

For continuous replay, the helper selects two original frontal test recordings by byte range and joins them with the official manually realigned annotations. It extracts pose windows with contiguous decoded cadence and retains the real intervening frames. The runner compares human-bounded clips with paced replay on those same windows using the same decoder settings. This is a small studio pilot. Inspect caption content against reference text and use the review annotations in the [runbook](RUNBOOK.md) to measure first useful and final-caption latency. Webcam and meeting latency still require a live trial. How2Sign and the checkpoint are published under noncommercial terms. [Official original files](https://dataverse.csuc.cat/dataset.xhtml?persistentId=doi%3A10.34810%2Fdata33) · [Checkpoint license](https://huggingface.co/ZechengLi19/Uni-Sign)

CPU staging commands use a Python 3.11 or 3.12 environment with NumPy, OpenCV, CPU ONNX Runtime, and tqdm and the pinned Uni-Sign checkout for RTMLib and labels. From the workspace, with `$stage` as a local temporary directory and `$repo` as that checkout:

```powershell
python -B phase2/prepare_weights.py "$stage/assets"
python -B phase2/how2sign_assets.py "$stage/prepared/how2sign-oracle" --sample-labels "$repo/data/How2Sign/labels.test"
python -B phase2/prepare_continuous.py "$stage/continuous" "$repo"
.\phase2\Stage-RunPod.ps1 -Stage $stage -Volume $volume -Part All
```

For webcam integration, keep one `StreamingSession` per signer session, send ordered COCO-WholeBody pose chunks through `push()`, and render the returned partial/final/uncertain events. The replay harness now uses the same session implementation. On disconnect, call `finish()` and create a new session when capture resumes. The class expects fixed-cadence poses with explicit handling of missing frames by the transport. The webcam client and full live transport still require integration and timing tests.

Before clicking **Deploy**, confirm the three staged completion files from PowerShell using your existing `$volume` and `$s3` values. If the Windows model transfer fails, the same runner can fetch any missing pinned weights after you manually deploy and arm the watchdog. Bootstrap verifies both published weight SHA-256 values. This fallback uses time inside the existing GPU budget.

```powershell
foreach ($key in @(
    'signgpt-validation/assets/staged.sha256',
    'signgpt-validation/prepared/how2sign-oracle/oracle.jsonl',
    'signgpt-validation/prepared/continuous/continuous.jsonl'
)) {
    aws s3api head-object --bucket $volume --key $key @s3 --query ContentLength --output text
    if ($LASTEXITCODE -ne 0) { throw "Not staged: $key" }
}
```

## 1. Stage your files before starting the GPU

1. Put **2–10 consented MP4 files** in a local folder named `signgpt-input` next to `phase2`. Use 20–45 s continuous recordings at constant 25 or 30 fps, with one clearly framed signer and a short trailing pause. Include a separate idle or fidgeting video named `idle-01.mp4` if available. Files beginning with `idle-` get idle-replay scoring and are excluded from completed-video translation.
2. In **Storage**, open the **existing 50 GB network volume**. Copy its volume ID, data center ID, and S3 endpoint URL. Confirm the selected L40S is in that same data center. Check the existing volume's storage charges and available capacity. [Network volumes](https://docs.runpod.io/storage/network-volumes) · [S3 locations](https://docs.runpod.io/storage/s3-api)
3. Under **Settings → Credentials**, use an **S3 API key** for the AWS CLI and a separate scoped **RunPod API key** for **REST v2 Pod read and stop**. Choose **Pods Read/Write** if the console groups those permissions, and leave unrelated scopes disabled. Use the RunPod key only at the hidden prompts in steps 2 and 3. Keep it out of S3, files, Pod template variables, and command history. Under **SSH Public Keys**, add your existing public key if needed. [S3 credentials](https://docs.runpod.io/storage/s3-api#setup-and-authentication) · [RunPod API keys](https://docs.runpod.io/get-started/credentials) · [SSH setup](https://docs.runpod.io/pods/configuration/use-ssh)
4. In PowerShell, open the workspace folder that contains `phase2`. With the [AWS CLI](https://docs.aws.amazon.com/cli/latest/userguide/getting-started-install.html) available, set up a separate local profile and upload the code and clips. In the AWS prompts, use the RunPod S3 access key and secret. Keep the PowerShell window open for the next steps.

```powershell
aws configure --profile signgpt-volume
$volume = '<volume ID>'
$region = '<data center ID>'
$endpoint = '<S3 endpoint URL>'
$s3 = @('--profile','signgpt-volume','--region',$region,'--endpoint-url',$endpoint)
aws s3 cp .\phase2 "s3://$volume/phase2/" --recursive @s3
aws s3 ls "s3://$volume/phase2/pose-imports.patch" @s3
aws s3 cp .\signgpt-input "s3://$volume/signgpt-input/" --recursive --exclude '*' --include '*.mp4' @s3
aws s3 ls "s3://$volume/signgpt-input/" @s3
```

Check that every expected clip appears before continuing.

## 2. Deploy and arm the backup stop

1. Open **Pods → Deploy**, choose **L40S 48 GB → 1 GPU → On-demand → EUR-IS-1**, and attach the existing network volume at `/workspace`. Use an SSH-enabled Linux image with Python 3.11 or 3.12 and venv support. Enable SSH and keep only SSH TCP access. Verify the displayed rate before deploying. The experiment installs PyTorch 2.6.0 and torchvision 0.21.0 CUDA 12.4 wheels in a separate venv and runs a BF16 GPU operation as a preflight. [PyTorch wheel matrix](https://docs.pytorch.org/get-started/previous-versions/) · [NVIDIA BF16 support](https://docs.nvidia.com/deeplearning/transformer-engine/support_matrix.html)
2. **Immediately before clicking Deploy**, run this in PowerShell. Keep this timestamp for the whole session. [RunPod Pod deployment](https://docs.runpod.io/pods/overview)

```powershell
$started = [DateTimeOffset]::UtcNow.ToUnixTimeSeconds()
$started
```

3. Click **Deploy**. As soon as the new Pod appears, copy its **Pod ID** and run the following in the same PowerShell window. Paste the separate RunPod API key at the hidden prompt. The background process wakes at the deadline and exits after checking or stopping that Pod. Closing SSH or Codex has no effect on it.

```powershell
$pod = '<Pod ID>'
$secret = Read-Host 'RunPod API key' -AsSecureString
$buffer = [Runtime.InteropServices.Marshal]::SecureStringToBSTR($secret)
try {
    $env:RUNPOD_API_KEY = [Runtime.InteropServices.Marshal]::PtrToStringBSTR($buffer)
    $watch = Start-Process -FilePath (Get-Command python).Source -WindowStyle Hidden -PassThru -ArgumentList @('-B','-u','phase2/run_validation.py','watchdog','--pod',$pod,'--started',"$started") -RedirectStandardOutput "$PWD\runpod-stop.log" -RedirectStandardError "$PWD\runpod-stop-error.log"
} finally {
    Remove-Item Env:RUNPOD_API_KEY -ErrorAction SilentlyContinue
    [Runtime.InteropServices.Marshal]::ZeroFreeBSTR($buffer)
    $secret.Dispose()
}
```

Open `runpod-stop.log` once and confirm it says **Backup stop armed**. If that message is absent and the error log shows a failure, fix it promptly or click **Stop** in the console. Keep Windows awake and connected until the Pod stops. The backup still works if the SSH session drops or the Pod workload crashes.

## 3. Start the detached benchmark

1. In **Pods**, open the same Pod and click **Connect → SSH**. Copy its direct SSH command and run it from PowerShell.
2. At the remote Linux prompt, replace `DEPLOY_EPOCH` below with the number printed in step 2. Paste the RunPod API key again at the hidden prompt. Start the job once.

```bash
started=DEPLOY_EPOCH
pod_python=$(command -v python3.12 || command -v python3.11)
"$pod_python" -V
nvidia-smi
df -h /workspace
umask 077
mkdir -p /workspace/signgpt-validation
read -rsp 'RunPod API key: ' RUNPOD_API_KEY; echo
export RUNPOD_API_KEY
nohup "$pod_python" -I -u /workspace/phase2/run_validation.py run --started "$started" \
  --dataset how2sign --official-continuous \
  </dev/null >/workspace/signgpt-validation/runner.log 2>&1 &
unset RUNPOD_API_KEY
sleep 3
head -20 /workspace/signgpt-validation/runner.log
```

Check that the Python command above reports 3.11 or 3.12, that `nvidia-smi` lists one L40S, and that `/workspace` is the volume with enough free space for the environment, model weights and results. If the runner reports a missing venv module, stop the Pod or use an image with Python 3.11/3.12 and its venv package. If the log shows a result directory and a setup command, you can disconnect. An early CUDA smoke failure stops the job before the weight downloads. If the log shows an immediate error, check that the Pod stopped and use **Stop** if needed. The job cuts off work two minutes before the backup deadline.

## 4. Download results after shutdown

In **Pods**, check that this Pod is **Stopped/Exited**. A Pod still running after the deadline needs a manual **Stop**. With the PowerShell variables from step 1 still set, download the outputs without restarting the GPU.

```powershell
New-Item -ItemType Directory -Force .\signgpt-results | Out-Null
aws s3 cp "s3://$volume/signgpt-validation/runner.log" .\signgpt-results\runner.log @s3
aws s3 cp "s3://$volume/signgpt-validation/results/" .\signgpt-results\ --recursive --exclude '*/poses/*' @s3
```

Open the timestamped folder and read `status.json`, `oracle-summary.json`, and `replay-summary.json`. The event logs contain the completed captions and incremental updates. Pose logs record extraction speed, and the summaries record model timing, GPU memory, dropped updates, caption revisions, and idle results where an idle clip was supplied. Human-reviewed meaning and first-useful/final caption latency require the review annotations in the [existing runbook](RUNBOOK.md). This replay uses prerecorded poses and does not measure live network or display latency.

Storage continues to bill after stop. Once you verify the download, revoke temporary keys you created for this run. Retain the existing network volume if you use it for other work. Deleting it removes all of its data. [RunPod storage lifecycle](https://docs.runpod.io/storage/network-volumes)
