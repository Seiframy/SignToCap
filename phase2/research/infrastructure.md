# Phase 2 infrastructure findings

Verified on November 26, 2026. Pricing below is in USD. RAM, disk and PATH checks are from the current laptop session. GPU runtime and VRAM figures are planning assumptions until measured.

## Laptop and preparation

- HP Envy x360 15-fe0xxx with Intel Core i7-1355U, 12 logical CPUs, approximately 15.7 GiB OS-visible RAM and 68.7 GiB free on C:. The registered display adapter is Intel Iris Xe and `nvidia-smi` is absent from PATH. Use this machine for manifests, replay/control-plane tests and later CPU pose benchmarking. [Intel CPU specifications](https://www.intel.com/content/www/us/en/products/sku/232160/intel-core-i71355u-processor-12m-cache-up-to-5-00-ghz/specifications.html)
- Windows x64 build 26300.10075, PowerShell 7.6.9, Python 3.14.3, Node 22.14.0, Git 2.56.1, `uv`, Windows OpenSSH client and SCP. `py -0p` listed only Python 3.14.3. `torch`, `cv2`, `onnxruntime`, `fastapi` and `pytest` were absent from that interpreter's isolated import search. `rustc`, `cargo`, `docker` and `ffmpeg` were absent from PATH. WSL is on PATH, its distribution query returned access denied, and CIM was unavailable. Run model steps in an isolated Linux environment. Dependency-free tools can use the current Windows Python.

## Pod configuration

| Use | GPU and VRAM | Secure Cloud public rate |
| --- | --- | ---: |
| First inference and training benchmark | L40S, 48 GB, 94 GB host RAM, 16 vCPU | $1.09/h |
| Same-memory alternative | L40, 48 GB, 94 GB host RAM, 8 vCPU | $0.82/h |
| Lower-cost inference after sizing | RTX 4090, 24 GB, 41 GB host RAM, 6 vCPU | $0.89/h |
| More memory if profiling warrants it | A100 PCIe, 80 GB, 117 GB host RAM, 8 vCPU | $2.09/h |

Check the actual Pod rate, availability, region and direct SSH access in the console before launching. The same public page lists A100 PCIe Community Cloud from $1.19/h. The report's $0.34/h RTX 4090 and $0.79/h L40S Community figures were absent from the accessible canonical public listing. Choose a single on-demand GPU, 30 GB temporary container disk, no exposed HTTP ports and only SSH over TCP for the private demo. Select a responsive Secure Cloud region from Cairo with the GPU in stock. Use a 40 GB encrypted volume disk for a single baseline session, or an appropriately sized data-center network volume for repeated sessions. [Runpod GPU pricing](https://www.runpod.io/pricing) · [Pod billing](https://docs.runpod.io/pods/pricing) · [SSH configuration](https://docs.runpod.io/pods/configuration/use-ssh)

The released How2Sign checkpoint is 1.19 GB and the four pose shards total approximately 8.47 GB. Combining the archives uses about 16.94 GB until the original shards are removed. Check the uncompressed size and select members from the training manifest. Size training storage for `shards + assembled archive + selected extracted data + base weights + two recovery checkpoints`, plus 20% margin. Mount files that must persist at `/workspace`. Volume disk persists through stop/restart on the same machine and is deleted on termination. Network volume persists independently and is scoped to its data center. Export recovery artifacts before termination. A stopped Pod may restart without a GPU if its machine no longer has one available. Pod billing runs per second. Container and running Pod disks are $0.10/GB-month each. Stopped Pod volume disks are $0.20/GB-month. Standard network volumes below 1 TB are $0.07/GB-month and remain billable independently. Runpod documents no Pod ingress or egress charge. [Official checkpoint](https://huggingface.co/ZechengLi19/Uni-Sign/blob/main/how2sign_pose_only_slt.pth) · [Pose shard listing](https://huggingface.co/ZechengLi19/Uni-Sign/tree/main) · [Storage types](https://docs.runpod.io/pods/storage/types) · [Billing](https://docs.runpod.io/pods/pricing) · [Restart recovery](https://docs.runpod.io/pods/troubleshooting/zero-gpus)

## Environment and preflight

Use SSH-ready `runpod/pytorch:2.2.0-py3.10-cuda12.1.1-devel-ubuntu22.04` as a host image and create an isolated Python 3.9 environment for the upstream model. Its bundled PyTorch 2.2 is a separate preinstall. The upstream environment specifies Python 3.9 and PyTorch `2.1.1+cu121`, TorchVision `0.16.1+cu121`, TorchAudio `2.1.1+cu121`, Transformers `4.40.0`, tokenizers `0.19.1` and sentencepiece `0.1.97`. Use the PyTorch CUDA 12.1 wheel index for the Torch packages. Avoid Python 3.11+ for a literal upstream install because the upstream SciPy `1.7.3` pin requires Python `<3.11`. Prefer the model-specific minimal dependency list in the main runbook for the experimental wrapper. The upstream dataset module imports Decord and calls utilities that import DeepSpeed even for pose-only data. Retain those dependencies when importing those modules directly. TensorFlow/BLEURT can use a separate evaluation environment if needed. [Official Runpod image](https://www.runpod.io/articles/guides/pytorch-2-2-cuda-12-1) · [Upstream install](https://github.com/ZechengLi19/Uni-Sign#%EF%B8%8F-installation) · [Upstream pins](https://github.com/ZechengLi19/Uni-Sign/blob/main/requirements.txt) · [PyTorch CUDA 12.1 wheels](https://docs.pytorch.org/get-started/previous-versions/#v211) · [SciPy metadata](https://pypi.org/project/scipy/1.7.3/) · [Dataset imports](https://github.com/ZechengLi19/Uni-Sign/blob/main/datasets.py) · [Utility imports](https://github.com/ZechengLi19/Uni-Sign/blob/main/utils.py)

If packaging an independent container later, the official `pytorch/pytorch:2.1.1-cuda12.1-cudnn8-devel` Linux amd64 image has manifest `sha256:b3d31523ed2f8cfe3f0af826bfccc8836c119cb5a06bc0c86ae82e611e1317e0` and is 8.25 GB compressed. That image needs a deliberate SSH entrypoint when using full Pod SSH. Resolve and record the chosen image digest, Git revision, tokenizer and checkpoint revisions and SHA256 values. Save `python --version`, `python -m pip freeze`, `python -m pip check`, `nvidia-smi` and the exact settings in each run directory. [Official PyTorch image manifest](https://hub.docker.com/layers/pytorch/pytorch/2.1.1-cuda12.1-cudnn8-devel/images/sha256-b3d31523ed2f8cfe3f0af826bfccc8836c119cb5a06bc0c86ae82e611e1317e0?context=explore) · [Runpod container guidance](https://github.com/runpod/runpod-plugins-official/blob/main/plugins/runpod/skills/runpod-usage/reference/building-images.md)

For pose extraction, use the vendored RTMLib revision with the CPU provider for initial parity checks. A Python 3.9 Linux CPU wheel exists for `onnxruntime==1.18.0`. If the CPU pose stage misses the frame budget, the CUDA 12 build of `onnxruntime-gpu==1.18.0` matches cuDNN 8 and comes from Microsoft's separate CUDA 12 package feed. The same PyPI version targets CUDA 11.8. Verify actual loaded providers and one frame inference before timing. [CPU wheel](https://pypi.org/project/onnxruntime/1.18.0/) · [CUDA compatibility](https://onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html#requirements) · [Official package sources](https://onnxruntime.ai/docs/install/#cuda-12x)

Run the following baselines with a single loaded model and batch size 1. Ensure that the reference decode follows the checkpoint's released settings. Compare greedy decoding, fewer output tokens and the candidate streaming cadence on the same held-out clips. Keep fresh sessions per candidate configuration. After warmup, synchronize the device around GPU-timing measurements, reset peak allocator stats and save peak allocated, peak reserved and `nvidia-smi` device usage. Record capture-to-render latency independently on the client. Cross-process CUDA consumers such as ONNX Runtime are visible in device sampling. Profile a longest-window forward, one optimizer step and 20 optimizer steps before projecting the full run. [PyTorch CUDA allocator](https://docs.pytorch.org/docs/stable/notes/cuda#memory-management)

```bash
nvidia-smi --query-gpu=timestamp,name,memory.used,memory.total,utilization.gpu \
  --format=csv -lms 250 -f /workspace/sign-gpt/runs/current/gpu.csv
```

mT5-base alone is about 580M parameters, or about 1.1 GiB of BF16 weight storage. That estimate excludes the pose network, activations, generation cache, runtime allocations and checkpoint load transients. Treat 4–12 GiB inference and 12–30 GiB adapter training as provisional capacity-planning ranges for a single model at batch 1, with no claim that these are measured. Consider a 24 GB inference card only if combined peak use stays below 18 GB on the longest supported input and that card meets live latency and quality gates. If training nears 40 GB on L40S, first check micro-batch 1, fixed input lengths and frozen components. Price an A100 only after a measured memory bottleneck persists. [mT5 model family](https://huggingface.co/google/mt5-base) · [mT5 paper](https://aclanthology.org/2021.naacl-main.41/)

For training planning, compute `steps × median measured optimizer-step seconds + validation + checkpoints + setup` and rerun the estimate after a 20-step profiling job. A small adapter trial can provision 2–6 GPU-h including setup, debugging and held-out evaluation, with a hard stop and no automatic extension. Data download and extraction, packaging and checkpoint time should be logged separately so they cannot be mistaken for model throughput.

## Storage and recovery layout

```text
/workspace/sign-gpt/
  artifacts/    pinned models and checksums
  cache/        prepared, reusable data
  runs/<id>/    source revisions, resolved environment, manifest, metrics
                checkpoints/latest.pt, checkpoints/best adapter, log
```

Write the last resume state atomically through a temporary file in the same directory at fixed optimizer-step intervals. Keep best and latest checkpoints, with one previous latest during verification. Resume state includes trained tensors, optimizer and scheduler states, optional scaler, global step, seed/RNG and data ordering cursor, dataset manifest hash and base checkpoint hash. Export the smaller inference artifact separately. Reopen a saved state and run one bounded step before paying for a longer trial. Export metrics, manifests, trained weights and checksums off the Pod at each decision gate. Keep webcam media out of logs and shared volumes.

## Deployment

For a private benchmark on the Pod, keep Uvicorn/FastAPI on `127.0.0.1:8000` and use its WebSocket through authenticated full SSH. Request the direct SSH command from the Runpod Connect tab. The basic proxied SSH method omits SCP and SFTP. Make liveness and readiness separate and set readiness only after the model is loaded and warmed. Keep a single GPU worker and bound outstanding requests per session. Issue explicit reset and commit events. Time or cap reconnect retries and clear interrupted segment state. Keep audio, video and pose data out of routine server logs. [Full SSH](https://docs.runpod.io/pods/configuration/use-ssh) · [Runpod port behavior](https://docs.runpod.io/pods/configuration/expose-ports)

```bash
ssh -N -o ExitOnForwardFailure=yes -o ServerAliveInterval=20 \
  -L 127.0.0.1:8000:127.0.0.1:8000 -p SSH_PORT root@POD_IP
```

For an authorized shared demo use TLS and an expiring per-session bearer token, keep the service single-tenant and add a 15–30 minute continuous connection/reconnect test. Runpod HTTP proxy is publicly accessible and Runpod recommends TCP exposure for persistent WebSockets. A direct public TCP endpoint needs its own TLS termination and authentication. A worker pool for independent clip requests can be considered separately from stateful meetings. [HTTP and TCP exposure](https://docs.runpod.io/pods/configuration/expose-ports)

For local inference, profile the RTMPose-compatible local CPU path separately from decoder inference on the actual laptop with a meeting client running. Only after meeting the streaming quality gate, export the pose/projection encoder and decoder with explicit past-key/value cache as appropriate. Validate dynamic mask, variable-length and token parity in FP32, then try ONNX Runtime dynamic INT8 on supported transformer operations and measure latency, RSS and semantic regression. Report local inference as unproven until warm replay passes the same gates. [ONNX Runtime transformer optimization](https://onnxruntime.ai/docs/performance/transformers-optimization.html) · [ONNX Runtime quantization](https://onnxruntime.ai/docs/performance/model-optimizations/quantization.html)

## Cost scenarios

| Scenario | Validation and setup | Conditional training | Scheduled demo inference | Storage allowance | Total |
| --- | ---: | ---: | ---: | ---: | ---: |
| Optimistic | 2 h L40S, $2.18 | $0 | 2 h L40S, $2.18 | $1 | $5.36 |
| Realistic | 4 h L40S, $4.36 | 8 h L40S, $8.72 | 6 h L40S, $6.54 | $2 | $21.62 |
| Pessimistic within scope | 6 h L40S, $6.54 | 18 h A100 PCIe, $37.62 | 8 h L40S, $8.72 | $3 | $55.88 |

All scenarios are estimates from the listed rates and planned billable hours, with no vendor API use or taxes included. Stop the first paid session at four billable hours or $10 total, whichever comes first. Include startup/download time in that cap. Further spend requires usable full-clip translations and a measured streaming bottleneck with a defined remediation. Schedule inference around meetings. At $1.09/h 20 Pod-hours/month costs $21.80 and an always-on 730-hour month costs about $796 before storage. Hosting time accrues while the Pod is running, including idle periods. Use the console's automatic stop when available and verify the state. [GPU pricing](https://www.runpod.io/pricing) · [Stop-after setting](https://docs.runpod.io/runpodctl/reference/runpodctl-remove-pods#flags-2)
