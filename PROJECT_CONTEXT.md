# Project context

The Python desktop broker serves a plain HTML/CSS/JavaScript UI on loopback. The browser records frame timestamps, queues a bounded number of frames, and batches transport. The guarded worker receives ordered requests over authenticated SSH, extracts COCO-WholeBody poses with RTMW via ONNX Runtime, and runs the pinned Uni-Sign How2Sign pose-only model with mT5 and PyTorch. Caption validity checks reject incomplete windows and low-visibility detections. Sign & Translate commits a complete, signer-delimited window. Local browser storage holds optional recordings and evidence.

The Windows broker uses the RunPod API for Pod discovery, attachment checks and verified shutdown. A separate Windows watchdog and a detached Pod watchdog bind to the exact generation and the actual startup clock. The hard Pod limit is 15 minutes. The worker checks the GPU, mounts, checksums and provider assignment. Paid work begins after manual startup and explicit rate approval.

## Code and dependencies

| Location | Responsibility |
| --- | --- |
| [phase2](phase2/README.md) | Pinned Uni-Sign setup, selective archives, pose conversion, oracle/replay, scoring and training experiments |
| [phase3](phase3/README.md) | Desktop UI, GPU lifecycle, app/worker protocol, profiling and paired adaptation pilot |
| [phase2-analysis](phase2-analysis/README.md) | Optional report-generation utilities |

Replay uses standard-library Python and a modern browser. The Pod runtime uses the pinned [Phase 2 requirements](phase2/requirements.txt), PyTorch, NumPy, OpenCV, and the [hash-pinned CUDA ONNX Runtime wheel](phase3/requirements-cuda-benchmark.txt). Windows live attachment also uses OpenSSH. Use the prepared Linux Python 3.12 environment on the volume for CUDA pose verification and the pilot.

## Data and asset access

- Use the official [How2Sign download page](https://how2sign.github.io/#download) for train annotations and the [Uni-Sign publisher archive](https://huggingface.co/ZechengLi19/Uni-Sign) for pose archives and the pretrained checkpoint. Obtain the mT5 base weights using the pinned model preparation script. Read upstream terms before downloading or sharing assets.
- Ask the maintainer for the private evaluation bundle, original replay events, authorized warm-up frame, and prepared runtime. The checkout excludes recordings, extracted landmarks, training/evaluation texts, model weights, and rejected adapters. Restore optional resources at the paths documented in [Quickstart](QUICKSTART.md) rather than committing them.
- The pilot joins frontal clip identities and reference windows against official training annotations. Split by source recording and exclude prior evaluation sources. Source separation alone does not establish signer separation.
- Use individual credentials and approved consent/usage terms. Keep secrets in Windows Credential Manager or the appropriate private AWS/SSH store. Share only sanitized aggregate metrics through Git.
