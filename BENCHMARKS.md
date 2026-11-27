# Benchmarks

| Run | Verified result |
| --- | --- |
| Original webcam test | About 908 ms median pose extraction and 1,117 ms median round trip. At a 3 FPS target, about 0.62 FPS processed, 37 frames processed, and zero captions. |
| Pretrained prerecorded inference | Uni-Sign generated a caption from a complete known ASL video. The numeric content differed from the reference. |
| RTX 4090 CUDA sentence test | 23.2 ms median pose extraction, 48/48 scheduled frames and zero drops, 284 ms final decode, 17.2 seconds total. The question caption was semantically incorrect. |
| Local 24 FPS versus 6 FPS | 24 FPS preserved more reference meaning on two of three matched fresh-pose clips. Both rates missed the third. |
| Overlapped sentence processing | Capture, batched transport, and pose processing overlap. Local checks passed. Remote latency remains unmeasured. |

The 908 ms and 23.2 ms pose medians are from different workloads and deployments. Treat the comparison as an observed change across runs.

## H100 training pilot

- Froze the pretrained stack and updated only the pose projection for eight optimizer steps on 16 official training clips. Four clips from a separate source were used for validation. Six separate evaluation clips supplied the matched fresh and publisher pose arms.
- Fresh poses at 6 FPS scored **21.03 → 20.03 chrF**. Publisher poses at native sampling scored **35.44 → 36.56 chrF**.
- Rejected the adapter because the fresh-pose score fell. The original checkpoint remains the deployment default.

The six clips are a readiness check. Generalization to unconstrained webcam signing and continuous real-time translation remains unverified.
