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

## Phase 4 local comparison

One authorized annotated sentence was exercised with real RTMW extraction and the original Uni-Sign checkpoint on CPU. All scheduled samples were retained. This single run includes extraction setup and decoding, with model work timed separately.

| Condition | Retained / scheduled | Median pose ms | Extraction s | Model s | Combined s |
| --- | ---: | ---: | ---: | ---: | ---: |
| Previous detection, native | 81/81 | 248.50 | 24.41 | 3.17 | 27.58 |
| Checked detection, native | 81/81 | 232.80 | 21.74 | 2.58 | 24.32 |
| Checked detection, 6 FPS | 21/21 | 242.76 | 7.80 | 1.76 | 9.56 |
| Checked detection, 12 FPS | 41/41 | 230.16 | 11.44 | 2.24 | 13.68 |
| Checked detection, 24 FPS | 81/81 | 231.92 | 22.06 | 3.23 | 25.29 |

The native and 24 FPS captions were identical and preserved more of this sentence, with the same incorrect number. The 6 and 12 FPS captions also changed the entity. This run leaves the accuracy impact of the detection fix and same-rate latency impact unverified. Transport and UI time are excluded. The CPU environment lacked pinned SacreBLEU. The benchmark supports chrF and BLEU when the dependency is available, and the paired JSON supports human meaning review.

### Preprocessing comparison

The pinned demo uses RTMW in COCO-WholeBody mode, divides x by width and y by height, and delegates body, hands, face slicing, confidence masking, and normalization to Uni-Sign's loader. SignToCap uses that loader for batch-one tensors of shape [1,T,joints,3] per part and an all-valid [1,T] mask for real, unpadded frames. Hand indices 91–111 and 112–132 retain publisher order. The facial subset, confidence threshold and window scaling remain upstream. The demo retains all source frames and takes the first detection in each frame. SignToCap tracks one signer and can explicitly sample within its 256-frame window.

On the earlier matched 81-frame publisher/fresh diagnostic, normalized median landmark differences were 0.0034 for body, 0.0023 for face, 0.0074 for the left hand, and 0.0082 for the right. These measurements do not establish a hand swap, normalization error, or the cause of the quality gap. Phase 4 ignores non-finite or entirely zero-confidence detections before choosing a signer or updating tracking. A regression test covers the boundary case. Model weights are unchanged.
