# Current status

| Status | Result |
| --- | --- |
| Verified on GPU | Pretrained Uni-Sign produced captions on complete prerecorded ASL. The RTX 4090 used CUDA for both pose stages. Sign & Translate processed all 48 scheduled frames in its recorded test. |
| Verified on H100 | Eight-step pose-projection adaptation and paired held-out scoring completed. The fresh-pose score fell, so the adapter was rejected and the pretrained checkpoint retained. |
| Verified locally | Sign & Translate overlaps capture, transport, and pose extraction. Ordered delivery, gap rejection, and drain-before-commit were checked locally. |
| Phase 4 offline | Upload metadata, local sampling, guarded pose handoff, and 6/12/24 FPS negotiation and scheduling are implemented. Real CPU inference ran on one authorized clip. The full upload UI against a live worker and sustained higher FPS webcam capture remain pending. |
| Unverified | Remote latency of the overlapping implementation, sustained higher-FPS capture, reliable window boundaries in continuous use, and broad webcam translation accuracy. |

The recorded webcam sentence produced an incorrect paraphrase of the intended question despite healthy transport. Use [Benchmarks](BENCHMARKS.md) for the numbers and [Next steps](NEXT_STEPS.md) for the engineering backlog.
