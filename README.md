# SignToCap

SignToCap is a Windows research prototype for turning American Sign Language into English captions. It supports offline replay, local video upload, complete prerecorded sentences, and a guarded webcam **Sign & Translate** workflow. Translation quality varies. Continuous real-time translation remains experimental.

```powershell
python -B .\phase3\desktop.py --open
```

A clean checkout starts with an illustrative caption replay and needs no model files, credentials, or GPU. See [Quickstart](QUICKSTART.md) for replay, webcam inference, and the training pilot.

- [Project context](PROJECT_CONTEXT.md) covers the architecture, dependencies, data, and asset access.
- [Current status](CURRENT_STATUS.md) records verified behavior and limitations.
- [Benchmarks](BENCHMARKS.md) records sanitized results and the adapter decision.
- [Next steps](NEXT_STEPS.md) defines independent engineering tasks and the Git workflow.
- [Phase 2](phase2/README.md) contains the model and evaluation tools. [Phase 3](phase3/README.md) contains the Windows app, GPU manager, and pilot.

The repository also preserves the team's [initial code notes](code/README.md), [research references](pappers/README.md), [presentation drafts](presentation/README.md), and [technology decision report](technologies/README.md).
