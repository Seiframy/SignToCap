# Next steps

Each task can proceed in its own branch with explicit inputs and a reproducible baseline.

| Task | Deliverable and review criterion |
| --- | --- |
| Higher FPS and transport | Compare 6, 12, and 24 FPS on identical complete references. Report actual FPS, queue and round-trip latency, dropped frames and caption meaning. Keep timestamp and validity checks. |
| Pose compatibility | Compare fresh RTMW with publisher poses on matched frames. Check joint order, coordinate normalization, masks, cadence and visibility. Record provider and model versions. |
| Data and adaptation | Expand approved official training data and separate sources/signers where possible. Freeze the evaluation set. Require an explicit budget and evaluate against the unchanged checkpoint before proposing deployment. |
| Accuracy and latency | Build a consented evaluation protocol with ASL-qualified review of meaning, names, numbers and negation. Separate pose, network, model and end-to-caption timing. Publish aggregates. |
| Application integration | Validate startup and reconnect behavior, capture and commit affordances, offline startup, and verified shutdown on Windows. Preserve GPU and credential safeguards. |

## Branch and pull request workflow

Fetch the remote, branch from current `origin/main`, and use a name such as `feat/higher-fps` or `docs/pilot-handoff`. Keep changes focused and stage explicit paths. Inspect tracked and staged files for secrets, recordings and restricted assets. Run the targeted checks, push the feature branch without force, and open a pull request into `main`. Include the baseline, evaluation sample size, measurements, and remaining uncertainty. Coordinate touching shared files before merging.
