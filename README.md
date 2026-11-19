# Sign GPT

Sign GPT is a four-person student project exploring real-time translation from sign language into written captions for professional meetings. The goal is to help students who sign communicate directly with classmates during discussions, group projects, and presentations.

## Project status

The repository contains early research notes, technology references, and a PowerPoint draft. Implementation starts in [code](code/README.md). The team still needs to select the target sign language, caption language, dataset, and meeting integration.

## Proposed system

1. Capture an opt-in video stream of the signer.
2. Process short windows of video while preserving hand, face, and body cues.
3. Translate the signed utterance into readable text and display it in the meeting caption interface.
4. Measure translation quality and end-to-end caption delay on held-out signers. Include tests for camera framing, occlusion, and lighting, and decide how the interface should indicate uncertain output.

Use consented recordings and document the usage terms for any dataset before training or sharing. Keep participant recordings and credentials out of Git.

## Repository guide

| Folder | Contents |
| --- | --- |
| [code](code/README.md) | Planned prototype and evaluation code |
| [pappers](pappers/README.md) | Research references for the related work section |
| [presentation](presentation/README.md) | PowerPoint drafts and review notes |
| [technologies](technologies/README.md) | Candidate tools and dataset links |

## Contributing

Clone the repository after accepting your GitHub invitation. Create a branch for your change, link it to the relevant issue, and open a pull request for teammate review before merging into `main`.
