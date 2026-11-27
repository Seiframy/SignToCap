# Controlled RunPod experiment

For the unattended, frozen-only experiment, use [Run independently](RUN-INDEPENDENTLY.md). This runbook covers the component commands and optional follow-up work.

These commands prepare and run a single-GPU validation after approval. The workspace package is self-contained except for publisher source and assets, the Pod environment, and consented inputs. No service needs to be provisioned to inspect the scripts locally.

## 1. Set up a single private Pod

- For the A100 SXM 80 GB, use the existing 50 GB network volume at `/workspace` and an SSH-enabled image with Python 3.11 or 3.12 and `venv`. Verify the console rate against the approved runner limit. Keep the isolated PyTorch 2.6.0/torchvision 0.21.0 CUDA 12.4 wheels on either host. The newer driver supports the older runtime. The bootstrapped process tests an actual BF16 CUDA matrix multiply before downloading weights. [GPU driver compatibility](https://docs.nvidia.com/deploy/cuda-compatibility/latest/frequently-asked-questions.html) · [PyTorch wheel matrix](https://docs.pytorch.org/get-started/previous-versions/)
- Stage the `phase2` folder at `/workspace/phase2` via the volume's S3 endpoint or direct SSH/SCP. Check disk availability before adding the optional pose archive, especially if the volume already contains other data. [Runpod SSH](https://docs.runpod.io/pods/configuration/use-ssh) · [Storage lifecycle](https://docs.runpod.io/pods/storage/types)
- The frozen runtime keeps the upstream model and tokenizer. Two small pinned patches skip the training loss during inference and defer DeepSpeed and decord imports to the unused upstream training and RGB data paths. ONNX Runtime runs pose extraction on the CPU. The frozen benchmark needs neither a CUDA extension build nor the base image's PyTorch. The Pod must still pass import, checkpoint load and token parity checks. [Upstream requirements](https://github.com/ZechengLi19/Uni-Sign/blob/eed438bcb49e30405cd6ccdfcccca330c134e830/requirements.txt)

```bash
export PHASE2=/workspace/phase2
export SIGNGPT_PYTHON="$(command -v python3.12 || command -v python3.11)"
bash "$PHASE2/bootstrap.sh" /workspace/signgpt-run
cd /workspace/signgpt-run
source "$(cat venv-path.txt)/bin/activate"
export PYTHONUNBUFFERED=1
export XDG_CACHE_HOME="$PWD/cache"
mkdir -p inputs runs/baseline
chmod 700 inputs runs/baseline
```

The script fetches one shallow pinned source revision, the 1.19 GB task checkpoint and only the six files needed by the pinned mT5 base. It checks the Uni-Sign checksum, tests imports, records environment resolution and requires CUDA with BF16 support. If a retry is needed, use the same directory. A wrong checked-out commit or a checksum mismatch requires inspection before proceeding. The Pod image and dependency combination is a proposed environment pending its first actual run.

## 2. Prepare the validation inputs

For a controlled webcam trial, provide authorized constant-frame-rate frontal recordings at 25 or 30 fps. If video is variable-frame-rate, normalize to the chosen FPS without changing elapsed time and verify against manual labels. Put only material authorized for Pod use under `inputs`. Extract each track once. The vendored RTMW lightweight assets download into the RTMLib cache on the first call. Record their names and hashes. This command times CPU pose work separately from generation.

```bash
# Example for one approved asset. Repeat for the pilot, holdout, and idle tracks.
python "$PHASE2/pose_data.py" video Uni-Sign inputs/turn-01.mp4 inputs/turn-01.npz \
  > runs/baseline/turn-01-pose.json
```

`inputs/pilot.jsonl` and `inputs/heldout.jsonl` each contain one JSON object per continuous track. `pose` is relative to the manifest. Annotate manual source time relative to the beginning of the recorded file, include leading/trailing idle and preserve actual gaps. These rows show the format with example text and times.

```jsonl
{"id":"turn-01","pose":"turn-01.npz","group":"signer-A","segments":[{"item":"u1","start_ms":650,"end_ms":3110,"reference":"Our next meeting is Tuesday."}]}
{"id":"idle-01","pose":"idle-01.npz","group":"signer-B","idle_only":true,"segments":[]}
```

Generate paired oracle clips directly from the held-out poses and manual ranges to isolate boundary error. Use one or more clips from the pilot for initial parity and tuning. Save a digest for the sealed acceptance manifest and its pose files before selecting settings.

```bash
python "$PHASE2/pose_data.py" split-oracle inputs/pilot.jsonl inputs/pilot-oracle
python "$PHASE2/pose_data.py" split-oracle inputs/heldout.jsonl inputs/heldout-oracle
```

### Optional release-domain check with publisher poses

The webcam pilot has first priority. When a format or domain check is needed, retrieve the following pinned **model-compatible** archive. Published How2Sign 2D keypoints from the dataset site follow their own format. The upstream How2Sign test labels come from the pinned checkout. Using this archive may require a separate session if download time threatens the first budget. [Publisher archive](https://huggingface.co/ZechengLi19/Uni-Sign/tree/a6d679fe0f4f1dcb09edb13b5cbdc006b8af1327) · [How2Sign annotations and terms](https://how2sign.github.io/#download)

```bash
mkdir -p datasets/how2sign
cd datasets/how2sign
source_base=https://huggingface.co/ZechengLi19/Uni-Sign/resolve/a6d679fe0f4f1dcb09edb13b5cbdc006b8af1327
for part in 00 01 02 03; do
  name="how2sign_pose_format.zip.$part"
  if [[ ! -f "$name" ]]; then
    curl -fL --retry 3 "$source_base/$name" -o "$name.part" || exit 1
    mv "$name.part" "$name"
  fi
done
sha256sum -c <<'CHECKSUMS'
cfbc90d0378b790fd8451f71c2bc15ef7763d5d52025853ff29c0fa2c80fbc80  how2sign_pose_format.zip.00
e1f8ad67748ac9141dd034b5bc3894601276ed921c70f9b35a9cfc379ccb39a8  how2sign_pose_format.zip.01
8e5d8043a58f100473fdcd340519458fd534bdcb5515ef33a023d1d286f7f8b7  how2sign_pose_format.zip.02
a091c9cd3d19ffcbd8a0009ceca905e32bf06ea1c2bee18b3af1a362ac2a6e24  how2sign_pose_format.zip.03
CHECKSUMS
cat how2sign_pose_format.zip.{00,01,02,03} > how2sign_pose_format.zip
unzip -tq how2sign_pose_format.zip
unzip -l how2sign_pose_format.zip | tail -8
df -h .
# After checking member paths and available disk
unzip -nq how2sign_pose_format.zip -d extracted
cd ../..
```

Verify the archive contains expected clip IDs and that its frame sampling cadence agrees with exact source temporal metadata before interpreting latency. For the release-domain check below, set `H2S_POSE_FPS` to that verified value. The converter deterministically selects 32 test rows, requires each pose match to be unambiguous, uses the available `start`/`end` span and emits model-compatible NPZ plus an oracle manifest. For an alignment-aware continuous run, join the official corrected timestamps to original source recordings and extract poses across real gaps. Sentence-only test poses can establish format and caption quality on isolated spans.

```bash
python "$PHASE2/pose_data.py" oracle Uni-Sign/data/How2Sign/labels.test \
  datasets/how2sign/extracted inputs/how2sign-test --limit 32 --fps "${H2S_POSE_FPS:?verify archive cadence first}"
python "$PHASE2/benchmark.py" oracle inputs/how2sign-test/oracle.jsonl runs/baseline/how2sign.jsonl \
  --beams 4 --tokens 100
python "$PHASE2/score.py" runs/baseline/how2sign.jsonl --bleu
```

The raw-video equivalent requires selection of only the relevant full frontal recordings from the train or validation resources. The published full train archive is 290 GB. Use a legally accessible subset or independent consented continuous examples to avoid pulling that archive for a small timing head. Annotate true source intervals and avoid treating empty camera views as pose-loss evidence for pauses. Keep the official test and acceptance sets sealed from tuning. [How2Sign downloads](https://how2sign.github.io/#download)

## 3. Benchmark the warmed translator and timed replay

In a second SSH shell, collect a sampled device-level trace during generation, then stop the logger when done. PyTorch peak values and this device trace have different scopes. Use one GPU session and one benchmark process at a time.

```bash
cd /workspace/signgpt-run
nvidia-smi --query-gpu=timestamp,name,memory.used,memory.total,utilization.gpu \
  --format=csv -lms 250 -f runs/baseline/device.csv
```

```bash
python "$PHASE2/benchmark.py" oracle inputs/pilot-oracle/oracle.jsonl runs/baseline/parity.jsonl \
  --parity --beams 4 --tokens 100
python "$PHASE2/benchmark.py" replay inputs/pilot.jsonl runs/baseline/pilot-pause.jsonl --policy pause --buffer-s 8
python "$PHASE2/benchmark.py" replay inputs/pilot.jsonl runs/baseline/pilot-punctuation.jsonl \
  --policy stable-punctuation --buffer-s 8
python "$PHASE2/benchmark.py" replay inputs/pilot.jsonl runs/baseline/pilot-short.jsonl \
  --policy stable-punctuation --buffer-s 5
# Switch to the holdout only after the policy, buffer, decode and activity settings are chosen.
python "$PHASE2/benchmark.py" oracle inputs/heldout-oracle/oracle.jsonl runs/baseline/heldout-oracle.jsonl \
  --beams 4 --tokens 100
# Example for a pilot winner. Pass chosen settings explicitly.
python "$PHASE2/benchmark.py" replay inputs/heldout.jsonl runs/baseline/heldout.jsonl \
  --policy stable-punctuation --buffer-s 8 --cadence-ms 500 --motion 0.7 --beams 1 --tokens 48
python "$PHASE2/score.py" runs/baseline/heldout.jsonl --manifest inputs/heldout.jsonl
```

On the pilot, also compare greedy 48 tokens with four beams/100 tokens at the same window and cadence to separate computational changes from segmentation differences. Verify shape, caption text and time distributions after each run. If needed, run oracle with the chosen smaller decode settings before comparing oracle and replay semantics. Equal token IDs in parity mode test the source-only patch, with warm latency and memory measured in a separate run that omits `--parity`. GPU allocations in the benchmark include model load and warm-up. Use the device trace to document total occupancy.

A reviewer maps each meaning unit to the earliest useful event and its corresponding final. Copy exact `emit_ms` numbers from the event log and use `null` when absent. Record the final caption's meaning and any changes to agency, negation, amounts or names. Examples below are format illustrations.

```jsonl
{"id":"turn-01","item":"u1","start_ms":650,"end_ms":3110,"first_useful_emit_ms":2134.52,"final_emit_ms":3821.21,"meaning":"preserved"}
{"id":"turn-01","item":"u2","start_ms":4250,"end_ms":6020,"first_useful_emit_ms":null,"final_emit_ms":null,"meaning":"missing"}
```

```bash
python "$PHASE2/score.py" runs/baseline/heldout.jsonl --manifest inputs/heldout.jsonl \
  --review runs/baseline/review.jsonl > runs/baseline/score.json
```

Review all eligible units, count unassigned or duplicate finals and record premature cuts and incorrect semantic content explicitly. The tool reports percentiles for observed events alongside missing counts, and final/partial counts on idle tracks. For networked testing, record actual client render timestamps alongside the manual source-relative times. Replay logs alone cannot establish the live gate.

## 4. Run the conditional boundary experiment

Proceed after the reviewed error is confirmed to be boundary timing and follow-up compute is approved. Prepare `inputs/boundary.jsonl` in the same continuous format with a `split` of `train` or `dev`, a group per signer or recording and reliable start/end labels. An idle negative example has `segments:[]`. Include genuine within-sentence pauses as negatives. Use separate manifest files for acceptance and training. The trainer rejects shared groups and saves both model/optimizer state and a digest of the manifest and pose bytes. Use the CPU by default. Use the single Pod GPU only for a bounded smoke if that validation is required.

```bash
python "$PHASE2/train_boundary.py" inputs/boundary.jsonl runs/boundary-smoke --device cuda --steps 3 --eval-every 1
python "$PHASE2/train_boundary.py" inputs/boundary.jsonl runs/boundary-smoke --device cuda --steps 4 --eval-every 1 --resume
python "$PHASE2/train_boundary.py" inputs/boundary.jsonl runs/boundary --device cpu --steps 300 --eval-every 50 \
  > runs/boundary-train.jsonl
python "$PHASE2/benchmark.py" replay inputs/heldout.jsonl runs/baseline/heldout-boundary.jsonl \
  --buffer-s 8 --cadence-ms 500 --motion 0.7 --beams 1 --tokens 48 \
  --boundary-head runs/boundary/best.pt
```

Re-review the pair with matched decode settings. Use the pilot or a diagnostic set to decide on the adaptation and choose settings on dev. If the held-out baseline has already been reviewed to choose the adaptation, treat that comparison as exploratory and confirm on a new sealed set. Adopt the head only if it corrects specific missed or late boundaries without additional premature cuts or deterioration in meaning and idle behavior. Retain the original checkpoint as the rollback. A failed boundary model gives no evidence that decoder LoRA will improve translation.

## 5. Preserve the evidence and shut down

Store or export the source revision, image digest, `requirements-resolved.txt`, model hashes, model and policy settings, split fingerprints, `gpu.txt`, device trace, timings and explicitly authorized reviewer summaries. Save `last.pt` for resume and `best.pt` for inference with checksums. Inspect progress and model-load tests before spending on a restart. SCP allowed artifacts off the direct SSH endpoint before terminating. Exclude signer recordings, captions, pose data and credentials from shared archives unless their use is explicitly authorized. Stopped Pod volume storage still incurs charges and Pod termination deletes Pod-bound volumes. Terminate the Pod and verify billing state after exporting artifacts, unless a separately approved persistence plan applies. [Storage lifecycle](https://docs.runpod.io/pods/storage/types) · [Pod pricing](https://docs.runpod.io/pods/pricing)
