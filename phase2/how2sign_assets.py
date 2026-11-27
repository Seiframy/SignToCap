"""Download the publisher's checksummed How2Sign pose archive. This needs no GPU."""

import argparse
import hashlib
import json
import math
from pathlib import Path
import pickle
import shutil
import subprocess
import sys
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from zipfile import ZipFile

from range_zip import RangedFile

BASE = "https://huggingface.co/ZechengLi19/Uni-Sign/resolve/a6d679fe0f4f1dcb09edb13b5cbdc006b8af1327"
HASHES = (
    "cfbc90d0378b790fd8451f71c2bc15ef7763d5d52025853ff29c0fa2c80fbc80",
    "e1f8ad67748ac9141dd034b5bc3894601276ed921c70f9b35a9cfc379ccb39a8",
    "8e5d8043a58f100473fdcd340519458fd534bdcb5515ef33a023d1d286f7f8b7",
    "a091c9cd3d19ffcbd8a0009ceca905e32bf06ea1c2bee18b3af1a362ac2a6e24",
)
SIZES = (2147483648, 2147483648, 2147483648, 2023023631)
LABEL_SHA256 = "3e4c70aa4dd3fcabf1380accad2e4787d1da5d9afe1f86e7518acee01895fad1"


def prepare_sample(labels, output):
    from pose_data import convert_oracle_members

    if not labels.is_file():
        raise ValueError("Use labels.test from the pinned Uni-Sign checkout")
    with labels.open("rb") as stream:
        labels_digest = hashlib.file_digest(stream, "sha256").hexdigest()
    if labels_digest != LABEL_SHA256:
        raise ValueError("Unexpected Uni-Sign test label checksum")
    output.mkdir(parents=True, exist_ok=True)
    if shutil.disk_usage(output).free < 1024**3:
        raise OSError("Need 1 GiB free for the sampled poses")
    parts = [(f"{BASE}/how2sign_pose_format.zip.{i:02}", size) for i, size in enumerate(SIZES)]
    with ZipFile(RangedFile(parts, 1024**2)) as zip_file:
        # Read selected members to EOF so ZIP CRC validation runs before any pickle is loaded.
        from io import BytesIO
        convert_oracle_members(SimpleNamespace(labels=labels, output=output, limit=32, fps=None),
                               zip_file.namelist(), lambda name: BytesIO(zip_file.read(name)))
    provenance = {"split": "test", "sample_seed": 42, "clip_count": 32,
                  "release": BASE, "verification": "pinned URL, exact byte ranges, ZIP CRC per selected member"}
    provenance["labels_sha256"] = labels_digest
    (output / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n", encoding="utf-8")


def prepare_training(annotations, output):
    import numpy as np
    from pose_data import save_track

    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "phase3"))
    from accuracy_plan import HERE, SETTINGS, official_training, protected_sources, rows, source

    labels = official_training(annotations)
    settings = json.loads(SETTINGS.read_text(encoding="utf-8"))
    excluded = protected_sources({arm: rows(HERE / "audit-data" / f"{arm}.jsonl")
                                  for arm in ("fresh", "publisher")})
    output.mkdir(parents=True, exist_ok=True)
    manifest = output / "train.jsonl"
    tracks = output / "poses"
    if manifest.exists() or tracks.exists():
        raise FileExistsError("The training subset already exists. Verify it or use a new output directory")
    count = settings["validation_min"]
    groups_needed = 1 + math.ceil(settings["train_min"] / count)
    if count > settings["per_source_limit"] or groups_needed >= 16:
        raise ValueError("The selective subset requires the short pilot size settings")
    parts = [(f"{BASE}/how2sign_pose_format.zip.{i:02}", size) for i, size in enumerate(SIZES)]
    reader = RangedFile(parts, 1024**2, transfer_limit=96 * 1024**2)
    with ZipFile(reader) as archive, TemporaryDirectory(prefix=".pose-subset-", dir=output) as temp:
        stage = Path(temp)
        (stage / "poses").mkdir()
        candidates = {}
        for member in archive.infolist():
            path = Path(member.filename)
            clip = path.stem
            if path.parent.as_posix() == "test-pose-rtmpose-192" and clip.endswith("-rgb_front"):
                excluded.add(source(clip))
            if (path.parent.as_posix() != "train-pose-rtmpose-192" or path.suffix != ".pkl"
                    or not clip.endswith("-rgb_front") or clip not in labels
                    or member.file_size > 8 * 1024**2):
                continue
            reference, duration = labels[clip]
            if reference and 2 <= duration <= 10:
                candidates.setdefault(source(clip), []).append(member)
        chosen = []
        fingerprints = set()
        fps = 24
        for recording, members in candidates.items():
            if recording in excluded or len(members) < count:
                continue
            group = []
            for member in members:
                # ZIP validates the member CRC before we decode the pinned publisher's data.
                data = pickle.loads(archive.read(member))
                points, scores = data["keypoints"], data["scores"]
                first, end = int(data.get("start", 0)), int(data.get("end", len(scores)))
                clip = Path(member.filename).stem
                reference, duration = labels[clip]
                if (not 0 <= first < end <= min(len(points), len(scores)) or end - first > 256
                        or abs((end - first) / fps - duration) > max(.2, 2 / fps)):
                    continue
                points = np.asarray(points[first:end], dtype=np.float32)
                scores = np.asarray(scores[first:end], dtype=np.float32)
                fingerprint = hashlib.sha256(points.tobytes() + scores.tobytes()).digest()
                if fingerprint in fingerprints:
                    continue
                relative = f"poses/{clip}.npz"
                try:
                    save_track(stage / relative, points, scores, fps)
                except ValueError:
                    continue
                fingerprints.add(fingerprint)
                group.append(dict(id=clip, pose=relative, reference=reference, dataset_split="train",
                                  publisher_member=member.filename, publisher_crc32=f"{member.CRC:08x}"))
                if len(group) == count:
                    break
            if len(group) == count:
                chosen.extend(group)
                print(f"Aligned {count} training clips from {recording}", flush=True)
                if len(chosen) == count * groups_needed:
                    break
            else:
                for row in group:
                    (stage / row["pose"]).unlink()
        if len(chosen) != count * groups_needed:
            raise ValueError("Too few publisher pose clips match the official training windows")
        (stage / "train.jsonl").write_text(
            "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in chosen), encoding="utf-8")
        (stage / "poses").rename(tracks)
        (stage / "train.jsonl").rename(manifest)
    print(f"Prepared {len(chosen)} clips from {groups_needed} sources at 24 FPS. "
          f"Selective transfer: {reader.requested_bytes / 1024**2:.1f} MiB.")


def prepare(directory):
    directory.mkdir(parents=True, exist_ok=True)
    archive = directory / "how2sign_pose_format.zip"
    shards = []
    for index, expected in enumerate(HASHES):
        shard = directory / f"{archive.name}.{index:02}"
        source = shard
        if not shard.exists():
            source = shard.with_suffix(shard.suffix + ".part")
            curl = shutil.which("curl.exe") or shutil.which("curl")
            if not curl:
                raise RuntimeError("curl is required to download the archive")
            subprocess.run([curl, "-fL", "--retry", "3", "-o", str(source), f"{BASE}/{shard.name}"], check=True)
        with source.open("rb") as stream:
            if hashlib.file_digest(stream, "sha256").hexdigest() != expected:
                raise ValueError(f"Publisher checksum mismatch for {source}")
        if source != shard:
            source.replace(shard)
        shards.append(shard)
    total_bytes = sum(shard.stat().st_size for shard in shards)
    if not archive.exists() or archive.stat().st_size != total_bytes:
        temp = archive.with_suffix(".zip.part")
        if shutil.disk_usage(directory).free < total_bytes + 2**30:
            raise OSError("Insufficient space to assemble the How2Sign pose archive")
        with temp.open("wb") as out:
            for shard in shards:
                with shard.open("rb") as stream:
                    shutil.copyfileobj(stream, out, length=8 * 1024**2)
        with ZipFile(temp) as check:
            if not any(name.endswith(".pkl") for name in check.namelist()):
                raise ValueError("No pose pickle members found")
        temp.replace(archive)
    return archive


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--sample-labels", type=Path, help="Stage the 32 official test clips using byte ranges")
    mode.add_argument("--training-annotations", type=Path, help="Build the small official training subset using byte ranges")
    args = parser.parse_args()
    if args.training_annotations:
        prepare_training(args.training_annotations, args.directory)
    elif args.sample_labels:
        prepare_sample(args.sample_labels, args.directory)
    else:
        print(prepare(args.directory))
