"""Extract the exact publisher RGB samples and compare them with the paired publisher poses."""

import argparse
from hashlib import sha256
import gzip
import json
from pathlib import Path
import pickle
import re
import sys
from zipfile import ZipFile

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "phase2"))
from how2sign_assets import BASE, LABEL_SHA256, SIZES
from pose_data import extract_video, load_track, save_track
from range_zip import RangedFile
from types import SimpleNamespace


RGB = "https://huggingface.co/ZechengLi19/Uni-Sign/resolve/eab251b7fe7e8521afc0e67be98add670ea40a0d"
RGB_PARTS = (2**32, 2**32, 2**32, 2**32, 3010900187)
DIAGNOSTIC = ("fzueHkUDyhE_1-8-rgb_front", "fzueHkUDyhE_3-8-rgb_front",
              "G3OMdQo1zjY_0-10-rgb_front", "G3OMdQo1zjY_2-10-rgb_front")


def members(archive, extension):
    result = {}
    for entry in archive.infolist():
        path = Path(entry.filename)
        if path.suffix == extension and path.stem not in result:
            result[path.stem] = entry
    return result


def group(name):
    return re.split(r"_\d+-\d+-rgb_front$", name)[0]


def summarize(pub, fresh):
    key1, conf1, _ = pub
    key2, conf2, fps = fresh
    match = len(key1) == len(key2)
    output = {"publisher_frames": len(key1), "fresh_frames": len(key2), "fresh_verified_fps": fps,
              "frame_counts_match": match, "identity_basis": "publisher release RGB filename and matching pose filename"}
    for label, points, scores in (("publisher", key1, conf1), ("fresh", key2, conf2)):
        output[label] = {"confidence_min": float(scores.min()), "confidence_max": float(scores.max()),
                         "low_confidence_fraction": float((scores <= .3).mean()),
                         "missing_frame_fraction": float((scores.max(axis=(1, 2)) == 0).mean()),
                         "active_xy_p01_p99": np.percentile(points[scores > .3], (1, 99), axis=0).tolist()
                         if (scores > .3).any() else None}
    if match:
        output["joint_error"] = {}
        for name, start, end in (("body", 0, 17), ("face", 23, 91), ("left_hand", 91, 112), ("right_hand", 112, 133)):
            mask = (conf1[..., start:end] > .3) & (conf2[..., start:end] > .3)
            error = np.linalg.norm(key1[..., start:end, :] - key2[..., start:end, :], axis=-1)[mask]
            output["joint_error"][name] = {"joint_pairs": int(mask.sum()),
                                           "median_normalized_distance": float(np.median(error)) if len(error) else None}
    return output


def prepare(args):
    raw_labels = args.labels.read_bytes()
    if sha256(raw_labels).hexdigest() != LABEL_SHA256:
        raise ValueError("Expected the pinned official test labels")
    labels = {Path(key).stem: row for key, row in pickle.loads(gzip.decompress(raw_labels)).items()}
    prior = [json.loads(line) for line in args.previous.read_text(encoding="utf-8").splitlines() if line]
    excluded = {group(Path(row["id"]).stem) for row in prior} | {group(name) for name in DIAGNOSTIC}
    args.output.mkdir(parents=True, exist_ok=True)
    cache = args.cache.resolve()
    cache.mkdir(parents=True, exist_ok=True)
    rgb_parts = [(f"{RGB}/how2sign_rgb_format.zip.{suffix}", size) for suffix, size in zip(
        ("aa", "ab", "ac", "ad", "ae"), RGB_PARTS)]
    pose_parts = [(f"{BASE}/how2sign_pose_format.zip.{i:02}", size) for i, size in enumerate(SIZES)]
    with ZipFile(RangedFile(rgb_parts, 1024**2)) as rgb, ZipFile(RangedFile(pose_parts, 1024**2)) as poses:
        videos = members(rgb, ".mp4")
        keypoints = members(poses, ".pkl")
        available = set(videos) & set(keypoints) & set(labels)
        chosen = [(name, "diagnostic") for name in DIAGNOSTIC if name in available]
        for name in sorted(available, key=lambda item: sha256(f"phase3-v1:{item}".encode()).digest()):
            source = group(name)
            if source in excluded or not 100_000 <= videos[name].file_size <= 3_000_000:
                continue
            chosen.append((name, "heldout"))
            excluded.add(source)
            if sum(split == "heldout" for _, split in chosen) >= args.count:
                break
        print("Selected:", chosen, flush=True)
        if len(chosen) != len(DIAGNOSTIC) + args.count:
            raise ValueError("Missing a predefined diagnostic or held-out RGB/pose pair")
        audit, manifests = [], {"publisher": [], "fresh": []}
        for name, split in chosen:
            raw = rgb.read(videos[name])
            video = cache / f"{name}.mp4"
            video.write_bytes(raw)
            data = pickle.loads(poses.read(keypoints[name]))
            fresh_path = args.output / f"{name}-fresh.npz"
            extract_video(SimpleNamespace(upstream=args.upstream, video=video, output=fresh_path,
                                          start_ms=0, end_ms=None, verify_cadence=True))
            start, end = int(data.get("start", 0)), int(data.get("end", len(data["scores"])))
            pub_path = args.output / f"{name}-publisher.npz"
            save_track(pub_path, data["keypoints"][start:end], data["scores"][start:end], None)
            report = summarize(load_track(pub_path), load_track(fresh_path))
            report.update(id=name, split=split, rgb_member=videos[name].filename, rgb_sha256=sha256(raw).hexdigest(),
                          publisher_pose_member=keypoints[name].filename, publisher_offset=start,
                          publisher_width_height=data.get("w_h"))
            audit.append(report)
            for arm, path in (("publisher", pub_path), ("fresh", fresh_path)):
                manifests[arm].append({"id": name, "split": split, "pose": path.name, "reference": labels[name]["text"]})
            print(name, report["publisher_frames"], report["fresh_frames"], flush=True)
    for arm, rows in manifests.items():
        (args.output / f"{arm}.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    (args.output / "pose-audit.json").write_text(json.dumps(audit, indent=2, default=lambda x: np.asarray(x).tolist()),
                                               encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("labels", "previous", "upstream", "cache", "output"):
        parser.add_argument(name, type=Path)
    parser.add_argument("--count", type=int, default=6, choices=range(1, 13))
    prepare(parser.parse_args())
