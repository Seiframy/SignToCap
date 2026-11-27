"""Prepare matched evaluation sets and optional source-disjoint How2Sign training manifests."""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import re
from tempfile import TemporaryDirectory

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
ARCHIVE = ROOT / "signgpt-results/20261127T062446887049Z"
SETTINGS = HERE / "accuracy-settings.json"


def rows(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def source(clip_id):
    match = re.match(r"^([-A-Za-z0-9_]{11})(?:_\d+)*-\d+-rgb_(?:front|side)(?:\.mp4|/|$)", clip_id)
    if not match:
        raise ValueError(f"Unrecognized How2Sign recording ID {clip_id}")
    return match[1]


def write_rows(path, samples, parent):
    normalized = []
    for row in samples:
        pose = (parent / row["pose"]).resolve(strict=True)
        if not pose.is_file() or pose.suffix.lower() != ".npz":
            raise ValueError(f"Expected a pose track for {row['id']}")
        normalized.append(dict(row, pose=str(pose), source_id=source(row["id"])))
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in normalized), encoding="utf-8")


def official_training(tsv):
    if tsv is None or tsv.name != "how2sign_realigned_train.csv":
        raise ValueError("Provide the official how2sign_realigned_train.csv")
    with tsv.open(encoding="utf-8-sig", newline="") as stream:
        table = csv.DictReader(stream, delimiter="\t")
        required = {"SENTENCE_NAME", "SENTENCE", "START_REALIGNED", "END_REALIGNED"}
        if not required <= set(table.fieldnames or []):
            raise ValueError("Expected the official tab-separated training annotations")
        records = {}
        for row in table:
            clip = row["SENTENCE_NAME"].removesuffix(".mp4")
            if clip in records:
                raise ValueError("Duplicate sentence ID in training annotations")
            try:
                duration = float(row["END_REALIGNED"]) - float(row["START_REALIGNED"])
            except (TypeError, ValueError):
                continue
            if math.isfinite(duration) and duration > 0:
                records[clip] = (" ".join(row["SENTENCE"].split()), duration)
    if not records:
        raise ValueError("Empty training annotations")
    return records


def protected_sources(full):
    excluded = {source(r["id"]) for data in full.values() for r in data}
    excluded.update(source(r["id"]) for name in ("how2sign-oracle-events.jsonl", "continuous-oracle-events.jsonl")
                    for r in rows(ARCHIVE / name) if r.get("kind") == "oracle")
    return excluded


def prepare(output, train_manifest=None, train_tsv=None):
    settings = json.loads(SETTINGS.read_text(encoding="utf-8"))
    audit = HERE / "audit-data"
    full = {branch: rows(audit / (branch + ".jsonl")) for branch in ("fresh", "publisher")}
    matched = {branch: [(r["id"], r["reference"]) for r in data if r["split"] == "heldout"]
               for branch, data in full.items()}
    if not matched["fresh"] or dict(matched["fresh"]) != dict(matched["publisher"]):
        raise ValueError("The held-out pose arms do not have matching references")
    excluded = protected_sources(full)
    exports = {f"heldout-{branch}.jsonl": ([r for r in data if r["split"] == "heldout"], audit)
               for branch, data in full.items()}
    plan = {"heldout": len(matched["fresh"]), "excluded_source_ids": sorted(excluded),
            "training_ready": False, "settings": settings}
    if train_manifest:
        official = official_training(train_tsv)
        eligible = rows(train_manifest)
        labels = [official.get(r["id"].removesuffix(".mp4")) for r in eligible]
        if (not eligible or len({r["id"] for r in eligible}) != len(eligible)
                or any(r.get("dataset_split") != "train" or not r.get("reference")
                       or source(r["id"]) in excluded or label is None
                       or label[0] != " ".join(r["reference"].split())
                       for r, label in zip(eligible, labels))):
            raise ValueError("Training requires distinct clips with matching official references outside protected recordings")
        by_source = {}
        timed = [dict(row, official_duration_s=label[1]) for row, label in zip(eligible, labels)]
        for row in sorted(timed, key=lambda r: r["id"]):
            by_source.setdefault(source(row["id"]), []).append(row)
        groups = sorted(by_source, key=lambda value: hashlib.sha256(f"{settings['seed']}:{value}".encode()).digest())
        if len(groups) < 2:
            raise ValueError("Training and validation require different source recordings")
        split = max(1, len(groups) // 8)
        for name, chosen, limit in (("validation", groups[:split], settings["validation_limit"]),
                                    ("train", groups[split:], settings["train_limit"])):
            samples = [r for group in chosen for r in by_source[group][:settings["per_source_limit"]]][:limit]
            exports[name + ".jsonl"] = (samples, train_manifest.parent)
            plan[name] = len(samples)
        if plan["train"] < settings["train_min"] or plan["validation"] < settings["validation_min"]:
            raise ValueError("Need more source-disjoint verified training and validation clips")
        plan["training_ready"] = True
        plan["official_train_sha256"] = hashlib.sha256(train_tsv.read_bytes()).hexdigest()
    if output.exists():
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix=".SignToCap-plan-", dir=output.parent) as temporary:
        stage = Path(temporary) / "plan"
        stage.mkdir()
        for name, (samples, parent) in exports.items():
            write_rows(stage / name, samples, parent)
        (stage / "plan.json").write_text(json.dumps(plan, indent=2) + "\n", encoding="utf-8")
        stage.rename(output)
    print(json.dumps({k:v for k,v in plan.items() if k not in {"settings","excluded_source_ids"}}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path, help="New directory for the immutable plan")
    parser.add_argument("--official-train", type=Path, help="JSONL with id, pose, reference and dataset_split=train")
    parser.add_argument("--official-train-tsv", type=Path, help="Official how2sign_realigned_train.csv")
    args = parser.parse_args()
    prepare(args.output, args.official_train, args.official_train_tsv)
