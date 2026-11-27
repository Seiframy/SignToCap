"""Validate an accuracy plan or explicitly run a bounded pose-projection trial."""
import argparse
import hashlib
import json
from pathlib import Path
import random
import sys

from accuracy_plan import ROOT, SETTINGS, rows, source

sys.path.insert(0, str(ROOT / "phase2"))
from pose_data import load_track
from unisign import Translator


def validate(plan_dir):
    plan = json.loads((plan_dir / "plan.json").read_text(encoding="utf-8"))
    settings = json.loads(SETTINGS.read_text(encoding="utf-8"))
    if not plan.get("training_ready") or not plan.get("official_train_sha256") or plan.get("settings") != settings:
        raise ValueError("Prepare a source-disjoint plan with an official-train manifest first")
    train, validation = (rows(plan_dir / (name + ".jsonl")) for name in ("train", "validation"))
    protected = set(plan["excluded_source_ids"])
    groups = []
    poses = set()
    for data, split in ((train, "train"), (validation, "validation")):
        if not settings[split + "_min"] <= len(data) <= settings[split + "_limit"] or len({r["id"] for r in data}) != len(data):
            raise ValueError("Invalid split size or duplicate clip")
        group = set()
        for row in data:
            recording = source(row["id"])
            if (row.get("dataset_split") != "train" or recording in protected or
                    row.get("source_id") != recording or not row.get("reference", "").strip()):
                raise ValueError("Invalid training source or reference")
            points, scores, fps = load_track(Path(row["pose"]))
            duration = row.get("official_duration_s")
            if (fps is None or not 6 <= fps <= 60 or len(points) > 256 or not duration
                    or abs(len(points) / fps - duration) > max(.2, 2 / fps)):
                raise ValueError(f"Pose window does not match the official duration for {row['id']}")
            digest = hashlib.sha256(points.tobytes())
            digest.update(scores.tobytes())
            fingerprint = digest.digest()
            if fingerprint in poses:
                raise ValueError(f"Duplicate pose content for {row['id']}")
            poses.add(fingerprint)
            group.add(recording)
        groups.append(group)
    if groups[0] & groups[1]:
        raise ValueError("Training and validation share source recordings")
    return train, validation, settings


def train_projection(work, train, validation, settings, output):
    import torch

    if output.exists():
        raise ValueError("Choose a fresh adapter output path")
    if torch.cuda.device_count() != 1 or not torch.cuda.is_bf16_supported():
        raise RuntimeError("Use one BF16-capable GPU for the pilot")
    gpu_name = torch.cuda.get_device_name()
    free, total = torch.cuda.mem_get_info()
    if "A100" in gpu_name and (gpu_name != "NVIDIA A100-SXM4-80GB" or total < 75 * 2**30):
        raise RuntimeError("The A100 pilot requires one full NVIDIA A100-SXM4-80GB device")
    print(json.dumps({"gpu": gpu_name, "free_mib": free // 2**20,
                      "total_mib": total // 2**20}), flush=True)
    checkpoint = work / "assets/how2sign_pose_only_slt.pth"
    translator = Translator(work / "Uni-Sign", work / "assets/mt5-base", checkpoint)
    model = translator.model
    model.requires_grad_(False)
    projection = model.pose_proj.float().requires_grad_(True)
    optimizer = torch.optim.AdamW(projection.parameters(), lr=settings["learning_rate"],
                                  weight_decay=settings["weight_decay"])
    rng = random.Random(settings["seed"])
    torch.manual_seed(settings["seed"])
    with checkpoint.open("rb") as file:
        digest = hashlib.file_digest(file, "sha256").hexdigest()

    def loss(row):
        points, scores, _ = load_track(Path(row["pose"]))
        with torch.autocast("cuda", dtype=torch.bfloat16):
            return model(translator.source(points, scores), {"gt_sentence": [row["reference"]]})["loss"]

    best = float("inf")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".partial")
    for step in range(1, settings["optimizer_steps"] + 1):
        optimizer.zero_grad(set_to_none=True)
        for _ in range(settings["accumulation"]):
            value = loss(rng.choice(train))
            if not torch.isfinite(value):
                raise RuntimeError("Non-finite training loss")
            (value / settings["accumulation"]).backward()
        torch.nn.utils.clip_grad_norm_(projection.parameters(), 1., error_if_nonfinite=True)
        optimizer.step()
        if step % settings["validation_interval"] == 0:
            with torch.no_grad():
                score = sum(float(loss(row)) for row in validation) / len(validation)
            if not float("-inf") < score < float("inf"):
                raise RuntimeError("Non-finite validation loss")
            print(json.dumps({"step": step, "validation_loss": score}), flush=True)
            if score < best:
                best = score
                torch.save({"base_sha256": digest, "step": step, "validation_loss": score,
                            "pose_proj": {k:v.detach().cpu() for k,v in projection.state_dict().items()}}, temporary)
                temporary.replace(output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("plan", type=Path)
    parser.add_argument("--run", action="store_true", help="Start the trial on a separately authorized GPU")
    parser.add_argument("--work", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.run and (not args.work or not args.output):
        parser.error("--run requires --work and --output")
    train, validation, settings = validate(args.plan)
    print(json.dumps({"train": len(train), "validation": len(validation), "mode": "train" if args.run else "check"}))
    if args.run:
        train_projection(args.work, train, validation, settings, args.output)
