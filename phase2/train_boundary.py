"""Train a small causal end-of-segment classifier on independently labelled continuous tracks."""

import argparse
from hashlib import file_digest, sha256
import json
from pathlib import Path

from captions import activity_features, boundary_features
from pose_data import load_track


def network(torch):
    nn = torch.nn
    return nn.Sequential(nn.Linear(4, 16), nn.ReLU(), nn.Linear(16, 1))


def examples(torch, manifest, rows):
    features, labels = [], []
    for row in rows:
        track = load_track(manifest.parent / row["pose"])
        _, _, fps = track
        if fps is None:
            raise ValueError(f"Missing source FPS for {row['id']}")
        motion, visible = activity_features(*track)
        bounds = sorted(float(segment["end_ms"]) for segment in row["segments"])
        if any(not 0 < end <= len(motion) / fps * 1000 for end in bounds):
            raise ValueError(f"Boundary outside track {row['id']}")
        for end in range(max(2, round(0.85 * fps)), len(motion) + 1, max(1, round(fps / 10))):
            timestamp = end / fps * 1000
            features.append(boundary_features(motion, visible, end, fps))
            labels.append(float(any(200 <= timestamp - cut <= 700 for cut in bounds)))
    if set(labels) != {0, 1}:
        raise ValueError("Each split requires positive and negative windows")
    return torch.tensor(features), torch.tensor(labels)


def load_predictor(path):
    import torch

    saved = torch.load(path, weights_only=True, map_location="cpu")
    model = network(torch)
    model.load_state_dict(saved["model"])
    model.eval()

    @torch.inference_mode()
    def predict(features):
        x = (torch.tensor(features) - saved["mean"]) / saved["std"]
        return bool(model(x).sigmoid().item() >= saved["threshold"])

    return predict


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--eval-every", type=int, default=50)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.steps < 1 or args.eval_every < 1:
        parser.error("Steps and evaluation interval must be positive")
    if args.output.exists() and not args.resume:
        parser.error("Use a fresh output directory or --resume")
    import torch

    torch.set_num_threads(1)
    torch.manual_seed(42)
    random = torch.Generator().manual_seed(42)
    rows = [json.loads(line) for line in args.manifest.read_text(encoding="utf-8").splitlines() if line.strip()]
    train = [row for row in rows if row["split"] == "train"]
    dev = [row for row in rows if row["split"] == "dev"]
    train_groups = {row["group"] for row in train}
    dev_groups = {row["group"] for row in dev}
    if not train_groups or not dev_groups or train_groups & dev_groups or len(train) + len(dev) != len(rows):
        parser.error("Manifest must have group-disjoint train/dev rows only")
    x, y = examples(torch, args.manifest, train)
    xv, yv = examples(torch, args.manifest, dev)
    mean = x.mean(0)
    std = x.std(0, unbiased=False).clamp_min(1e-6)
    positive_weight = ((1 - y).sum() / y.sum()).clamp(1, 4).to(args.device)
    x, xv = ((x - mean) / std).to(args.device), ((xv - mean) / std).to(args.device)
    y, yv = y.to(args.device), yv.to(args.device)
    model = network(torch).to(args.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
    loss = torch.nn.BCEWithLogitsLoss(pos_weight=positive_weight)
    step, best = 0, -1.0
    args.output.mkdir(parents=True, exist_ok=True)
    latest = args.output / "last.pt"
    fingerprint = sha256(args.manifest.read_bytes())
    for row in rows:
        with (args.manifest.parent / row["pose"]).open("rb") as source:
            fingerprint.update(file_digest(source, "sha256").digest())
    dataset_hash = fingerprint.hexdigest()
    if args.resume:
        saved = torch.load(latest, weights_only=True, map_location="cpu")
        if saved["dataset_sha256"] != dataset_hash:
            parser.error("Resume requires the same manifest and pose data")
        if saved["step"] >= args.steps:
            parser.error("--steps must exceed the saved step when resuming")
        model.load_state_dict(saved["model"])
        optimizer.load_state_dict(saved["optimizer"])
        random.set_state(saved["rng"])
        step, best = saved["step"], saved["best"]
    for step in range(step + 1, args.steps + 1):
        index = torch.randint(len(x), (128,), generator=random).to(args.device)
        optimizer.zero_grad(set_to_none=True)
        loss(model(x[index]).flatten(), y[index]).backward()
        optimizer.step()
        if step % args.eval_every and step != args.steps:
            continue
        with torch.inference_mode():
            prob = model(xv).flatten().sigmoid()
            positive = yv.bool()
            candidates = []
            for threshold in (0.4, 0.5, 0.6, 0.7, 0.8):
                pred = prob >= threshold
                tp = int((pred & positive).sum())
                fp = int((pred & ~positive).sum())
                fn = int((~pred & positive).sum())
                f05 = 1.25 * tp / max(1, 1.25 * tp + fp + 0.25 * fn)
                candidates.append((f05, threshold, tp, fp, fn))
            metric, threshold, tp, fp, fn = max(candidates)
        improved = metric > best
        best = max(best, metric)
        output = {
            "model": model.state_dict(), "optimizer": optimizer.state_dict(), "rng": random.get_state(),
            "step": step, "best": best, "mean": mean, "std": std, "threshold": threshold,
            "dataset_sha256": dataset_hash,
        }
        for dest in ([latest, args.output / "best.pt"] if improved else [latest]):
            torch.save(output, dest.with_suffix(".tmp"))
            dest.with_suffix(".tmp").replace(dest)
        print(json.dumps({
            "step": step, "dev_window_f05": metric, "threshold": threshold,
            "tp": tp, "fp": fp, "fn": fn, "train_windows": len(x), "dev_windows": len(xv),
        }), flush=True)


if __name__ == "__main__":
    main()
