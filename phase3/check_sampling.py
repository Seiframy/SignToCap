"""Compare native and 6 FPS input and measure the effect of real surrounding footage."""
import argparse
import json
from pathlib import Path
import sys

from accuracy_plan import HERE, ROOT, rows

sys.path.insert(0, str(ROOT / "phase2"))
from captions import activity_features
from pose_data import load_track
from unisign import Translator


def at_six(points, scores, fps):
    import numpy as np

    if fps < 6:
        raise ValueError("Cannot sample at 6 FPS from a slower recording")
    indices = (np.arange(int(np.ceil(len(points) * 6 / fps))) * fps / 6).astype(int)
    return points[indices], scores[indices]


def compare(work, continuous, upstream, comparison, clip_id):
    import numpy as np
    import torch

    torch.set_num_threads(4)
    model = Translator(upstream, work / "assets/mt5-base",
                       work / "assets/how2sign_pose_only_slt.pth", device="cpu")

    def predict(points, scores):
        return model.translate(points, scores)[0]

    if comparison == "rate":
        for row in rows(HERE / "audit-data/fresh.jsonl"):
            if row["split"] != "heldout" or (clip_id and row["id"] != clip_id):
                continue
            points, scores, fps = load_track(HERE / "audit-data" / row["pose"])
            if len(points) > 256:
                print(json.dumps(dict(id=row["id"], skipped="native window exceeds 256 frames")), flush=True)
                continue
            six = at_six(points, scores, fps)
            print(json.dumps(dict(id=row["id"], reference=row["reference"], native_fps=fps,
                                  native=predict(points, scores), six=predict(*six))), flush=True)
        return

    for row in rows(continuous / "continuous.jsonl"):
        points, scores, fps = load_track(continuous / row["pose"])
        for i, clip in enumerate(row["segments"]):
            start, end = (round(clip[k] * fps / 1000) for k in ("start_ms", "end_ms"))
            previous = round(row["segments"][i-1]["end_ms"] * fps / 1000) if i else 0
            following = round(row["segments"][i+1]["start_ms"] * fps / 1000) if i+1 < len(row["segments"]) else len(points)
            # Keep the same 6 FPS grid when adding context to avoid confounding sample phase.
            if fps != 24:
                raise ValueError("Context comparison expects the staged 24 FPS tracks")
            grid = np.arange(0, len(points), 4)
            core = grid[(grid >= start) & (grid < end)]
            extra = grid[((grid >= max(previous, start-24)) & (grid < start)) |
                         ((grid >= end) & (grid < min(following, end+24)))]
            if not len(core) or len(extra) < 2:
                continue
            expanded = np.sort(np.concatenate((core, extra)))
            motion, _ = activity_features(points[expanded], scores[expanded], 6)
            print(json.dumps(dict(id=clip["item"], reference=clip["reference"],
                                  context_frames=len(extra), context_motion=float(np.median(motion[np.isin(expanded, extra)])),
                                  core_motion=float(np.median(motion[np.isin(expanded, core)])),
                                  six=predict(points[core], scores[core]),
                                  with_context=predict(points[expanded], scores[expanded]))), flush=True)


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("work", type=Path)
    cli.add_argument("--upstream", required=True, type=Path)
    cli.add_argument("--comparison", choices=("context", "rate"), required=True)
    cli.add_argument("--clip-id", help="Compare one held-out clip in rate mode")
    args = cli.parse_args()
    if args.clip_id and args.comparison != "rate":
        cli.error("--clip-id requires rate mode")
    compare(args.work, args.work / "continuous/prepared", args.upstream, args.comparison, args.clip_id)
