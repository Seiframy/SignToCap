"""Compare matched annotated sentences with the pinned checkpoint and selected extraction paths."""
import argparse
import contextlib
import io
import json
import math
import os
from pathlib import Path
import statistics
import sys
from tempfile import TemporaryDirectory

from desktop import video_tool
from pose_data import extract_video, load_track
from unisign import Translator


def segments(catalog, media, clip_ids):
    for line in catalog.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        for clip in row["segments"]:
            if clip["item"] not in clip_ids:
                continue
            offset = row["source"]["source_start_ms"]
            yield clip, media / (row["id"] + ".mp4"), offset + clip["start_ms"], offset + clip["end_ms"]


def extract(args, media, track, start, end, rate, legacy):
    from types import SimpleNamespace

    arguments = ["video", args.pose_upstream, media, track, "--start-ms", start, "--end-ms", end,
                 "--verify-cadence", "--ort-threads", 2]
    if rate is not None:
        arguments += ["--sample-fps", rate]
    if legacy:
        arguments.append("--legacy-detection")
    if os.name == "nt":
        return video_tool(*arguments)
    with contextlib.redirect_stdout(io.StringIO()):
        return extract_video(SimpleNamespace(upstream=args.pose_upstream, video=media, output=track,
                             start_ms=start, end_ms=end, verify_cadence=True, ort_threads=2,
                             sample_fps=rate, legacy_detection=legacy))


def accuracy(rows, with_model):
    if not with_model:
        return None
    try:
        from sacrebleu.metrics import BLEU, CHRF
    except ImportError:
        return {"error": "Install the pinned sacrebleu dependency for corpus scores"}
    texts = [row["caption"] for row in rows]
    references = [[row["reference"] for row in rows]]
    bleu, chrf = BLEU(tokenize="13a"), CHRF()
    return {"bleu": bleu.corpus_score(texts, references).score, "chrf": chrf.corpus_score(texts, references).score,
            "bleu_signature": str(bleu.get_signature()), "chrf_signature": str(chrf.get_signature())}


def run(args):
    clips = list(segments(args.catalog, args.media, set(args.clip)))
    if len(clips) != len(args.clip) or len({clip["item"] for clip, *_ in clips}) != len(clips):
        raise ValueError("Each requested annotated clip must match exactly once")
    model = None
    if os.name != "nt":
        os.environ.setdefault("TORCH_HOME", str(args.work / "cache/rtmlib"))
    if args.device:
        import torch
        if args.device == "cpu":
            torch.set_num_threads(4)
        model = Translator(args.model_upstream or args.work / "Uni-Sign", args.work / "assets/mt5-base",
                           args.work / "assets/how2sign_pose_only_slt.pth", device=args.device)
    configs = (("legacy-native", None, True), ("checked-native", None, False),
               ("checked-6", 6, False), ("checked-12", 12, False), ("checked-24", 24, False))
    results = {name: [] for name, *_ in configs}
    warmed = False
    with TemporaryDirectory(prefix="SignToCap-phase4-") as directory:
        track = Path(directory) / "track.npz"
        for clip, media, start, end in clips:
            for name, rate, legacy in configs:
                metrics = extract(args, media, track, start, end, rate, legacy)
                points, scores, fps = load_track(track)
                if len(points) > 256:
                    raise ValueError("Choose a complete benchmark clip with at most 256 native frames")
                decoded = metrics["decoded_frames"]
                expected = math.ceil(decoded * fps / metrics["source_fps"])
                row = dict(clip=clip["item"], reference=clip["reference"], frames=len(points),
                           scheduled=expected, dropped=expected-len(points), source_frames=decoded,
                           pose_median_ms=metrics["pose_p50_ms"], pose_p95_ms=metrics["pose_p95_ms"],
                           extraction_ms=metrics["extraction_ms"],
                           no_pose_frames=int((~(scores > 0).any(axis=(1, 2))).sum()))
                if model:
                    if not warmed:
                        model.translate(points[:1], scores[:1], tokens=8)
                        warmed = True
                    caption, _, timing = model.translate(points, scores)
                    row.update(caption=caption, model_ms=timing["work_ms"], meaning="review pending",
                               pipeline_ms=metrics["extraction_ms"] + timing["work_ms"])
                results[name].append(row)
                print(f"{clip['item']} {name}: {len(points)}/{expected} sampled frames", file=sys.stderr, flush=True)
    report = {"model_device": args.device, "conditions": {}}
    lines = ["# Phase 4 comparison", "", "CPU or Pod pose preparation is timed separately from model work.",
             "Pipeline time sums local extraction and model work. It omits UI and network transport.",
             "", "| Condition | Retained / scheduled | Median pose ms | Median extraction ms | Median model ms | Median pipeline ms | chrF | BLEU |",
             "| --- | --- | --- | --- | --- | --- | --- | --- |"]
    for name, group in results.items():
        score = accuracy(group, bool(model))
        stats = {field: statistics.median(row[field] for row in group) for field in
                 ("pose_median_ms", "extraction_ms", "model_ms", "pipeline_ms") if field in group[0]}
        retained, expected = (sum(row[key] for row in group) for key in ("frames", "scheduled"))
        report["conditions"][name] = {"accuracy": score, "performance": stats, "clips": group}
        values = [stats.get(key) for key in ("pose_median_ms", "extraction_ms", "model_ms", "pipeline_ms")]
        values += [(score or {}).get(key) for key in ("chrf", "bleu")]
        lines.append(f"| {name} | {retained}/{expected} | " +
                     " | ".join("—" if x is None else f"{x:.2f}" for x in values) + " |")
    if model:
        lines += ["", "## Meaning review", "", "Compare the paired references and captions in the JSON. "
                  "Record missing, altered, and preserved meaning manually."]
    args.output.with_suffix(".json").write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    args.output.with_suffix(".md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("catalog", type=Path)
    cli.add_argument("media", type=Path)
    cli.add_argument("output", type=Path, help="Output path stem for private JSON and Markdown reports")
    cli.add_argument("--clip", action="append", required=True)
    cli.add_argument("--work", required=True, type=Path)
    cli.add_argument("--pose-upstream", type=Path)
    cli.add_argument("--model-upstream", type=Path, help="Patched Uni-Sign source if separate from the pose runtime")
    cli.add_argument("--device", choices=("cpu", "cuda"), help="Omit to measure decoding and pose extraction only")
    args = cli.parse_args()
    args.pose_upstream = args.pose_upstream or args.work / "Uni-Sign"
    run(args)
