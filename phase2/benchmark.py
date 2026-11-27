"""Benchmark an official checkpoint over oracle clips or timestamp-paced continuous pose replay."""

import argparse
import json
import math
from pathlib import Path
from time import perf_counter, sleep

from captions import AdaptiveStreamingSession, StreamingSession
from pose_data import load_track
from unisign import Translator


def read_rows(manifest):
    with manifest.open(encoding="utf-8") as source:
        return [json.loads(line) for line in source if line.strip()]


def emit(out, **event):
    print(json.dumps(event, ensure_ascii=False), file=out, flush=True)


def oracle(args, out, translator, row, track):
    keypoints, scores, fps = track
    if args.sample_fps is not None:
        if fps is None or fps < args.sample_fps:
            raise ValueError("Sampling requires a known source rate at least as high as the requested rate")
        steps = math.ceil(len(keypoints) * args.sample_fps / fps)
        indices = (translator.np.arange(steps) * fps / args.sample_fps).astype(int)
        keypoints, scores = keypoints[indices], scores[indices]
    indices = translator.np.linspace(0, len(keypoints) - 1, min(len(keypoints), 256), dtype=int)
    keypoints, scores = keypoints[indices], scores[indices]
    text, ids, timings = translator.translate(keypoints, scores, beams=args.beams, tokens=args.tokens)
    parity = None
    if args.parity:
        _, reference_ids, _ = translator.translate(
            keypoints, scores, beams=args.beams, tokens=args.tokens, source_only=False,
        )
        parity = ids == reference_ids
        if not parity:
            raise RuntimeError(f"Source-only token parity failed for {row['id']}")
    emit(out, kind="oracle", id=row["id"], text=text, reference=row.get("reference"),
         input_ms=round(len(track[0]) / fps * 1000) if fps else None, parity=parity, **timings)


def replay(args, out, translator, row, track, head):
    keypoints, scores, fps = track
    if fps is None or fps > 30:
        raise ValueError("Replay requires a verified constant frame rate at or below 30 fps")
    kind = AdaptiveStreamingSession if args.controller == "adaptive" else StreamingSession
    session = kind(args, translator, fps, head)
    stride = max(1, round(args.cadence_ms * fps / 1000))
    tick = stride
    observed = dropped = calls = 0
    total_work_ms = 0
    started = perf_counter()
    while True:
        scheduled = min(len(keypoints), tick)
        sleep(max(0, started + scheduled / fps - perf_counter()))
        arrived = min(len(keypoints), max(scheduled, math.floor((perf_counter() - started) * fps)))
        end = min(len(keypoints), max(scheduled, arrived // stride * stride))
        if arrived == len(keypoints):
            end = arrived
        dropped += max(0, (end - scheduled) // stride)
        started_ms = (perf_counter() - started) * 1000
        event = session.push(keypoints[observed:end], scores[observed:end])
        completed_ms = (perf_counter() - started) * 1000
        observed = end
        if event is not None:
            if "work_ms" in event:
                total_work_ms += event["work_ms"]
                calls += 1
            emit(out, id=row["id"], emit_ms=completed_ms,
                 queue_ms=max(0, started_ms - event["source_end_ms"]), dropped_updates=dropped, **event)
        if end == len(keypoints):
            break
        tick = end + stride
    event = session.finish()
    if event is not None:
        emit(out, id=row["id"], emit_ms=(perf_counter() - started) * 1000, **event)
    endpoint_ms = len(keypoints) / fps * 1000
    emit(out, kind="replay_summary", id=row["id"], model_calls=calls, dropped_updates=dropped,
         scheduled_updates=math.ceil(len(keypoints) / stride), source_ms=endpoint_ms,
         model_work_ms=total_work_ms, work_rtf=total_work_ms / endpoint_ms)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["oracle", "replay"])
    parser.add_argument("manifest", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--upstream", type=Path, default=Path("Uni-Sign"))
    parser.add_argument("--mt5", type=Path, default=Path("assets/mt5-base"))
    parser.add_argument("--checkpoint", type=Path, default=Path("assets/how2sign_pose_only_slt.pth"))
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    parser.add_argument("--pose-adapter", type=Path)
    parser.add_argument("--sample-fps", type=int)
    parser.add_argument("--beams", type=int, default=1)
    parser.add_argument("--tokens", type=int, default=48)
    parser.add_argument("--parity", action="store_true", help="Compare to the official teacher-forced path")
    parser.add_argument("--cadence-ms", type=int, default=500)
    parser.add_argument("--buffer-s", type=float, default=8)
    parser.add_argument("--motion", type=float, default=0.7, help="Tune only on the pilot")
    parser.add_argument("--policy", choices=["pause", "stable-punctuation", "explicit"], default="stable-punctuation")
    parser.add_argument("--controller", choices=["baseline", "adaptive"], default="baseline")
    parser.add_argument("--decode-interval-ms", type=int, default=900)
    parser.add_argument("--partial-tokens", type=int, default=24)
    parser.add_argument("--boundary-head", type=Path)
    args = parser.parse_args()
    rows = read_rows(args.manifest)
    if not rows or len({row["id"] for row in rows}) != len(rows):
        parser.error("Manifest must contain uniquely identified tracks")
    if args.beams < 1 or args.tokens < 1 or args.cadence_ms < 100 or args.buffer_s < 1:
        parser.error("Invalid decode or replay parameters")
    if args.decode_interval_ms < 100 or args.partial_tokens < 1:
        parser.error("Invalid preview interval or token limit")
    if args.sample_fps is not None and (args.mode != "oracle" or not 1 <= args.sample_fps <= 60):
        parser.error("--sample-fps requires oracle mode and a rate from 1 to 60")
    if args.policy == "explicit" and args.controller == "baseline":
        parser.error("Explicit commitment requires the adaptive controller")
    if args.boundary_head and args.controller != "baseline":
        parser.error("The experimental boundary head is available only in the original controller")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    cold_started = perf_counter()
    translator = Translator(args.upstream, args.mt5, args.checkpoint, device=args.device)
    if args.pose_adapter:
        translator.load_pose_adapter(args.pose_adapter, args.checkpoint)
    load_ms = (perf_counter() - cold_started) * 1000
    first = load_track(args.manifest.parent / rows[0]["pose"])
    _, _, first_timings = translator.translate(first[0][:60], first[1][:60], beams=args.beams, tokens=args.tokens)
    for _ in range(2):
        translator.translate(first[0][:60], first[1][:60], beams=args.beams, tokens=args.tokens)
    head = None
    if args.boundary_head:
        from train_boundary import load_predictor

        head = load_predictor(args.boundary_head)
    with args.output.open("w", encoding="utf-8") as out:
        emit(out, kind="config", mode=args.mode, device=args.device, pose_adapter=bool(args.pose_adapter),
             sample_fps=args.sample_fps,
             beams=args.beams, tokens=args.tokens,
             policy=args.policy, cadence_ms=args.cadence_ms, buffer_s=args.buffer_s,
             motion=args.motion, boundary_head=bool(head), controller=args.controller,
             decode_interval_ms=args.decode_interval_ms, partial_tokens=args.partial_tokens)
        emit(out, kind="cold_start", model_load_ms=load_ms, first_model_work_ms=first_timings["work_ms"])
        for row in rows:
            track = load_track(args.manifest.parent / row["pose"])
            if args.mode == "oracle":
                oracle(args, out, translator, row, track)
            else:
                replay(args, out, translator, row, track, head)
        if args.device == "cuda":
            emit(out, kind="gpu_memory", **translator.memory())


if __name__ == "__main__":
    main()
