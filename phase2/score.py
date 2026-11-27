"""Summarize experiment logs and score explicitly reviewed caption events."""

import argparse
from collections import defaultdict
import json
import math
from pathlib import Path


def read_rows(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def distribution(values):
    ordered = sorted(values)
    return {
        "n": len(ordered),
        "p50": round(ordered[math.ceil(0.50 * len(ordered)) - 1], 3) if ordered else None,
        "p95": round(ordered[math.ceil(0.95 * len(ordered)) - 1], 3) if ordered else None,
    }


def erasure(events):
    deleted = introduced = revisions = 0
    previous = []
    for event in events:
        current = event["text"].split()
        common = 0
        for old, new in zip(previous, current):
            if old != new:
                break
            common += 1
        deleted += len(previous) - common
        introduced += len(current) - common
        revisions += int(common < len(previous))
        previous = current if event["kind"] == "partial" else []
    return deleted, introduced, revisions


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("events", type=Path)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--review", type=Path)
    parser.add_argument("--bleu", action="store_true")
    args = parser.parse_args()
    events = read_rows(args.events)
    if not events or events[-1]["kind"] != "gpu_memory":
        parser.error("Expected a complete benchmark log ending in gpu_memory")
    captions = defaultdict(list)
    for event in events:
        if event["kind"] in {"partial", "final", "uncertain"}:
            captions[event["id"]].append(event)
    erased, added, revisions = map(sum, zip(*(erasure(group) for group in captions.values()))) if captions else (0, 0, 0)
    replay = [event for event in events if event["kind"] == "replay_summary"]
    total_source_s = sum(event["source_ms"] for event in replay) / 1000
    scheduled = sum(event["scheduled_updates"] for event in replay)
    dropped = sum(event["dropped_updates"] for event in replay)
    measured = [event for event in events if "work_ms" in event]
    result = {
        "cold_start": next((event for event in events if event["kind"] == "cold_start"), None),
        "model_work_ms": distribution([event["work_ms"] for event in measured]),
        "queue_ms": distribution([event["queue_ms"] for event in measured if "queue_ms" in event]),
        "pose_normalize_ms": distribution([event["prep_ms"] for event in measured]),
        "pose_encode_ms": distribution([event["pose_encode_ms"] for event in measured]),
        "generation_ms": distribution([event["generation_ms"] for event in measured]),
        "stream_work_rtf": distribution([event["work_rtf"] for event in replay]),
        "stream_dropped_updates": dropped,
        "stream_dropped_update_fraction": dropped / scheduled if scheduled else None,
        "stream_model_calls": sum(event.get("model_calls", 0) for event in replay),
        "partial_captions": sum(event["kind"] == "partial" for group in captions.values() for event in group),
        "final_captions": sum(event["kind"] == "final" for group in captions.values() for event in group),
        "suppressed_updates": sum(event["kind"] == "telemetry" for event in events),
        "normalized_erasure": erased / max(1, added),
        "revisions_per_source_min": revisions * 60 / total_source_s if total_source_s else None,
        "uncertain_segments": sum(event["kind"] == "uncertain" for group in captions.values() for event in group),
        "gpu_memory": events[-1],
    }
    if args.manifest:
        idle_ids = {row["id"] for row in read_rows(args.manifest) if row.get("idle_only")}
        idle_seconds = sum(event["source_ms"] for event in replay if event["id"] in idle_ids) / 1000
        result["idle_finals"] = sum(event["kind"] == "final" for key in idle_ids for event in captions[key])
        result["idle_nonempty_partials"] = sum(
            event["kind"] == "partial" and bool(event["text"]) for key in idle_ids for event in captions[key]
        )
        result["idle_seconds"] = idle_seconds
    if args.review:
        first, final = [], []
        meaning = {"preserved": 0, "partial": 0, "wrong": 0, "missing": 0}
        seen = set()
        assigned_finals = set()
        rows = read_rows(args.review)
        for row in rows:
            key = row["id"], row["item"]
            if key in seen or row["end_ms"] <= row["start_ms"]:
                raise ValueError(f"Duplicate item or invalid times: {key}")
            seen.add(key)
            meaning[row["meaning"]] += 1
            group = captions[row["id"]]
            for review_key, allowed, origin, values in [
                ("first_useful_emit_ms", {"partial", "final"}, "start_ms", first),
                ("final_emit_ms", {"final"}, "end_ms", final),
            ]:
                stamp = row.get(review_key)
                if stamp is None:
                    continue
                if not any(event["kind"] in allowed and event["emit_ms"] == stamp and event["text"] for event in group):
                    raise ValueError(f"{key}: {stamp} does not identify a usable {review_key} event")
                if review_key == "first_useful_emit_ms" and stamp < row["start_ms"]:
                    raise ValueError(f"{key}: useful caption predates signing")
                if review_key == "final_emit_ms":
                    final_key = row["id"], stamp
                    if final_key in assigned_finals:
                        raise ValueError(f"{key}: final already attributed to another item")
                    assigned_finals.add(final_key)
                values.append(stamp - row[origin])
        result.update(
            review_items=len(rows), meaning=meaning, first_useful_from_onset_ms=distribution(first),
            final_from_end_ms=distribution(final), first_useful_missing=len(rows) - len(first),
            final_missing=len(rows) - len(final), early_finals=sum(delta < 0 for delta in final),
            unassigned_finals=sum(
                event["kind"] == "final" and (key, event["emit_ms"]) not in assigned_finals
                for key, group in captions.items() for event in group
            ),
        )
    if args.bleu:
        from sacrebleu import BLEU

        clips = [event for event in events if event["kind"] == "oracle" and event.get("reference")]
        if not clips:
            parser.error("--bleu requires oracle captions with references")
        metric = BLEU(tokenize="13a")
        result["oracle_bleu"] = {
            "score": metric.corpus_score([clip["text"] for clip in clips], [[clip["reference"] for clip in clips]]).score,
            "signature": str(metric.get_signature()), "clips": len(clips),
        }
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
