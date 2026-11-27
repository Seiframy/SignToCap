"""One warm-model process for the targeted Phase 3 GPU experiment."""

from dataclasses import asdict, dataclass, replace
import json
from pathlib import Path
import shutil
import subprocess
import sys
from time import perf_counter

PHASE2 = Path(__file__).resolve().parents[1] / "phase2"
sys.path.insert(0, str(PHASE2))
from benchmark import emit, oracle, read_rows, replay
from pose_data import load_track
from unisign import Translator


@dataclass
class Settings:
    controller: str = "baseline"
    policy: str = "stable-punctuation"
    buffer_s: int = 8
    cadence_ms: int = 500
    motion: float = 0.7
    beams: int = 1
    tokens: int = 48
    parity: bool = False
    decode_interval_ms: int = 900
    partial_tokens: int = 24


def summarize(path, basename, manifest, mode):
    args = [sys.executable, str(PHASE2 / "score.py"), str(path), "--manifest", str(manifest)]
    if mode == "oracle":
        args.append("--bleu")
    with (path.parent / f"{basename}-summary.json").open("wb") as out:
        subprocess.run(args, stdout=out, check=True)


def main(work, output):
    audit = work / "prepared/phase3"
    continuous = work / "prepared/continuous/continuous.jsonl"
    metrics = json.loads((audit / "pose-audit.json").read_text(encoding="utf-8"))
    ids = {row["id"] for row in metrics if row["frame_counts_match"]}
    pub, fresh = [read_rows(audit / f"{arm}.jsonl") for arm in ("publisher", "fresh")]
    if [(r["id"], r["reference"], r["split"]) for r in pub] != [(r["id"], r["reference"], r["split"]) for r in fresh]:
        raise ValueError("Pose pairs do not have identical IDs and references")
    for arm in ("pose-audit.json", "publisher.jsonl", "fresh.jsonl"):
        shutil.copyfile(audit / arm, output / arm)
    shutil.copyfile(continuous, output / "continuous-inputs.jsonl")
    plans = []
    for split in ("diagnostic", "heldout"):
        if not any(row["split"] == split and row["id"] in ids for row in pub):
            raise ValueError(f"No verified exact-frame matches in {split}")
        for name, rows in (("publisher", pub), ("fresh", fresh)):
            subset = [row for row in rows if row["split"] == split and row["id"] in ids]
            plans.append((f"{split}-{name}", "oracle", audit / f"{name}.jsonl", subset,
                          Settings(beams=4, tokens=100)))
    original = read_rows(continuous)
    for name, options in (("baseline", Settings()), ("adaptive", replace(Settings(), controller="adaptive"))):
        plans.append((f"historical-replay-{name}", "replay", continuous, original, options))
    heldout_fresh = [row for row in fresh if row["split"] == "heldout" and row["id"] in ids]
    plans.append(("heldout-fresh-3fps", "oracle", audit / "fresh.jsonl", heldout_fresh,
                  Settings(beams=4, tokens=100)))

    started = perf_counter()
    translator = Translator(work / "Uni-Sign", work / "assets/mt5-base", work / "assets/how2sign_pose_only_slt.pth")
    startup = (perf_counter() - started) * 1000
    for basename, mode, manifest, rows, options in plans:
        print(f"{basename}: {len(rows)} tracks", flush=True)
        first = load_track(manifest.parent / rows[0]["pose"])
        for _ in range(2):
            translator.translate(first[0][:60], first[1][:60], beams=options.beams, tokens=options.tokens)
        translator.torch.cuda.reset_peak_memory_stats()
        path = output / f"{basename}-events.jsonl"
        with path.open("w", encoding="utf-8") as out:
            emit(out, kind="config", mode=mode, shared_model_load_ms=startup,
                 sampling_fps=3 if basename == "heldout-fresh-3fps" else None, **asdict(options))
            for row in rows:
                track = load_track(manifest.parent / row["pose"])
                if basename == "heldout-fresh-3fps":
                    points, scores, fps = track
                    if not fps or fps < 3 or not (fps / 3).is_integer():
                        raise ValueError(f"Cannot simulate an exact 3 fps cadence for {row['id']}")
                    step = int(fps / 3)
                    track = points[::step], scores[::step], 3.0
                if mode == "oracle":
                    oracle(options, out, translator, row, track)
                else:
                    replay(options, out, translator, row, track, None)
            emit(out, kind="gpu_memory", **translator.memory())
        summarize(path, basename, manifest, mode)


if __name__ == "__main__":
    main(*(Path(value).resolve() for value in sys.argv[1:]))
