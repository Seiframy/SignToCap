"""Evaluate an approved alternate backend's predictions on the same held-out references."""

import argparse
import json
from pathlib import Path


def rows(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def predictions_for(path, references):
    raw = rows(path)
    supplied = [row for row in raw if row["kind"] == "oracle"] if raw and "kind" in raw[0] else raw
    predictions = {row["id"]: row["text"] for row in supplied}
    if not references or set(predictions) != set(references) or len(supplied) != len(predictions):
        raise ValueError("Supply exactly one prediction per held-out ID")
    return predictions


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("predictions", type=Path, help="JSONL with one id and text for each held-out clip")
    parser.add_argument("--baseline", type=Path, help="Paired base checkpoint predictions")
    args = parser.parse_args()
    references = {row["id"]: row["reference"] for row in rows(args.manifest) if row["split"] == "heldout"}
    predictions = predictions_for(args.predictions, references)
    from sacrebleu.metrics import BLEU, CHRF
    bleu = BLEU(tokenize="13a")
    scores = bleu.corpus_score([predictions[key] for key in references], [list(references.values())])
    chrf = CHRF()
    reference_set = [list(references.values())]
    candidate_score = chrf.corpus_score([predictions[key] for key in references], reference_set).score
    report = {"bleu": scores.score, "signature": str(bleu.get_signature()), "chrf": candidate_score,
              "pairs": [{"id": key, "reference": value, "prediction": predictions[key]}
                        for key, value in references.items()]}
    if args.baseline:
        baseline = predictions_for(args.baseline, references)
        baseline_score = chrf.corpus_score([baseline[key] for key in references], reference_set).score
        report.update(baseline_chrf=baseline_score, delta_chrf=candidate_score-baseline_score)
        for pair in report["pairs"]:
            pair["baseline"] = baseline[pair["id"]]
    print(json.dumps(report, indent=2))
