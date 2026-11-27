#!/usr/bin/env bash
# Run on the already guarded, manually started Pod. This does not control the Pod.
set -euo pipefail
root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
work=/workspace/signgpt-validation
python="$work/.venv-py312/bin/python"
command=${1:?Use check or run}
data=${2:?Provide the data directory}
out=${3:?Provide a fresh output directory}
cd "$root"
if [[ "$command" == check ]]; then
    "$python" -B phase3/accuracy_plan.py "$out" --official-train "$data/train.jsonl" \
        --official-train-tsv "$data/how2sign_realigned_train.csv"
    "$python" -B phase3/train_pose_adapter.py "$out"
    "$python" -B -c 'import sacrebleu'
    exit 0
fi
if [[ "$command" != run ]]; then exit 2; fi
mkdir -m 700 -- "$out/run"
exec timeout --kill-after=10s 8m bash -euo pipefail -c '
    python=$1 work=$2 out=$3
    export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
    "$python" -B phase3/train_pose_adapter.py "$out" --run --work "$work" --output "$out/run/pose-adapter.pt"
    for arm in fresh publisher; do
        sampling=()
        if [[ "$arm" == fresh ]]; then sampling=(--sample-fps 6); fi
        options=(--upstream "$work/Uni-Sign" --mt5 "$work/assets/mt5-base"
                 --checkpoint "$work/assets/how2sign_pose_only_slt.pth" "${sampling[@]}")
        manifest="$out/heldout-$arm.jsonl"
        "$python" -B phase2/benchmark.py oracle "$manifest" "$out/run/$arm-base.jsonl" "${options[@]}"
        "$python" -B phase2/benchmark.py oracle "$manifest" "$out/run/$arm-adapter.jsonl" "${options[@]}" \
            --pose-adapter "$out/run/pose-adapter.pt"
        "$python" -B phase3/evaluate_external.py "$manifest" "$out/run/$arm-adapter.jsonl" \
            --baseline "$out/run/$arm-base.jsonl" > "$out/run/$arm-score.json"
    done
    printf "Pilot complete. Review both score files in %s/run before adoption.\n" "$out"
' bash "$python" "$work" "$out"
