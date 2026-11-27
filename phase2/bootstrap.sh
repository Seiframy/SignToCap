#!/usr/bin/env bash
set -euo pipefail

# Run only on the authorized Linux Pod.
phase_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
run_dir=${1:?usage: bash bootstrap.sh /workspace/signgpt-run}
mkdir -p "$run_dir"
cd "$run_dir"
[[ $(uname -s) == Linux ]]
unset PYTHONHOME PYTHONPATH
python_bin=${SIGNGPT_PYTHON:-python3}
python_tag=$("$python_bin" -c 'import sys; print(f"{sys.version_info.major}{sys.version_info.minor}")')
if [[ $python_tag != 311 && $python_tag != 312 ]]; then
    echo 'Use Python 3.11 or 3.12 for the pinned inference wheels.' >&2
    exit 1
fi
venv="$PWD/.venv-py$python_tag"
"$python_bin" -m venv "$venv"
source "$venv/bin/activate"
printf '%s\n' "$venv" > venv-path.txt
export PIP_CACHE_DIR="$PWD/cache/pip"
python -m pip install pip==25.1.1 setuptools==75.8.0 wheel==0.45.1
python -m pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu124
python -m pip install -r "$phase_dir/requirements.txt"

code_rev=eed438bcb49e30405cd6ccdfcccca330c134e830
code_origin=https://github.com/ZechengLi19/Uni-Sign.git
if [[ ! -d Uni-Sign/.git ]]; then
    git init Uni-Sign
fi
if ! git -C Uni-Sign config --get remote.origin.url >/dev/null; then
    git -C Uni-Sign remote add origin "$code_origin"
fi
[[ $(git -C Uni-Sign config --get remote.origin.url) == "$code_origin" ]]
if ! git -C Uni-Sign rev-parse --verify HEAD >/dev/null 2>&1; then
    git -C Uni-Sign fetch --depth=1 "$code_origin" "$code_rev"
    git -C Uni-Sign checkout --detach FETCH_HEAD
fi
[[ $(git -C Uni-Sign rev-parse HEAD) == "$code_rev" ]]
for patch in source-only.patch pose-imports.patch; do
    if ! git -C Uni-Sign apply --reverse --check "$phase_dir/$patch"; then
        git -C Uni-Sign apply --check "$phase_dir/$patch"
        git -C Uni-Sign apply "$phase_dir/$patch"
    fi
done

# Check GPU wheels, a real BF16 CUDA operation and CPU ORT before downloading assets.
python -m pip check
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv > gpu.txt
PYTHONPATH="$PWD/Uni-Sign/demo/rtmlib-main:$PWD/Uni-Sign" python -c \
    'import torch, torchvision, onnxruntime; from models import Uni_Sign; from datasets import load_part_kp; from rtmlib import Wholebody; assert torch.version.cuda == "12.4" and torch.cuda.device_count() == 1 and torch.cuda.is_bf16_supported(); assert "CPUExecutionProvider" in onnxruntime.get_available_providers(); x = torch.ones((64, 64), device="cuda", dtype=torch.bfloat16); assert (x @ x)[0, 0].item() == 64; print(torch.__version__, torch.version.cuda, torch.cuda.get_device_name(), onnxruntime.get_available_providers())'

python "$phase_dir/prepare_weights.py" assets

python -m pip freeze > requirements-resolved.txt
git -C Uni-Sign rev-parse HEAD > unisign-revision.txt
(cd assets && sha256sum --check staged.sha256)
cp assets/staged.sha256 assets.sha256
