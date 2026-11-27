#!/usr/bin/env bash
set -euo pipefail
python=${1:?expected Pod Python path}
target=${2:?expected isolated install directory}
here=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
"$python" -c 'import platform,sys; assert sys.version_info[:2] == (3,12) and platform.machine() == "x86_64"'
if [[ -f "$target/onnxruntime/__init__.py" && -f "$target/.verified-1.20.2" ]]; then
    exit 0
fi
mkdir -p -- "$(dirname -- "$target")"
temp=$(mktemp -d "$target.partial.XXXXXX")
trap 'rm -rf -- "$temp"' EXIT
"$python" -m pip install --disable-pip-version-check --no-deps --only-binary=:all: \
    --require-hashes --index-url https://pypi.org/simple --timeout 8 --retries 1 \
    --target "$temp" -r "$here/requirements-cuda-benchmark.txt"
touch "$temp/.verified-1.20.2"
# An existing cache is left intact. The worker verifies the imported provider again.
mv -T -n -- "$temp" "$target"
test -f "$target/.verified-1.20.2"
