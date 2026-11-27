"""Stage pinned inference assets without installing PyTorch or opening a GPU session."""

import argparse
from hashlib import file_digest
from pathlib import Path
import shutil
import subprocess


UNISIGN = "https://huggingface.co/ZechengLi19/Uni-Sign/resolve/64a2e24003b6e6bdbafca4f24c5f387921cdc6f0"
MT5 = "https://huggingface.co/google/mt5-base/resolve/2eb15465c5dd7f72a8f7984306ad05ebc3dd1e1f"
CHECKPOINT = "how2sign_pose_only_slt.pth"
CHECKPOINT_HASH = "1bfd5f3312f04e4736f0a52f4ef9535916e6de9676a2a0d00c708748683fb00d"
MT5_WEIGHT_HASH = "180573b534144580f04af026da62bf71bc976ee1b7eb311b8945e2fefde8d614"
MT5_FILES = ("config.json", "generation_config.json", "pytorch_model.bin",
             "special_tokens_map.json", "spiece.model", "tokenizer_config.json")


def download(url, path, expected=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    candidate = path
    if not path.is_file():
        candidate = path.with_suffix(path.suffix + ".part")
        curl = shutil.which("curl.exe") or shutil.which("curl")
        if not curl:
            raise RuntimeError("curl is required to download assets")
        print(f"Downloading {path.name}", flush=True)
        subprocess.run([curl, "-fsSL", "--retry", "3", "--connect-timeout", "20",
                        "-o", str(candidate), url], check=True)
    with candidate.open("rb") as stream:
        digest = file_digest(stream, "sha256").hexdigest()
    if expected is not None and digest != expected:
        raise ValueError(f"Checksum mismatch: {candidate}")
    if candidate != path:
        candidate.replace(path)
    print(f"{digest}  {path.name}", flush=True)
    return digest


def prepare(root):
    root.mkdir(parents=True, exist_ok=True)
    artifacts = [(f"{UNISIGN}/{CHECKPOINT}", root / CHECKPOINT, CHECKPOINT_HASH)]
    artifacts += [(f"{MT5}/{name}", root / "mt5-base" / name,
                   MT5_WEIGHT_HASH if name == "pytorch_model.bin" else None) for name in MT5_FILES]
    if any(not path.exists() for _, path, _ in artifacts) and shutil.disk_usage(root).free < 5 * 1024**3:
        raise OSError("Need 5 GiB free for the pinned model files")
    sums = root / "staged.sha256"
    recorded = {}
    if sums.is_file():
        with sums.open(encoding="utf-8") as stream:
            for line in stream:
                digest, name = line.strip().split("  ", 1)
                recorded[name] = digest
    lines = []
    for url, path, expected in artifacts:
        name = path.relative_to(root).as_posix()
        digest = download(url, path, expected or recorded.get(name))
        lines.append(f"{digest}  {name}\n")
    pending = sums.with_suffix(".tmp")
    pending.write_text("".join(lines), encoding="utf-8")
    pending.replace(sums)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("assets", type=Path)
    prepare(parser.parse_args().assets)
