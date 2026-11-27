"""Verify staged model bytes and publish the S3 completion marker only after verification."""

import argparse
import hashlib
import json
from pathlib import Path
import subprocess

from prepare_weights import CHECKPOINT, CHECKPOINT_HASH, MT5_FILES, MT5_WEIGHT_HASH


def fingerprints(path, chunk_size):
    sha = hashlib.sha256()
    plain = hashlib.md5(usedforsecurity=False)
    parts = []
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(chunk_size), b""):
            sha.update(chunk)
            plain.update(chunk)
            parts.append(hashlib.md5(chunk, usedforsecurity=False).digest())
    composite = hashlib.md5(b"".join(parts), usedforsecurity=False).hexdigest()
    return sha.hexdigest(), {plain.hexdigest(), f"{composite}-{len(parts)}"}


def verify(args):
    checksums = dict((name, digest) for digest, name in
                     (line.split("  ", 1) for line in args.manifest.read_text(encoding="utf-8").splitlines()))
    expected = {CHECKPOINT, *(f"mt5-base/{name}" for name in MT5_FILES)}
    if (set(checksums) != expected or checksums[CHECKPOINT] != CHECKPOINT_HASH
            or checksums["mt5-base/pytorch_model.bin"] != MT5_WEIGHT_HASH):
        raise ValueError("Local manifest does not match the pinned model files")
    command = [args.aws, "--profile", args.profile, "--region", args.region,
               "--endpoint-url", f"https://s3api-{args.region.lower()}.runpod.io/", "--cli-read-timeout", "180",
               "--output", "json", "--no-cli-pager"]
    prefix = "signgpt-validation/assets/"
    for name, required in checksums.items():
        local = args.assets / name
        digest, etags = fingerprints(local, 5 * 1024**2)
        if digest != required:
            raise ValueError(f"Local SHA-256 mismatch: {name}")
        key = prefix + name
        info = json.loads(subprocess.check_output(
            [*command, "s3api", "head-object", "--bucket", args.volume, "--key", key]))
        if info["ContentLength"] != local.stat().st_size:
            raise ValueError(f"Remote size mismatch: {name}")
        if info["ETag"].strip('"') not in etags:
            print(f"ETag does not identify the uploaded bytes. Verifying remote SHA-256 for {name}.", flush=True)
            with subprocess.Popen([*command, "s3", "cp", f"s3://{args.volume}/{key}", "-",
                                   "--only-show-errors"], stdout=subprocess.PIPE) as download:
                digest = hashlib.file_digest(download.stdout, "sha256").hexdigest()
                if download.wait() or digest != required:
                    raise ValueError(f"Remote SHA-256 mismatch: {name}")
        print(f"Verified {name}", flush=True)
    subprocess.run([*command, "s3", "cp", str(args.manifest), f"s3://{args.volume}/{prefix}staged.sha256",
                    "--only-show-errors"], check=True)
    marker = subprocess.check_output([*command, "s3", "cp",
                                      f"s3://{args.volume}/{prefix}staged.sha256", "-", "--only-show-errors"])
    if marker != args.manifest.read_bytes():
        raise ValueError("Published marker does not match the verified manifest")
    print("Verified model upload and completion marker.", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--aws", required=True)
    parser.add_argument("--volume", required=True)
    parser.add_argument("--region", default="EUR-IS-1")
    parser.add_argument("--profile", default="signgpt-volume")
    parser.add_argument("assets", type=Path)
    parser.add_argument("manifest", type=Path)
    verify(parser.parse_args())
