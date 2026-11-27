"""Select intact official How2Sign test recordings via ZIP byte ranges and prepare aligned pose windows."""

import argparse
import csv
import hashlib
import io
import json
from pathlib import Path
import re
import shutil
import subprocess
import sys
from urllib.request import urlopen
from zipfile import ZipFile
import zlib

from range_zip import RangedFile

DATAVERSE = "https://dataverse.csuc.cat/api/access/datafile"
VIDEO_BYTES = 24_941_280_047
ANNOTATION_MD5 = "e231fe6bc92e6369bc8552e737dfcce3"


def load_annotations(directory):
    path = directory / "how2sign_realigned_test.csv"
    if path.exists():
        raw = path.read_bytes()
    else:
        with urlopen(f"{DATAVERSE}/45362", timeout=25) as source:
            raw = source.read(1024**2)
    if hashlib.md5(raw, usedforsecurity=False).hexdigest() != ANNOTATION_MD5:
        raise ValueError("Unexpected official test annotation checksum")
    path.write_bytes(raw)
    text = raw.decode("utf-8-sig")
    fields = next(csv.reader(io.StringIO(text), delimiter="\t"))
    delimiter = "\t" if "VIDEO_NAME" in fields else ","
    groups = {}
    for row in csv.DictReader(io.StringIO(text), delimiter=delimiter):
        groups.setdefault(row["VIDEO_NAME"], []).append(row)
    return groups


def select_window(rows):
    bounds = [(float(row["START_REALIGNED"]), float(row["END_REALIGNED"]), row) for row in rows
              if row["START_REALIGNED"] and row["END_REALIGNED"] and row["SENTENCE"].strip()]
    bounds.sort(key=lambda item: (item[0], item[1], item[2]["SENTENCE_ID"]))
    for index in range(len(bounds) - 2):
        triple = bounds[index:index + 3]
        start, end = triple[0][0], triple[-1][1]
        if (start < 0 or not 8 <= end - start <= 40
                or any(left >= right for left, right, _ in triple)
                or any(left[1] > right[0] for left, right in zip(triple, triple[1:]))
                or (index > 0 and bounds[index - 1][1] > start)
                or (index + 3 < len(bounds) and bounds[index + 3][0] < end)):
            continue
        lower = max(start - 0.5, bounds[index - 1][1] if index else 0)
        upper = min(end + 0.8, bounds[index + 3][0] if index + 3 < len(bounds) else end)
        return lower * 1000, upper * 1000, [
            {"item": row["SENTENCE_ID"], "start_ms": first * 1000,
             "end_ms": last * 1000, "reference": row["SENTENCE"]}
            for first, last, row in triple
        ]
    return None


def fetch_source(archive, member, path):
    if path.exists():
        with path.open("rb") as source:
            crc = 0
            for chunk in iter(lambda: source.read(1024**2), b""):
                crc = zlib.crc32(chunk, crc)
        if path.stat().st_size == member.file_size and crc == member.CRC:
            return
    temp = path.with_suffix(".part")
    with archive.open(member) as source, temp.open("wb") as out:
        shutil.copyfileobj(source, out, 1024**2)
    temp.replace(path)


def prepare(directory, upstream, count):
    directory.mkdir(parents=True, exist_ok=True)
    if shutil.disk_usage(directory).free < 1024**3:
        raise OSError("Need at least 1 GiB free for selected continuous recordings")
    groups = load_annotations(directory)
    rows, seen = [], set()
    poses = directory / "prepared"
    poses.mkdir(exist_ok=True)
    with ZipFile(RangedFile([(f"{DATAVERSE}/13763", VIDEO_BYTES)], 8 * 1024**2)) as archive:
        members = sorted((m for m in archive.infolist() if m.filename.endswith(".mp4")),
                         key=lambda m: (m.compress_size, m.filename))
        for member in members:
            video_id = Path(member.filename).stem
            if not re.fullmatch(r"[\w-]+", video_id) or video_id not in groups:
                continue
            group = groups[video_id][0]["VIDEO_ID"]
            window = select_window(groups[video_id])
            if window is None or group in seen or member.file_size > 256 * 1024**2:
                continue
            lo, hi, segments = window
            source = directory / f"{video_id}.mp4"
            print(f"Official test member: {member.filename} ({member.file_size} bytes)", flush=True)
            fetch_source(archive, member, source)
            pose = poses / f"{video_id}.npz"
            command = [sys.executable, str(Path(__file__).with_name("pose_data.py")), "video", str(upstream),
                       str(source), str(pose), "--start-ms", str(lo), "--end-ms", str(hi), "--verify-cadence"]
            process = subprocess.run(command, check=True, stdout=subprocess.PIPE, text=True)
            timing = json.loads(process.stdout.splitlines()[-1])
            offset = timing["source_start_ms"]
            for segment in segments:
                segment["start_ms"] -= offset
                segment["end_ms"] -= offset
            rows.append({"id": video_id, "group": group, "pose": pose.name, "segments": segments,
                         "source": {"dataverse_id": 13763, "member": member.filename,
                                    "csv_md5": ANNOTATION_MD5, **timing}})
            seen.add(group)
            if len(rows) == count:
                break
    if len(rows) != count:
        raise ValueError(f"Only {len(rows)}/{count} official original recordings qualified")
    manifest = poses / "continuous.jsonl"
    manifest.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    print(manifest, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("upstream", type=Path)
    parser.add_argument("--count", type=int, default=2, choices=range(1, 5))
    args = parser.parse_args()
    prepare(args.directory.resolve(), args.upstream.resolve(), args.count)
