"""Download the Grid Audio-Visual Speech Corpus from Zenodo (record 3625687).

Dataset: https://zenodo.org/records/3625687
Total: ~16.2 GB across 38 files (audio + alignments + PDF + 33 speaker video archives).

Default destination: /storage/kfir/data/audio_and_video/grid_corpus/
"""

from __future__ import annotations

import argparse
import hashlib
import sys
import time
from pathlib import Path
from urllib.parse import urlparse

import requests
from tqdm import tqdm

ZENODO_RECORD_ID = "3625687"
ZENODO_API = f"https://zenodo.org/api/records/{ZENODO_RECORD_ID}"
DEFAULT_DEST = Path("/storage/kfir/data/audio_and_video/grid_corpus")
CHUNK_SIZE = 1 << 20  # 1 MiB


def fetch_file_manifest() -> list[dict]:
    """Query the Zenodo API for the list of files in the record."""
    resp = requests.get(ZENODO_API, timeout=30)
    resp.raise_for_status()
    data = resp.json()
    files = data.get("files", [])
    manifest = []
    for f in files:
        manifest.append(
            {
                "key": f["key"],
                "size": f["size"],
                "url": f["links"]["self"],
                "checksum": f.get("checksum", ""),  # e.g. "md5:abcd..."
            }
        )
    return manifest


def md5_of_file(path: Path) -> str:
    h = hashlib.md5()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(CHUNK_SIZE), b""):
            h.update(chunk)
    return h.hexdigest()


def verify_checksum(path: Path, expected: str) -> bool:
    """expected is in 'md5:<hex>' form (Zenodo convention)."""
    if not expected or ":" not in expected:
        return True  # nothing to check against
    algo, want = expected.split(":", 1)
    if algo.lower() != "md5":
        print(f"  [warn] unsupported checksum algo {algo}, skipping verify")
        return True
    got = md5_of_file(path)
    return got.lower() == want.lower()


def download_file(url: str, dest: Path, expected_size: int, max_retries: int = 5) -> None:
    """Download with resume support + tqdm progress bar."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")

    for attempt in range(1, max_retries + 1):
        existing = tmp.stat().st_size if tmp.exists() else 0
        if existing >= expected_size and expected_size > 0:
            tmp.rename(dest)
            return

        headers = {"Range": f"bytes={existing}-"} if existing else {}
        try:
            with requests.get(url, headers=headers, stream=True, timeout=60) as r:
                if r.status_code == 416:  # range not satisfiable -> already done
                    tmp.rename(dest)
                    return
                r.raise_for_status()
                mode = "ab" if existing else "wb"
                total = expected_size or int(r.headers.get("content-length", 0)) + existing
                with tmp.open(mode) as fh, tqdm(
                    total=total,
                    initial=existing,
                    unit="B",
                    unit_scale=True,
                    unit_divisor=1024,
                    desc=dest.name,
                    leave=False,
                ) as bar:
                    for chunk in r.iter_content(chunk_size=CHUNK_SIZE):
                        if chunk:
                            fh.write(chunk)
                            bar.update(len(chunk))
            tmp.rename(dest)
            return
        except (requests.RequestException, ConnectionError) as e:
            wait = min(60, 2 ** attempt)
            print(f"  [retry {attempt}/{max_retries}] {dest.name}: {e} -- sleeping {wait}s")
            time.sleep(wait)

    raise RuntimeError(f"failed to download {url} after {max_retries} attempts")


def human(n: int) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if n < 1024:
            return f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} PB"


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dest", type=Path, default=DEFAULT_DEST,
                   help=f"download destination (default: {DEFAULT_DEST})")
    p.add_argument("--skip-verify", action="store_true",
                   help="skip md5 verification of completed downloads")
    p.add_argument("--only", nargs="*", default=None,
                   help="optional whitelist of filenames (e.g. --only audio_25k.zip s1.zip)")
    p.add_argument("--list", action="store_true",
                   help="just print the file manifest and exit")
    args = p.parse_args()

    print(f"Fetching manifest from {ZENODO_API} ...")
    manifest = fetch_file_manifest()
    if args.only:
        wanted = set(args.only)
        manifest = [m for m in manifest if m["key"] in wanted]
        missing = wanted - {m["key"] for m in manifest}
        if missing:
            print(f"  [warn] not in record: {sorted(missing)}")

    total_size = sum(m["size"] for m in manifest)
    print(f"\n{len(manifest)} file(s), total {human(total_size)}:")
    for m in manifest:
        print(f"  {m['key']:<24} {human(m['size']):>10}")

    if args.list:
        return 0

    args.dest.mkdir(parents=True, exist_ok=True)
    print(f"\nDestination: {args.dest.resolve()}\n")

    failures = []
    for i, m in enumerate(manifest, 1):
        dest = args.dest / m["key"]
        print(f"[{i}/{len(manifest)}] {m['key']} ({human(m['size'])})")

        if dest.exists() and dest.stat().st_size == m["size"]:
            if args.skip_verify or verify_checksum(dest, m["checksum"]):
                print("  already present, skipping")
                continue
            print("  checksum mismatch, redownloading")
            dest.unlink()

        try:
            download_file(m["url"], dest, m["size"])
        except Exception as e:
            print(f"  [error] {e}")
            failures.append(m["key"])
            continue

        if not args.skip_verify and not verify_checksum(dest, m["checksum"]):
            print(f"  [error] md5 mismatch for {m['key']}")
            failures.append(m["key"])
        else:
            print("  ok")

    if failures:
        print(f"\nFailed: {failures}")
        return 1
    print("\nAll files downloaded successfully.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
