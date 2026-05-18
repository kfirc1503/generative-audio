"""Download raw 50 kHz GRID audio from Sheffield (the paper's actual source).

Zenodo's audio_25k.zip ships *endpointed* audio (silence trimmed) — different from
what Morrone et al. used. Sheffield's per-speaker `s<N>_50kHz.tar` files are the
raw 50 kHz wavs the paper authors started from.

Downloads ~10 GB across 33 speakers (s21 absent in GRID), then extracts each
tar into <dest>/s<N>_50kHz/<utt>.wav.
"""
from __future__ import annotations

import argparse
import sys
import tarfile
import time
from pathlib import Path
from urllib.request import Request, urlopen

import requests
from tqdm import tqdm


BASE = "https://spandh.dcs.shef.ac.uk/gridcorpus"
PAPER_SPEAKERS = [s for s in range(1, 35) if s != 21]   # 33 speakers
DEFAULT_DEST = Path("/storage/kfir/data/audio_and_video/grid_corpus/extracted/audio_50k")
DEFAULT_TAR_DIR = Path("/storage/kfir/data/audio_and_video/grid_corpus/audio_50k_tars")
CHUNK = 1 << 20  # 1 MiB


def download_one(url: str, dest: Path, expected_size: int = 0, retries: int = 5) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    for attempt in range(1, retries + 1):
        existing = tmp.stat().st_size if tmp.exists() else 0
        if expected_size and existing >= expected_size:
            tmp.rename(dest)
            return
        headers = {"Range": f"bytes={existing}-"} if existing else {}
        try:
            with requests.get(url, headers=headers, stream=True, timeout=60) as r:
                if r.status_code == 416:
                    tmp.rename(dest)
                    return
                r.raise_for_status()
                total = expected_size or int(r.headers.get("content-length", 0)) + existing
                mode = "ab" if existing else "wb"
                with tmp.open(mode) as fh, tqdm(total=total, initial=existing, unit="B",
                                                unit_scale=True, unit_divisor=1024,
                                                desc=dest.name, leave=False) as bar:
                    for chunk in r.iter_content(chunk_size=CHUNK):
                        if chunk:
                            fh.write(chunk)
                            bar.update(len(chunk))
            tmp.rename(dest)
            return
        except (requests.RequestException, ConnectionError) as e:
            wait = min(60, 2 ** attempt)
            print(f"  [retry {attempt}/{retries}] {dest.name}: {e}  sleeping {wait}s")
            time.sleep(wait)
    raise RuntimeError(f"failed to download {url}")


def expected_size(url: str) -> int:
    try:
        r = requests.head(url, allow_redirects=True, timeout=30)
        r.raise_for_status()
        return int(r.headers.get("content-length", 0))
    except Exception:
        return 0


def extract_tar(tar_path: Path, dest_root: Path) -> int:
    """Extract `s<N>.tar` into `<dest_root>/s<N>_50kHz/`. Returns number of wavs extracted."""
    spk_id = tar_path.stem.replace("_50kHz", "")
    spk_dir = dest_root / f"{spk_id}_50kHz"
    spk_dir.mkdir(parents=True, exist_ok=True)
    n = 0
    with tarfile.open(tar_path) as tf:
        for member in tf:
            if not member.isfile():
                continue
            # tar may contain s<N>/<utt>.wav; flatten to <utt>.wav under spk_dir
            name = Path(member.name).name
            if not name.endswith(".wav"):
                continue
            target = spk_dir / name
            if target.exists():
                n += 1
                continue
            f = tf.extractfile(member)
            if f is not None:
                target.write_bytes(f.read())
                n += 1
    return n


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--speakers", type=int, nargs="*", default=PAPER_SPEAKERS,
                   help="speaker IDs to fetch (default: 33 paper speakers)")
    p.add_argument("--tar-dir", type=Path, default=DEFAULT_TAR_DIR,
                   help="where to keep tars (default: %(default)s)")
    p.add_argument("--dest", type=Path, default=DEFAULT_DEST,
                   help="where to extract wavs (default: %(default)s)")
    p.add_argument("--keep-tars", action="store_true",
                   help="keep .tar files after extracting (default: delete to save disk)")
    p.add_argument("--list", action="store_true", help="just list URLs and sizes, don't download")
    args = p.parse_args()

    args.tar_dir.mkdir(parents=True, exist_ok=True)
    args.dest.mkdir(parents=True, exist_ok=True)

    print(f"speakers ({len(args.speakers)}): {args.speakers}")
    print(f"tar dir:   {args.tar_dir}")
    print(f"extracted: {args.dest}\n")

    failures = []
    t0 = time.time()
    for i, spk in enumerate(args.speakers, 1):
        url = f"{BASE}/s{spk}/audio/s{spk}_50kHz.tar"
        size = expected_size(url)
        print(f"[{i}/{len(args.speakers)}] s{spk}  {size/1024/1024:.0f} MB  {url}")
        if args.list:
            continue
        tar_path = args.tar_dir / f"s{spk}_50kHz.tar"
        if not (tar_path.exists() and tar_path.stat().st_size == size):
            try:
                download_one(url, tar_path, size)
            except Exception as e:
                print(f"  [error] {e}")
                failures.append(spk)
                continue
        n = extract_tar(tar_path, args.dest)
        print(f"  extracted {n} wavs → {args.dest}/s{spk}_50kHz/")
        if not args.keep_tars:
            tar_path.unlink(missing_ok=True)

    elapsed = time.time() - t0
    print(f"\ndone in {elapsed/60:.1f} min  failures={failures}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
