"""Extract Grid corpus archives into a per-speaker layout.

Usage:
    python extract_grid.py --inspect                # peek at every zip's structure
    python extract_grid.py --inspect audio_25k.zip  # peek at a single zip
    python extract_grid.py                          # extract everything
    python extract_grid.py --only s1.zip s2.zip     # extract a subset
    python extract_grid.py --skip-existing          # skip zips whose first member already exists

Default source:  /storage/kfir/data/audio_and_video/grid_corpus/
Default dest:    /storage/kfir/data/audio_and_video/grid_corpus/extracted/
"""
from __future__ import annotations

import argparse
import sys
import time
import zipfile
from pathlib import Path

DEFAULT_SRC = Path("/storage/kfir/data/audio_and_video/grid_corpus")
DEFAULT_DEST = DEFAULT_SRC / "extracted"


def human(n: float) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if n < 1024:
            return f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} PB"


def inspect(zip_path: Path) -> None:
    with zipfile.ZipFile(zip_path) as z:
        names = z.namelist()
        total_size = sum(i.file_size for i in z.infolist())
    print(f"\n=== {zip_path.name} — {len(names)} entries, {human(total_size)} uncompressed ===")
    for n in names[:6]:
        print(f"  {n}")
    if len(names) > 6:
        print(f"  ... ({len(names) - 6} more)")


def _is_cruft(name: str) -> bool:
    return name.startswith("__MACOSX/") or name.endswith("/.DS_Store") or name.endswith(".DS_Store")


def extract_one(zip_path: Path, dest: Path, skip_existing: bool = False) -> None:
    dest.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as z:
        infos = [i for i in z.infolist() if not _is_cruft(i.filename)]
        total_files = len(infos)
        total_bytes = sum(i.file_size for i in infos)

        if skip_existing and infos:
            first_target = dest / infos[0].filename
            if first_target.exists():
                print(f"[skip] {zip_path.name} (first member already extracted)")
                return

        print(f"[extract] {zip_path.name} → {dest}  ({total_files} files, {human(total_bytes)})")
        t0 = time.time()
        last_report = t0
        done_files = 0
        done_bytes = 0
        for info in infos:
            z.extract(info, dest)
            done_files += 1
            done_bytes += info.file_size
            now = time.time()
            if now - last_report > 5.0:
                pct = 100 * done_bytes / max(1, total_bytes)
                rate = done_bytes / max(0.001, now - t0) / (1024 * 1024)
                print(f"    {done_files}/{total_files}  {human(done_bytes)}/{human(total_bytes)}  {pct:5.1f}%  {rate:.1f} MB/s")
                last_report = now
        elapsed = time.time() - t0
        rate = (done_bytes / 1024 / 1024) / max(0.001, elapsed)
        print(f"  done {zip_path.name} in {elapsed:.1f}s  ({rate:.1f} MB/s avg)")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--src", type=Path, default=DEFAULT_SRC, help=f"source dir (default: {DEFAULT_SRC})")
    p.add_argument("--dest", type=Path, default=DEFAULT_DEST, help=f"destination dir (default: {DEFAULT_DEST})")
    p.add_argument("--inspect", nargs="*", default=None,
                   metavar="ZIP",
                   help="just print zip layouts; no args = all zips, or pass specific filenames")
    p.add_argument("--only", nargs="*", default=None, help="only extract these zip filenames")
    p.add_argument("--skip-existing", action="store_true", help="skip zips already extracted (cheap heuristic: first member's path exists)")
    args = p.parse_args()

    src = args.src.resolve()
    if not src.is_dir():
        print(f"source dir does not exist: {src}", file=sys.stderr)
        return 1

    all_zips = sorted(src.glob("*.zip"))
    if not all_zips:
        print(f"no zips found in {src}", file=sys.stderr)
        return 1

    if args.inspect is not None:
        chosen = [src / n for n in args.inspect] if args.inspect else all_zips
        for z in chosen:
            inspect(z)
        return 0

    if args.only:
        chosen = [src / n for n in args.only]
    else:
        chosen = all_zips

    dest = args.dest.resolve()
    print(f"src:  {src}")
    print(f"dest: {dest}")
    print(f"{len(chosen)} zip(s) to extract\n")
    t0 = time.time()
    for z in chosen:
        if not z.is_file():
            print(f"[warn] missing: {z.name}", file=sys.stderr)
            continue
        extract_one(z, dest, skip_existing=args.skip_existing)
    print(f"\nall done in {time.time() - t0:.1f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
