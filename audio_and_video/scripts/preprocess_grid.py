"""Reorganize extracted Grid corpus into the per-speaker layout the dataset expects,
and resample audio to 16 kHz.

Defaults source from `audio_50k/s<N>_50kHz/<utt>.wav` (Sheffield raw 50 kHz, paper's
actual source). Pass `--source audio_25k` to fall back to Zenodo's 25 kHz endpointed
audio if needed.

Source layout (after extract_grid.py + download_grid_50khz.py):
    <root>/audio_50k/s<N>_50kHz/<utt>.wav   50 kHz, full 3-s recordings (paper's source)
    <root>/audio_25k/s<N>/<utt>.wav         25 kHz, silence-trimmed (Zenodo)  -- alternative
    <root>/alignments/s<N>/<utt>.align
    <root>/s<N>/<utt>.mpg                   already in place

Target layout (what GridAVDataset expects):
    <root>/s<N>/s<N>_16kHz/<utt>.wav    16 kHz mono
    <root>/s<N>/align/<utt>.align       (symlink/copy)
    <root>/s<N>/<utt>.mpg               (untouched)

Run:
    python preprocess_grid.py                         # all speakers, paper-faithful 50k → 16k
    python preprocess_grid.py --source audio_25k      # use Zenodo trimmed 25k instead
    python preprocess_grid.py --speakers 1 2 3
    python preprocess_grid.py --workers 8
"""
from __future__ import annotations

import argparse
import shutil
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import soundfile as sf
from scipy.signal import resample_poly

DEFAULT_ROOT = Path("/storage/kfir/data/audio_and_video/grid_corpus/extracted")
SRC_SR_BY_DIR = {"audio_50k": 50_000, "audio_25k": 25_000}
DST_SR = 16_000


def src_speaker_dir(root: Path, source: str, spk: int) -> Path:
    """Sheffield's 50k tar extracts to `audio_50k/s<N>_50kHz/`; Zenodo's 25k to `audio_25k/s<N>/`."""
    return root / source / (f"s{spk}_50kHz" if source == "audio_50k" else f"s{spk}")


def list_speakers(root: Path, source: str):
    base = root / source
    if source == "audio_50k":
        # dirs look like s1_50kHz, s2_50kHz, ...
        return sorted(int(d.name[1:].replace("_50kHz", "")) for d in base.glob("s*_50kHz"))
    return sorted(int(d.name[1:]) for d in base.glob("s*") if d.name[1:].isdigit())


def resample_one(src: Path, dst: Path, src_sr: int) -> None:
    if dst.exists():
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    audio, sr = sf.read(str(src), dtype="float32", always_2d=False)
    assert sr == src_sr, f"expected {src_sr} Hz, got {sr} for {src}"
    if audio.ndim > 1:
        audio = audio.mean(axis=-1)
    out = resample_poly(audio, DST_SR, src_sr).astype("float32")
    sf.write(str(dst), out, DST_SR, subtype="PCM_16")


def link_align(src: Path, dst: Path) -> None:
    if dst.exists():
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    try:
        dst.symlink_to(src.resolve())
    except OSError:
        shutil.copy2(src, dst)


def process_speaker(args):
    root, source, spk = args
    src_sr = SRC_SR_BY_DIR[source]
    audio_src = src_speaker_dir(root, source, spk)
    audio_dst = root / f"s{spk}" / f"s{spk}_16kHz"
    align_src = root / "alignments" / f"s{spk}"
    align_dst = root / f"s{spk}" / "align"

    n_resampled = 0
    for w in audio_src.glob("*.wav"):
        resample_one(w, audio_dst / w.name, src_sr)
        n_resampled += 1

    n_aligned = 0
    if align_src.is_dir():
        for a in align_src.glob("*.align"):
            link_align(a, align_dst / a.name)
            n_aligned += 1

    return spk, n_resampled, n_aligned


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    p.add_argument("--source", choices=list(SRC_SR_BY_DIR), default="audio_50k",
                   help="audio source dir (default: audio_50k = paper's raw 50 kHz)")
    p.add_argument("--speakers", type=int, nargs="*", help="speaker IDs (default: all found)")
    p.add_argument("--workers", type=int, default=4, help="parallel workers (default 4)")
    args = p.parse_args()

    if not (args.root / args.source).is_dir():
        print(f"{args.source} not found at {args.root}.", file=sys.stderr)
        return 1

    speakers = args.speakers or list_speakers(args.root, args.source)
    print(f"root:    {args.root}")
    print(f"source:  {args.source} (sr={SRC_SR_BY_DIR[args.source]} Hz)")
    print(f"target:  {DST_SR} Hz")
    print(f"speakers ({len(speakers)}): {speakers}")
    print(f"workers: {args.workers}\n")

    t0 = time.time()
    total_audio = 0
    total_align = 0
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(process_speaker, (args.root, args.source, spk)): spk for spk in speakers}
        for fut in as_completed(futs):
            spk, n_a, n_l = fut.result()
            total_audio += n_a
            total_align += n_l
            print(f"  s{spk:<2}  audio={n_a}  align={n_l}")
    elapsed = time.time() - t0
    print(f"\ndone in {elapsed:.1f}s   {total_audio} wavs resampled, {total_align} alignments linked")
    return 0


if __name__ == "__main__":
    sys.exit(main())
