"""Train the StackedBLSTM speech inpainter on GRID.

Defaults match Morrone §3.3. Audio mean/std required:
    python compute_audio_stats.py    # writes audio_feat_mean.npy / audio_feat_std.npy

Usage:
    python train_blstm.py --mode a                                           # audio only, fast
    python train_blstm.py --mode av                                          # audio+video
    python train_blstm.py --mode av --ctc                                    # AV + MTL
    python train_blstm.py --mode a --train-speakers 1 2 --val-speakers 26    # quick dev loop
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "audio-visual-speech-inpainting-refactor"))
from av_si.data.grid_dataset import GridAVDataset, PAPER_TRAIN, PAPER_VAL
from av_si.models.blstm import StackedBLSTM
from av_si.trainer.train import TrainConfig, train


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mode", choices=("a", "v", "av"), default="av")
    p.add_argument("--ctc", action="store_true", help="enable MTL CTC head (paper +MTL variants)")
    p.add_argument("--n-phones", type=int, default=34, help="phones + 1 blank for CTC")
    p.add_argument("--root", type=Path, default=Path("/storage/kfir/data/audio_and_video/grid_corpus/extracted"))
    p.add_argument("--audio-mean", type=Path, default=None)
    p.add_argument("--audio-std", type=Path, default=None)
    p.add_argument("--train-speakers", type=int, nargs="*", default=PAPER_TRAIN)
    p.add_argument("--val-speakers", type=int, nargs="*", default=PAPER_VAL)
    p.add_argument("--out-dir", type=Path, default=Path("runs/blstm_av"))
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--max-epochs", type=int, default=50)
    p.add_argument("--patience", type=int, default=5)
    args = p.parse_args()

    audio_mean = args.audio_mean or (args.root / "audio_feat_mean.npy")
    audio_std = args.audio_std or (args.root / "audio_feat_std.npy")
    if not (audio_mean.is_file() and audio_std.is_file()):
        print(f"missing audio stats: {audio_mean} / {audio_std}\n  run compute_audio_stats.py first", file=sys.stderr)
        return 1

    load_video = args.mode in ("v", "av")

    common = dict(
        root=str(args.root),
        audio_mean_path=str(audio_mean),
        audio_std_path=str(audio_std),
        load_video=load_video,
        load_phones=args.ctc,
    )
    train_ds = GridAVDataset(speakers=args.train_speakers, **common)
    val_ds = GridAVDataset(speakers=args.val_speakers, **common)

    print(f"mode={args.mode} ctc={args.ctc}  train={len(train_ds)} val={len(val_ds)}")

    model = StackedBLSTM(input_mode=args.mode, n_phones=args.n_phones if args.ctc else None)
    cfg = TrainConfig(
        lr=args.lr,
        batch_size=args.batch_size,
        max_epochs=args.max_epochs,
        early_stop_patience=args.patience,
        out_dir=str(args.out_dir),
    )
    train(model, train_ds, val_ds, cfg)
    return 0


if __name__ == "__main__":
    sys.exit(main())
