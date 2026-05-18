# Audio + Video NPPC — Implementation Plan

Two phases: **(1)** reproduce Morrone et al. 2021 audio-visual speech inpainting (the deterministic restoration model) **as a PyTorch refactor** of the original TF1 code, then **(2)** plug NPPC on top of the trained restorer for uncertainty quantification.

- Reference paper: [papers/morrone2020_av_speech_inpainting.pdf](papers/morrone2020_av_speech_inpainting.pdf)
- Reference code (TF1, no weights): [audio-visual-speech-inpainting/](audio-visual-speech-inpainting/)
- Our PyTorch refactor lives in: [audio-visual-speech-inpainting-refactor/](audio-visual-speech-inpainting-refactor/)
- Framework decision: **PyTorch** (matches the rest of the workspace incl. [nppc_audio/](../nppc_audio/); avoids EOL TF1 toolchain)

---

## Phase 0 — Prerequisites

- [x] Clone reference repo (Morrone) → [audio-visual-speech-inpainting/](audio-visual-speech-inpainting/)
- [x] Download Dlib 68-landmark predictor → `/storage/kfir/data/audio_and_video/dlib_models/shape_predictor_68_face_landmarks.dat` (96 MB)
- [x] Download Grid corpus (~16 GB) → `/storage/kfir/data/audio_and_video/grid_corpus/` (36 files, see [download.log](download.log))
- [ ] Extract zips into `grid_corpus/extracted/` *(running in background, PID 3296, see [extract.log](extract.log); script at [scripts/extract_grid.py](scripts/extract_grid.py))*
- [x] PESQ binary not needed — kfir_env has the python `pesq` package, sufficient for wb/nb scoring

**Project rule:** all Python runs from `/storage/kfir/kfir_env/` per workspace [README.md](../README.md). The PyTorch refactor stays inside that env; we never install TF.

---

## Phase 1 — PyTorch refactor of the deterministic AV-SI model

Goal: a clean PyTorch reimplementation of the Morrone pipeline that matches the paper's Table 1 numbers (L1, PER, STOI, PESQ) within reasonable tolerance on the GRID test set, across the four variants A / AV / A+MTL / AV+MTL. The cloned TF1 repo is the reference spec only — we do not run it. New code lives in [audio-visual-speech-inpainting-refactor/](audio-visual-speech-inpainting-refactor/).

### 1.0 — TF1 → PyTorch port

Reimplement, do not translate line-by-line. Use the TF1 source as a reference for shapes, hyperparameters, and data formats; everything else (data pipeline, training loop) follows PyTorch idioms.

- [x] Replace TFRecords with a plain PyTorch `Dataset` reading `.wav` + landmark `.npy` directly → [grid_dataset.py](audio-visual-speech-inpainting-refactor/av_si/data/grid_dataset.py)
- [x] Reimplement `StackedBLSTMModel` as `torch.nn.Module` (3-layer BLSTM × 250, FC head, optional CTC head) → [blstm.py](audio-visual-speech-inpainting-refactor/av_si/models/blstm.py)
- [x] Port the audio pipeline (STFT, log-mag, normalization) → [audio.py](audio-visual-speech-inpainting-refactor/av_si/data/audio.py)
- [x] Port masking (binary TF mask, paper's gap distribution) → [masking.py](audio-visual-speech-inpainting-refactor/av_si/data/masking.py)
- [x] Port phase reconstruction (`torchaudio.transforms.GriffinLim` + oracle phase) → [audio.py](audio-visual-speech-inpainting-refactor/av_si/data/audio.py)
- [x] Use `torch.nn.CTCLoss` for the MTL head → [trainer/train.py](audio-visual-speech-inpainting-refactor/av_si/trainer/train.py)
- [x] Standalone phone recognizer (paper footnote 1) → [phone_recognizer.py](audio-visual-speech-inpainting-refactor/av_si/models/phone_recognizer.py)
- [x] Eval metrics (L1, STOI, PESQ, PER, CTC greedy decode) → [eval/metrics.py](audio-visual-speech-inpainting-refactor/av_si/eval/metrics.py)
- [x] Dlib landmark extraction + motion vectors + 25 → 83.33 fps upsampling → [landmarks.py](audio-visual-speech-inpainting-refactor/av_si/data/landmarks.py)
- [x] Phone label preprocessing (GRID `.align` → label sequence) → [phones.py](audio-visual-speech-inpainting-refactor/av_si/data/phones.py)
- [x] All offline pieces smoke-tested with synthetic tensors (forward pass, loss step, 2-epoch training loop)
- [ ] CLI entry points (train.py / eval.py / preprocess.py wrappers) — deferred until we have data on disk

### 1.1 — Data preparation (one-off, depends on extraction finishing)

- [ ] Unpack Grid zips into `grid_corpus/extracted/` *(running, PID 3296)*
- [ ] Reorganize layout into `<root>/s<spk>/{s<spk>_16kHz, s<spk>.landmarks, align}/` to match [grid_dataset.py](audio-visual-speech-inpainting-refactor/av_si/data/grid_dataset.py) — small Python helper
- [ ] Resample audio 25 kHz → 16 kHz across ~33 000 wavs (`scipy.signal.resample` or `librosa.resample`; already in kfir_env)
- [ ] `pip install dlib opencv-python imutils` into kfir_env (only deps still missing)
- [ ] Run landmark extraction across all speakers using [landmarks.py](audio-visual-speech-inpainting-refactor/av_si/data/landmarks.py). Slow (Dlib on CPU); one-off
- [ ] Compute audio mean/std on training-split spectrograms; save to `audio_feat_mean.npy` / `audio_feat_std.npy`
- [ ] Source TIMIT 61-phone dictionary + GRID word→phones lexicon (small text files); run align→`.lbl` conversion via [phones.py](audio-visual-speech-inpainting-refactor/av_si/data/phones.py)
- Speaker split (already encoded in [grid_dataset.py:PAPER_TRAIN/PAPER_VAL/PAPER_TEST](audio-visual-speech-inpainting-refactor/av_si/data/grid_dataset.py)):
  - train: s1–s20, s22–s25, s28 (25 spk)
  - val: s26, s27, s29, s31 (4 spk)
  - test: s30, s32, s33, s34 (4 spk)
  - Note: GRID has 33 talkers (s21 absent in the corpus).
- Masking is sampled on-the-fly per epoch by [masking.py:sample_gap_mask](audio-visual-speech-inpainting-refactor/av_si/data/masking.py) (paper §3.1: cov ~ N(900, 300) ms, 1–8 gaps, ≥ 36 ms each, ≤ 80% total). Fixed-gap test sets via `fixed_gap_mask`.

### 1.2 — Refactor project layout (current state)

```
audio-visual-speech-inpainting-refactor/av_si/
├── data/
│   ├── audio.py           # STFT pipeline (paper §3.2)
│   ├── masking.py         # Random gap masks (paper §3.1)
│   ├── landmarks.py       # Dlib face-landmark extraction + motion vectors
│   ├── phones.py          # GRID align → phone label sequence
│   └── grid_dataset.py    # PyTorch Dataset
├── models/
│   ├── blstm.py           # StackedBLSTM (audio / video / AV; optional CTC head)
│   └── phone_recognizer.py
├── trainer/train.py       # Adam + L1 + optional CTC + early stopping
└── eval/metrics.py        # L1 / STOI / PESQ / PER + CTC greedy decode
```

All paths point at `/storage/kfir/data/audio_and_video/...` from the start. Still missing: a `scripts/` entry-point layer and YAML configs — added once the data-prep helpers settle.

### 1.3 — Train the four variants

Hyperparameters from paper §3.3 (and confirmed in the original [blstm.config](audio-visual-speech-inpainting/scripts/config/blstm.config)): 3-layer BLSTM × 250 units, Adam, lr 0.001, batch 8, early-stop after 5 epochs of no val-loss improvement, λ=0.001 for the CTC head.

- [ ] **A** (audio-only): `training` subcommand, `input='a'`
- [ ] **AV** (audio-visual): `training` subcommand, `input='av'`
- [ ] **A+MTL** (audio + CTC phone-rec head): `training_ctc`, audio-only branch
- [ ] **AV+MTL** (audio-visual + CTC): `training_ctc`, AV branch
- [ ] Train the standalone phone recognizer used for PER (2-layer BLSTM × 250, paper footnote 1) — needed to compute PER on inpainted outputs
- [ ] Save best checkpoints (val-loss); export inference graphs via `inference_model_generation`

### 1.4 — Inference + evaluation

- [ ] Run `inference` (and `inference_siasr` for joint SI+ASR) on the variable-length test set
- [ ] Run inference on each fixed-gap test set (100/200/400/800/1600 ms) for Fig. 2 reproduction
- [ ] Compute L1 (masked region only), STOI, PESQ via `evaluation` subcommand (needs PESQ binary)
- [ ] Compute PER via the trained phone recognizer + beam search (width 20)
- [ ] Compare to paper Table 1:

  | Variant | L1 ↓ | PER ↓ | STOI ↑ | PESQ ↑ |
  |---|---|---|---|---|
  | Unprocessed | 0.838 | 0.508 | 0.480 | 1.634 |
  | A | 0.482 | 0.228 | 0.794 | 2.458 |
  | AV | 0.452 | 0.151 | 0.811 | 2.506 |
  | A + MTL | 0.476 | 0.214 | 0.799 | 2.466 |
  | AV + MTL | **0.445** | **0.137** | **0.817** | **2.525** |

- [ ] Reproduce Fig. 2 (per-gap-size curves); the audio-vs-visual gap is supposed to widen sharply for ≥800 ms gaps

### 1.5 — Acceptance

Numbers within ~5% relative error of Table 1 (allowing for non-determinism / minor data-pipeline differences). If we miss meaningfully, debug before moving on — Phase 2 stacks on top of Phase 1.

---

## Phase 2 — NPPC on top of the trained restorer

Once Phase 1 is solid: freeze the restoration model and train NPPC heads to learn the principal directions of the posterior over the inpainted spectrogram, mirroring the existing audio-only pattern at [nppc_audio/inpainting/](../nppc_audio/inpainting/).

- [ ] Decide modality for the NPPC backbone: AV+MTL (best performer) or AV (cleaner gradient path; no auxiliary CTC loss to interfere)
- [ ] Decide framework for NPPC head: stay in TF1 (tight coupling, harder dev) **or** export the frozen restorer and train NPPC in PyTorch using the existing `nppc_audio` patterns. Lean towards the second.
- [ ] Adapt [nppc_audio/inpainting/nppc/](../nppc_audio/inpainting/nppc/) (PCWrapper + NPPCModel) to take the AV restorer's prediction + masked input + visual features
- [ ] Train NPPC head; validate principal components are diverse and align with intelligibility / PER variation
- [ ] Generate qualitative samples (multiple plausible completions per masked utterance, especially in the 800–1600 ms regime where audio context is insufficient)

Out of scope for this plan: deciding final NPPC architecture details, evaluation metric set for uncertainty quality (we'll spec this once Phase 1 is reproduced and we know the restorer's residual structure).

---

## Open questions

1. ~~TF1 strategy~~ — **resolved**: clean PyTorch refactor, kfir_env only, no TF.
2. Do we need the full 25-speaker train set or can we get away with a subset for a faster iteration loop on data-pipeline correctness before committing to the full training run?
3. Does the upstream code's beam-search PER evaluation match the paper's reported PER protocol exactly? The footnote is terse — we'll need to verify when implementing PER.
4. **Loss: L1 vs MSE.** Paper §2.1 says MSE; the upstream TF code minimizes mean L1 over the full normalized spec. Our refactor follows the code (L1) since that's what produced the reported Table 1 numbers. Revisit if reproduction diverges.

## Status snapshot

- ✅ **Code** — all PyTorch modules written + smoke-tested with synthetic tensors
- 🔄 **Data extraction** running (PID 3296)
- ⏳ **Preprocessing** (resample, landmarks, audio stats, phone labels) — blocked on extraction + 3 pip installs
- ⏳ **Training** — blocked on preprocessing
- ⏳ **Evaluation vs paper Table 1 / Fig. 2** — blocked on training
