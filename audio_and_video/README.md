# audio_and_video

NPPC experiments on audio + video data.

## Reference paper

Morrone, Michelsanti, Tan, Jensen — *Audio-Visual Speech Inpainting with Deep Learning*, ICASSP 2021 ([arXiv:2010.04556](https://arxiv.org/abs/2010.04556)). Local copy: [papers/morrone2020_av_speech_inpainting.pdf](papers/morrone2020_av_speech_inpainting.pdf). Reference implementation: https://github.com/dr-pato/audio-visual-speech-inpainting.

Speaker split used in the paper:
- **Train** (25): s1–s20, s22–s25, s28
- **Val** (4): s26, s27, s29, s31
- **Test** (4): s30, s32, s33, s34

## Dataset

[Grid Audio-Visual Speech Corpus](https://zenodo.org/records/3625687) — 34 talkers, 1000 sentences each, 16.2 GB total (audio @ 25 kHz + per-speaker JPG video archives + word alignments).

Data is stored **outside** this repo at `/storage/kfir/data/audio_and_video/grid_corpus/`.

### Download

```bash
# install deps if needed
pip install requests tqdm

# list what will be downloaded (no transfer)
python download_grid_corpus.py --list

# download everything (~16.2 GB) to the default location
python download_grid_corpus.py

# only audio + alignments + docs (skip speaker videos)
python download_grid_corpus.py --only audio_25k.zip alignments.zip jasagrid.pdf

# custom destination
python download_grid_corpus.py --dest /some/other/path
```

The script resumes interrupted downloads and verifies MD5 checksums from the Zenodo API.
