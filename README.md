# NPPC Audio Inpainting

A PyTorch implementation of Neural Posterior Principal Components (NPPC) for audio inpainting tasks. This repository provides tools for training and validating deep learning models that can restore missing segments in audio spectrograms using uncertainty quantification through principal component analysis.

**Note**: The models in this repository have been trained and validated on the LibriSpeech dataset.

## Overview

This project implements a two-stage approach for audio inpainting:

1. **Restoration Model**: A U-Net based neural network that performs initial audio inpainting by predicting missing spectrogram regions
2. **NPPC Model**: A Neural Posterior Principal Components model that quantifies uncertainty in the restoration predictions and provides multiple plausible completions

### What is NPPC on Audio?

Neural Posterior Principal Components (NPPC) is a method for uncertainty quantification in neural network predictions. In the context of audio inpainting:

- **Audio Inpainting**: The task of filling in missing or corrupted segments of audio signals
- **Uncertainty Quantification**: Understanding how confident the model is about its predictions and exploring multiple plausible solutions
- **Principal Components**: NPPC learns to represent the uncertainty in the prediction space using principal components, allowing for diverse and realistic audio completions

The approach is particularly useful for:
- Speech restoration in noisy environments
- Audio artifact removal
- Creative audio editing with multiple completion options
- Robust audio processing with uncertainty awareness

## Repository Structure

```
generative-audio/
├── nppc_audio/
│   └── inpainting/
│       ├── networks/           # Neural network architectures
│       │   ├── unet.py        # U-Net restoration model
│       │   └── tmp_utils.py   # Utility functions for networks
│       ├── trainer/           # Training modules
│       │   ├── restoration_trainer.py  # Trainer for restoration model
│       │   └── nppc_trainer.py        # Trainer for NPPC model
│       ├── nppc/             # NPPC-specific components
│       │   ├── nppc_model.py # Main NPPC model implementation
│       │   └── pc_wrapper.py # Principal component wrapper
│       ├── validator/        # Model validation tools
│       │   └── validator_restoration_model.py
│       └── scripts/          # Training and validation scripts
│           ├── train/
│           │   ├── train_restoration_model.py
│           │   ├── train_nppc_model.py
│           │   └── config/   # Configuration files
│           └── validator/
├── dataset/                  # Dataset handling
│   ├── audio_dataset_inpainting.py  # Main dataset class
│   ├── audio_dataset.py            # Base audio dataset
│   └── sample_generator.py         # Test sample generation
├── utils.py                 # Utility functions
└── requirements.txt         # Python dependencies
```

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd generative-audio
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. (Optional) For GPU support, ensure you have CUDA installed and compatible PyTorch version.

## Data Preparation

The models expect audio data in WAV format. The current implementation has been trained and tested on the **LibriSpeech dataset**.

### LibriSpeech Dataset

**LibriSpeech** is a corpus of approximately 1000 hours of 16kHz read English speech derived from audiobooks in the LibriVox project. It contains:
- Clean speech recordings
- Multiple speakers
- Diverse content from public domain audiobooks
- High-quality audio suitable for speech processing research

**Download LibriSpeech**:
```bash
# Download LibriSpeech train-clean-100 (100 hours of clean training data)
wget https://www.openslr.org/resources/12/train-clean-100.tar.gz
tar -xzf train-clean-100.tar.gz

# Download LibriSpeech test-clean (test set)
wget https://www.openslr.org/resources/12/test-clean.tar.gz
tar -xzf test-clean.tar.gz
```

### Data Structure

For training, you need:

- **Clean audio files**: High-quality audio samples (LibriSpeech or similar datasets)
- **Directory structure**: Organize audio files in directories that can be recursively searched

Example directory structure (LibriSpeech format):
```
data/
├── LibriSpeech/
│   ├── train-clean-100/
│   │   ├── 19/
│   │   │   ├── 198/
│   │   │   │   ├── 19-198-0000.wav
│   │   │   │   ├── 19-198-0001.wav
│   │   │   │   └── ...
│   │   │   └── ...
│   │   └── ...
│   └── test-clean/
│       ├── 61/
│       │   ├── 70968/
│       │   │   ├── 61-70968-0000.wav
│       │   │   └── ...
│       │   └── ...
│       └── ...
```

### Alternative Datasets

While trained on LibriSpeech, the models can potentially work with other clean speech datasets:
- **VCTK Corpus**: Multi-speaker English corpus
- **Common Voice**: Mozilla's open speech dataset
- **Custom datasets**: Any clean speech recordings in WAV format

**Requirements for custom datasets**:
- 16kHz sample rate (or will be resampled automatically)
- WAV format
- Clean speech without background noise
- Sufficient duration for training (recommended: 10+ hours)

## Training

### 1. Training the Restoration Model

The restoration model is a U-Net that learns to inpaint missing audio segments.

```bash
cd nppc_audio/inpainting/scripts/train
python train_restoration_model.py
```

**Configuration**: Modify the configuration files in `nppc_audio/inpainting/scripts/train/config/` to adjust:
- Model architecture parameters
- Dataset paths
- Training hyperparameters
- Logging settings (Weights & Biases integration available)

**Key parameters to configure**:
- `data_configuration.clean_path`: Path to clean audio files
- `model_configuration`: U-Net architecture settings
- `optimizer_configuration`: Optimizer type and parameters
- `n_epochs` or `n_steps`: Training duration

### 2. Training the NPPC Model

The NPPC model builds upon a pre-trained restoration model to provide uncertainty quantification.

```bash
cd nppc_audio/inpainting/scripts/train
python train_nppc_model.py
```

**Prerequisites**: You need a trained restoration model checkpoint before training the NPPC model.

**Configuration**: Update the NPPC configuration to specify:
- Path to pre-trained restoration model checkpoint
- NPPC-specific hyperparameters
- Principal component dimensions
- Training parameters

## Validation

### Restoration Model Validation

```bash
cd nppc_audio/inpainting/scripts/validator
python validate_restoration_model.py
```

The validator will:
- Load a trained restoration model
- Process test samples
- Generate spectrogram comparisons
- Calculate reconstruction metrics (MSE, MAE)
- Save visualization plots

### NPPC Model Validation

```bash
cd nppc_audio/inpainting/scripts/validator
python validate_nppc_model.py
```

The NPPC validator provides:
- Multiple completion samples for each input
- Uncertainty visualization
- Principal component analysis
- Comparison with baseline restoration

## Model Architecture

### Restoration Model (U-Net)
- **Input**: Masked magnitude spectrogram (log-normalized)
- **Output**: Restored magnitude spectrogram
- **Architecture**: Encoder-decoder with skip connections
- **Loss**: MSE loss on masked regions only

### NPPC Model
- **Input**: Masked spectrogram + restoration model prediction
- **Output**: Principal components representing uncertainty
- **Components**:
  - Pre-trained restoration model (frozen)
  - Principal component wrapper network
  - Uncertainty quantification through PCA

## Configuration

The project uses Hydra for configuration management. Key configuration files:

- `config.yaml`: Main configuration for restoration model training
- `config_nppc.yaml`: Configuration for NPPC model training
- Model-specific configs in respective directories

### Key Configuration Parameters

**Audio Processing**:
- `sample_rate`: Audio sample rate (default: 16000 Hz)
- `stft_configuration`: STFT parameters (n_fft, hop_length, win_length)
- `missing_length_seconds`: Duration of gaps to inpaint (default: 0.128s)

**Training**:
- `batch_size`: Training batch size
- `learning_rate`: Optimizer learning rate
- `n_epochs`: Number of training epochs
- `device`: Training device ('cuda' or 'cpu')

## Logging and Monitoring

The project supports Weights & Biases (wandb) integration for experiment tracking:

- Set `use_wandb: true` in configuration
- Configure `wandb_project_name` and other wandb settings
- Monitor training progress, loss curves, and model artifacts

## Output

### Training Outputs
- **Checkpoints**: Saved model states for resuming training
- **Loss curves**: Training and validation loss visualization
- **Metrics**: JSON files with training statistics

### Validation Outputs
- **Spectrograms**: Visual comparisons of clean, masked, and restored audio
- **Audio samples**: Reconstructed audio files
- **Metrics**: Quantitative evaluation results
- **Uncertainty maps**: NPPC uncertainty visualizations

## Hardware Requirements

- **GPU**: CUDA-compatible GPU recommended for training
- **Memory**: 8GB+ GPU memory for typical batch sizes
- **Storage**: Sufficient space for audio datasets and model checkpoints

## Citation

If you use this code in your research, please cite the relevant papers on NPPC and audio inpainting.

## License

[Add your license information here]

## Contributing

[Add contribution guidelines here]

## Support

For questions and issues, please [add contact information or issue tracker link]. 