# FullSubNet_plus/use_pre_model2.py

import hydra
from omegaconf import DictConfig
import torch
import torchaudio
from pathlib import Path
from torch.utils.data import DataLoader

from config.schema import Config
from dataset import AudioDataset
from utils import prepare_input, model_outputs_to_waveforms
from FullSubNet_plus.speech_enhance.fullsubnet_plus.model.fullsubnet_plus import FullSubNet_Plus


@hydra.main(config_path="config", config_name="config")
def main(cfg: DictConfig):
    # Convert Hydra config to Pydantic model for validation
    config = Config(**cfg)

    # Initialize model
    model = FullSubNet_Plus(**config.model.dict(exclude={'checkpoint_path', 'device'}))
    checkpoint = torch.load(config.model.checkpoint_path, map_location="cpu")
    model.load_state_dict(checkpoint["model"])
    device = torch.device(config.model.device)
    model.to(device)
    model.eval()

    # Setup data paths
    data_dir = Path("data")
    noisy_dir = data_dir / "noisy"
    enhanced_dir = data_dir / "enhanced"
    enhanced_dir.mkdir(parents=True, exist_ok=True)

    # Initialize dataset and dataloader
    dataset = AudioDataset(noisy_path=noisy_dir, sample_rate=config.audio.sr)
    dataloader = DataLoader(
        dataset,
        batch_size=config.audio.batch_size,
        num_workers=config.audio.num_workers,
        shuffle=False
    )

    # Process audio files
    with torch.no_grad():
        for batch_idx, noisy_wavs in enumerate(dataloader):
            noisy_wavs = noisy_wavs.to(device)

            # Prepare input
            noisy_mag, noisy_real, noisy_imag = prepare_input(
                noisy_wavs,
                config.audio.n_fft,
                config.audio.hop_length,
                config.audio.win_length
            )

            # Forward pass
            enhanced_masks = model(noisy_mag, noisy_real, noisy_imag)

            # Convert to waveform
            enhanced_wavs = model_outputs_to_waveforms(
                enhanced_masks,
                noisy_real,
                noisy_imag,
                config.audio.n_fft,
                config.audio.hop_length,
                config.audio.win_length
            )

            # Save enhanced audio
            for i in range(enhanced_wavs.shape[0]):
                save_path = enhanced_dir / f"enhanced_batch{batch_idx}_sample{i}.wav"
                torchaudio.save(
                    save_path,
                    enhanced_wavs[i].cpu().unsqueeze(0),
                    config.audio.sr
                )
                print(f"Saved enhanced audio to {save_path}")


if __name__ == "__main__":
    main()