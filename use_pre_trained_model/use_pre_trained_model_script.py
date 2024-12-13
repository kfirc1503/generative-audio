# FullSubNet_plus/use_pre_model2.py

import hydra
from omegaconf import DictConfig
import torch
import torchaudio
from pathlib import Path
from torch.utils.data import DataLoader

from config.schema import Config
from dataset.dataset import AudioDataset
from utils import model_outputs_to_waveforms, load_pretrained_model, prepare_input_from_waveform


@hydra.main(config_path="config", config_name="config")
def main(cfg: DictConfig):
    # Convert Hydra config to Pydantic model for validation
    config = Config(**cfg)

    # Initialize model
    model_path:str = config.pre_trained_model.checkpoint_path
    sub_net_plus_config = config.pre_trained_model.model
    model = load_pretrained_model(model_path, sub_net_plus_config)

    device = torch.device(config.pre_trained_model.device)
    model.to(device)
    model.eval()

    # Setup data paths
    enhanced_dir = Path(config.pre_trained_data_model.enhanced_dir_path)
    enhanced_dir.mkdir(parents=True, exist_ok=True)

    # Initialize dataset and dataloader
    dataset = AudioDataset(config.pre_trained_data_model.dataset)
    dataloader = DataLoader(
        dataset,
        batch_size=config.audio.batch_size,
        num_workers=config.audio.num_workers,
        shuffle=False
    )

    # Process audio files
    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            clean_waveforms, noisy_waveforms = batch
            clean_waveforms = clean_waveforms.to(device)
            noisy_waveforms = noisy_waveforms.to(device)
            # Prepare input
            noisy_mag, noisy_real, noisy_imag = prepare_input_from_waveform(noisy_waveforms)
            model = model.to(device)
            # Forward pass
            enhanced_masks = model(noisy_mag, noisy_real, noisy_imag)

            # Convert to waveform
            orig_len = config.pre_trained_data_model.dataset.sub_sample_length
            enhanced_wavs = model_outputs_to_waveforms(
                enhanced_masks,
                noisy_real,
                noisy_imag,
                orig_length= orig_len
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