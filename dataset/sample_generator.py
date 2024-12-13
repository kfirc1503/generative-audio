import hydra
from omegaconf import DictConfig
import torch
import torchaudio
import random
from pathlib import Path
import soundfile as sf
from typing import Union, Tuple
import pydantic
from dataset import AudioDataSetConfig, AudioDataset
import os


class TestSampleGeneratorConfig(pydantic.BaseModel):
    """Configuration for generating test samples"""
    clean_path: Union[str, Path]
    noisy_path: Union[str, Path]
    output_dir: Union[str, Path]
    sample_rate: int = 16000
    snr: int = 10
    num_samples: int = 100
    sample_length_seconds: float = 3.0
    target_dB_FS: float = -25.0
    silence_length: float = 0.2


class TestSampleGenerator:
    def __init__(self, config: TestSampleGeneratorConfig):
        self.config = config

        # Create dataset config
        dataset_config = AudioDataSetConfig(
            clean_path=config.clean_path,
            noisy_path=config.noisy_path,
            sample_rate=config.sample_rate,
            snr_range=(config.snr, config.snr),
            sub_sample_length_seconds=config.sample_length_seconds,
            target_dB_FS=config.target_dB_FS,
            silence_length=config.silence_length
        )

        # Initialize dataset
        self.dataset = AudioDataset(dataset_config)

        # Create output directories
        self.output_dir = Path(config.output_dir)
        self.clean_dir = self.output_dir / f"clean_snr_{config.snr}"
        self.noisy_dir = self.output_dir / f"noisy_snr_{config.snr}"

        self.clean_dir.mkdir(parents=True, exist_ok=True)
        self.noisy_dir.mkdir(parents=True, exist_ok=True)

    def generate_samples(self):
        """Generate and save test samples"""
        print(f"Generating {self.config.num_samples} samples at SNR={self.config.snr}dB")

        for i in range(self.config.num_samples):
            # Get a random sample from dataset
            noisy, clean = self.dataset[random.randint(0, len(self.dataset) - 1)]

            # Save files
            clean_path = self.clean_dir / f"sample_{i:04d}_clean.wav"
            noisy_path = self.noisy_dir / f"sample_{i:04d}_noisy.wav"

            sf.write(clean_path, clean.numpy(), self.config.sample_rate)
            sf.write(noisy_path, noisy.numpy(), self.config.sample_rate)

            if (i + 1) % 10 == 0:
                print(f"Generated {i + 1}/{self.config.num_samples} samples")


# @hydra.main(version_base=None, config_path="configs", config_name="test_sample_generator")
# def main(cfg: DictConfig) -> None:
#     print("Configuration:")
#     print(f"Clean path: {cfg.test_sample_generator.clean_path}")
#     print(f"Noisy path: {cfg.test_sample_generator.noisy_path}")
#     print(f"Output directory: {cfg.test_sample_generator.output_dir}")
#     print(f"SNR: {cfg.test_sample_generator.snr}")
#     print(f"Number of samples: {cfg.test_sample_generator.num_samples}")
#
#     # Convert Hydra config to pydantic config
#     config = TestSampleGeneratorConfig(
#         clean_path=cfg.test_sample_generator.clean_path,
#         noisy_path=cfg.test_sample_generator.noisy_path,
#         output_dir=cfg.test_sample_generator.output_dir,
#         snr=cfg.test_sample_generator.snr,
#         num_samples=cfg.test_sample_generator.num_samples,
#         sample_length_seconds=cfg.test_sample_generator.sample_length_seconds,
#         sample_rate=cfg.test_sample_generator.sample_rate,
#         target_dB_FS=cfg.test_sample_generator.target_dB_FS,
#         silence_length=cfg.test_sample_generator.silence_length
#     )
#
#     # Create generator and generate samples
#     generator = TestSampleGenerator(config)
#     generator.generate_samples()
#
#
# if __name__ == "__main__":
#     main()