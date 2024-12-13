import hydra
from omegaconf import DictConfig
from dataset.sample_generator import TestSampleGenerator, TestSampleGeneratorConfig


@hydra.main(version_base=None, config_path="config", config_name="config")
def main(cfg: DictConfig) -> None:
    # Convert Hydra config to Pydantic model for validation
    config = TestSampleGeneratorConfig(**cfg)

    print("Configuration:")
    print(f"Clean path: {config.clean_path}")
    print(f"Noisy path: {config.noisy_path}")
    print(f"Output directory: {config.output_dir}")
    print(f"SNR: {config.snr}")
    print(f"Number of samples: {config.num_samples}")
    # # Convert Hydra config to pydantic config
    # config = TestSampleGeneratorConfig(
    #     clean_path=cfg.test_sample_generator.clean_path,
    #     noisy_path=cfg.test_sample_generator.noisy_path,
    #     output_dir=cfg.test_sample_generator.output_dir,
    #     snr=cfg.test_sample_generator.snr,
    #     num_samples=cfg.test_sample_generator.num_samples,
    #     sample_length_seconds=cfg.test_sample_generator.sample_length_seconds,
    #     sample_rate=cfg.test_sample_generator.sample_rate,
    #     target_dB_FS=cfg.test_sample_generator.target_dB_FS,
    #     silence_length=cfg.test_sample_generator.silence_length
    # )

    # Create generator and generate samples
    generator = TestSampleGenerator(config)
    generator.generate_samples()


if __name__ == "__main__":
    main()
