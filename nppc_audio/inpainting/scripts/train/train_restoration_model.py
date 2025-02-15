import hydra
import torch
from omegaconf import DictConfig
import os
from datetime import datetime
from nppc_audio.inpainting.trainer.restoration_trainer import InpaintingTrainer
from nppc_audio.inpainting.scripts.train.config.schema import Config
from dataset.audio_dataset_inpainting import AudioInpaintingDataset


@hydra.main(version_base=None, config_path="config", config_name="config")
def main(cfg: DictConfig):
    # Create unique run name if not specified
    if cfg.inpainting_training_configuration.use_wandb and \
            not cfg.inpainting_training_configuration.wandb_run_name:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        cfg.inpainting_training_configuration.wandb_run_name = f"run_{timestamp}"

    # Create config object
    config = Config(**cfg)

    # Initialize trainer
    trainer = InpaintingTrainer(config.inpainting_training_configuration)

    # Setup validation if configured
    val_dataloader = None
    if config.validation_data_configuration is not None:
        # Create validation dataset
        validation_dataset = AudioInpaintingDataset(config.validation_data_configuration)
        print(f"Validation dataset size: {len(validation_dataset)}")

        # Create validation dataloader
        val_dataloader = torch.utils.data.DataLoader(
            validation_dataset,
            batch_size=config.inpainting_training_configuration.dataloader_configuration.batch_size,
            shuffle=config.inpainting_training_configuration.dataloader_configuration.shuffle,
            num_workers=config.inpainting_training_configuration.dataloader_configuration.num_workers,
            pin_memory=config.inpainting_training_configuration.dataloader_configuration.pin_memory
        )

    # Ensure checkpoint directory exists
    os.makedirs(config.checkpoint_dir, exist_ok=True)

    # Train model
    trainer.train(
        n_steps=config.n_steps,
        n_epochs=config.n_epochs,
        checkpoint_dir=config.checkpoint_dir,
        save_flag=True,
        val_dataloader=val_dataloader
    )


if __name__ == "__main__":
    main()