import hydra
import torch
from omegaconf import DictConfig

from nppc_audio.inpainting.trainer.restoration_trainer import InpaintingTrainer
from nppc_audio.inpainting.trainer.nppc_trainer import NPPCAudioInpaintingTrainer
from nppc_audio.inpainting.scripts.train.config.schema_nppc import Config
from dataset.audio_dataset_inpainting import AudioInpaintingDataset,AudioInpaintingConfig

@hydra.main(version_base=None, config_path="config", config_name="config_nppc")
def main(cfg: DictConfig):
    config = Config(**cfg)
    trainer = NPPCAudioInpaintingTrainer(config.inpainting_nppc_training_configuration)
    val_dataloader = None
    if config.validation_data_configuration is not None:
        # Create dataset
        validation_dataset = AudioInpaintingDataset(config.validation_data_configuration)

        # Create dataloader for Validation
        val_dataloader = torch.utils.data.DataLoader(
            validation_dataset,
            batch_size=config.inpainting_nppc_training_configuration.dataloader_configuration.batch_size,  # Adjust based on your GPU memory
            shuffle=config.inpainting_nppc_training_configuration.dataloader_configuration.shuffle,
            num_workers=config.inpainting_nppc_training_configuration.dataloader_configuration.num_workers,
            pin_memory=config.inpainting_nppc_training_configuration.dataloader_configuration.pin_memory
        )

    trainer.train(n_steps= config.n_steps, n_epochs= config.n_epochs,checkpoint_dir = config.checkpoint_dir,save_flag=True ,val_dataloader=val_dataloader)

if __name__ == "__main__":
    main()