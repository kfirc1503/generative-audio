import torch
import torch.nn as nn
import torch.optim as optim
import pydantic
import os
from datetime import datetime
import json
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
import wandb
from typing import Optional, List
from nppc_audio.inpainting.networks.unet import UNetConfig, RestorationWrapper, UNet
from dataset.audio_dataset_inpainting import AudioInpaintingConfig, AudioInpaintingDataset
from use_pre_trained_model.model_validator.config.schema import DataLoaderConfig
import utils
from nppc.auxil import LoopLoader


class OptimizerConfig(pydantic.BaseModel):
    type: str
    args: dict


class InpaintingTrainerConfig(pydantic.BaseModel):
    """Configuration for Inpainting trainer"""
    model_configuration: UNetConfig
    data_configuration: AudioInpaintingConfig
    dataloader_configuration: DataLoaderConfig
    optimizer_configuration: OptimizerConfig
    device: str = "cuda"
    # save_interval: int = 10
    # save_best_model: bool = True

    # Wandb configuration
    use_wandb: bool = False
    wandb_project_name: Optional[str] = "generative-audio"
    wandb_run_name: Optional[str] = None
    wandb_tags: Optional[List[str]] = None

class InpaintingTrainer(nn.Module):
    def __init__(self, config: InpaintingTrainerConfig):
        super().__init__()
        self.config = config

        # Initialize wandb if enabled
        if config.use_wandb:
            wandb.init(
                project=config.wandb_project_name,
                name=config.wandb_run_name,
                config=config.model_dump(),
                tags=config.wandb_tags
            )

        # Initialize model and move to device
        base_network = UNet(self.config.model_configuration)
        self.model = RestorationWrapper(base_network)
        self.device = config.device
        if config.device == 'cuda':
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

        # Create optimizer
        self.optimizer = getattr(optim, config.optimizer_configuration.type)(
            self.model.parameters(),
            **config.optimizer_configuration.args
        )
        dataset = AudioInpaintingDataset(config.data_configuration)

        print(f"Total sample pairs in dataset: {len(dataset)}")

        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=config.dataloader_configuration.batch_size,  # Adjust based on your GPU memory
            shuffle=config.dataloader_configuration.shuffle,
            num_workers=config.dataloader_configuration.num_workers,
            pin_memory=config.dataloader_configuration.pin_memory
        )
        self.dataloader = dataloader

        # Initialize step counter
        self.step = 0

    def train(self, n_steps=None, n_epochs=None, checkpoint_dir="checkpoints",
              save_flag=False, val_dataloader=None):
        """Main training loop"""
        assert n_steps is not None or n_epochs is not None, "Must specify either n_steps or n_epochs"

        # Create checkpoint directory
        os.makedirs(checkpoint_dir, exist_ok=True)

        # Initialize loss tracking
        loss_history: List[float] = []
        val_loss_history: List[float] = []
        best_val_loss = float('inf')

        # Create loop loader
        loop_loader = LoopLoader(
            dataloader=self.dataloader,
            n_steps=n_steps,
            n_epochs=n_epochs
        )

        # Training loop with progress bar
        pbar = tqdm(loop_loader, total=len(loop_loader))
        for batch in pbar:
            # Move batch to device
            masked_spec, mask, clean_spec = [x.to(self.device) for x in batch]
            batch = (masked_spec, mask, clean_spec)

            # Forward and backward pass
            loss, log_dict = self.base_step(batch)
            self.optimizer.zero_grad()
            loss.backward()

            # Apply gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=5)
            self.optimizer.step()

            # Store loss
            loss_history.append(loss.item())

            # Update progress bar
            pbar.set_description(f'Loss: {loss.item():.4f}')
            self.step += 1

        # Final validation
        if val_dataloader:
            val_loss = self.validate(val_dataloader)
            val_loss_history.append(val_loss)
            print(f"Final Validation Loss: {val_loss:.4f}")

        # Plot loss curve
        fig = self.plot_loss_curve(loss_history, val_loss_history)

        if save_flag:
            # Save final checkpoint
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            final_checkpoint_path = os.path.join(
                checkpoint_dir,
                f"checkpoint_final_{timestamp}.pt"
            )
            self._get_and_save_metrics(checkpoint_dir, log_dict, n_epochs, n_steps, timestamp)
            self.save_checkpoint(final_checkpoint_path)

        # Log everything to wandb at the end
        if self.config.use_wandb:
            wandb.log({
                "train/final_loss": loss_history[-1],
                "train/avg_loss": sum(loss_history) / len(loss_history),
                "loss_curve": wandb.Image(fig)
            })

            if val_dataloader:
                wandb.log({
                    "val/final_loss": val_loss,
                    "val/best_loss": min(val_loss_history) if val_loss_history else None
                })

            wandb.finish()

        plt.close(fig)

    def base_step(self, batch):
        """Base training step"""
        masked_spec, mask, clean_spec = batch
        clean_spec_mag_norm_log, mask, masked_spec_mag_log = utils.preprocess_data(
            clean_spec, masked_spec, mask
        )
        output = self.model(masked_spec_mag_log, mask)

        # Compute loss in normalized space
        opposite_mask = 1 - mask
        masked_loss = ((torch.abs(output - clean_spec_mag_norm_log)) ** 2) * opposite_mask
        loss = masked_loss.sum() / (opposite_mask.sum() + 1e-6)

        # Store logs
        log = {
            'clean_spec': clean_spec.detach(),
            'output': output.detach(),
            'loss': loss.detach()
        }
        return loss, log

    def validate(self, val_dataloader):
        """Validation loop"""
        self.model.eval()
        val_losses = []

        with torch.no_grad():
            for batch in val_dataloader:
                masked_spec, mask, clean_spec = [x.to(self.device) for x in batch]
                loss, _ = self.base_step((masked_spec, mask, clean_spec))
                val_losses.append(loss.item())

        avg_val_loss = sum(val_losses) / len(val_losses)
        self.model.train()
        return avg_val_loss

    def save_checkpoint(self, checkpoint_path):
        """Save model checkpoint"""
        checkpoint = {
            'model_state_dict': self.model.net.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'step': self.step,
            'config': self.config.model_dump()
        }

        os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
        torch.save(checkpoint, checkpoint_path)
        print(f"Checkpoint saved to {checkpoint_path}")

        if self.config.use_wandb:
            # Save checkpoint as artifact
            artifact = wandb.Artifact(
                name=f'model-{wandb.run.id}',
                type='model',
                description=f'Model checkpoint at step {self.step}'
            )
            artifact.add_file(checkpoint_path)
            wandb.log_artifact(artifact)

    def plot_loss_curve(self, loss_history, val_loss_history):
        """Plot the training and validation loss curves"""
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 12))

        # Plot raw losses
        ax1.plot(loss_history, label='Training Loss', alpha=0.5)
        if val_loss_history:
            ax1.plot(val_loss_history, label='Validation Loss')
        ax1.set_xlabel('Steps')
        ax1.set_ylabel('Loss')
        ax1.set_title('Raw Training and Validation Loss')
        ax1.legend()
        ax1.grid(True)

        # Plot smoothed losses
        smoothed_loss = self._smooth_losses(loss_history)
        ax2.plot(smoothed_loss, label='Smoothed Training Loss')
        if val_loss_history:
            ax2.plot(val_loss_history, label='Validation Loss')
        ax2.set_xlabel('Steps')
        ax2.set_ylabel('Loss')
        ax2.set_title('Smoothed Training and Validation Loss')
        ax2.legend()
        ax2.grid(True)

        plt.tight_layout()
        return fig

    def _smooth_losses(self, losses, window_size=100):
        """Smooth losses using moving average"""
        smoothed = []
        for i in range(len(losses)):
            start_idx = max(0, i - window_size + 1)
            smoothed.append(sum(losses[start_idx:(i + 1)]) / (i - start_idx + 1))
        return smoothed

    def _get_and_save_metrics(self, checkpoint_dir, log_dict, n_epochs, n_steps, timestamp):
        """Save training metrics to JSON file and wandb"""
        final_metrics = {
            'timestamp': timestamp,
            'total_steps': self.step,
            'final_loss': log_dict['loss'].item(),
            'training_config': {
                'n_steps': n_steps,
                'n_epochs': n_epochs,
                'learning_rate': self.config.optimizer_configuration.args.get('lr'),
                'device': self.config.device,
                'batch_size': self.config.dataloader_configuration.batch_size,
                'audio_len': self.config.data_configuration.sub_sample_length_seconds,
                'missing_length_seconds': self.config.data_configuration.missing_length_seconds,
                'missing_start_seconds': self.config.data_configuration.missing_start_seconds,
                'nfft': self.config.data_configuration.stft_configuration.nfft
            }
        }

        # Save locally
        metrics_path = os.path.join(
            checkpoint_dir,
            f"metrics_final_{timestamp}.json"
        )
        with open(metrics_path, 'w') as f:
            json.dump(final_metrics, f, indent=4)

        # Log to wandb
        if self.config.use_wandb:
            # Log metrics
            wandb.log({
                "final_metrics/total_steps": final_metrics['total_steps'],
                "final_metrics/final_loss": final_metrics['final_loss'],
                "config/learning_rate": final_metrics['training_config']['learning_rate'],
                "config/batch_size": final_metrics['training_config']['batch_size'],
                "config/audio_len": final_metrics['training_config']['audio_len'],
                "config/missing_length": final_metrics['training_config']['missing_length_seconds'],
                "config/nfft": final_metrics['training_config']['nfft']
            })

            # Save metrics file as artifact
            metrics_artifact = wandb.Artifact(
                name=f'metrics-{wandb.run.id}',
                type='metrics',
                description=f'Training metrics at step {self.step}'
            )
            metrics_artifact.add_file(metrics_path)
            wandb.log_artifact(metrics_artifact)