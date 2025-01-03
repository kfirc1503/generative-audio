import torch
import torch.nn as nn
from typing import Literal
import pydantic
import torch.optim as optim
import json
import os
from datetime import datetime
from pathlib import Path
from tqdm.auto import tqdm
from nppc.auxil import LoopLoader
import matplotlib.pyplot as plt
from nppc_audio.inpainting.nppc.pc_wrapper import AudioInpaintingPCWrapperConfig, AudioInpaintingPCWrapper
from nppc_audio.inpainting.networks.unet import UNet, UNetConfig, RestorationWrapper
from nppc_audio.inpainting.nppc.nppc_model import NPPCModelConfig, NPPCModel
from dataset.audio_dataset_inpainting import AudioInpaintingDataset, AudioInpaintingConfig

import utils
from nppc_audio.trainer import NPPCAudioTrainer
from utils import OptimizerConfig, DataLoaderConfig


class NPPCAudioInpaintingTrainerConfig(pydantic.BaseModel):
    nppc_model_configuration: NPPCModelConfig
    data_configuration: AudioInpaintingConfig
    data_loader_configuration: DataLoaderConfig
    optimizer_configuration: OptimizerConfig
    # output_dir: str
    learning_rate: float = 1e-4
    device: str = "cuda"
    save_interval: int = 10
    log_interval: int = 100
    second_moment_loss_lambda: float = 1.0
    second_moment_loss_grace: int = 500


class NPPCAudioInpaintingTrainer(NPPCAudioTrainer):
    def __init__(self, config: NPPCAudioInpaintingTrainerConfig):
        super().__init__()
        self.config = config
        ## this is suppose to be the same thing
        # self.nppc_model = self.config.nppc_model_configuration.make_instance()
        self.nppc_model = NPPCModel(self.config.nppc_model_configuration)
        self.device = self.config.device
        # create data loader:
        dataset = AudioInpaintingDataset(config.data_configuration)

        print(f"Total sample pairs in dataset: {len(dataset)}")

        # Create dataloader
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=config.data_loader_configuration.batch_size,  # Adjust based on your GPU memory
            shuffle=config.data_loader_configuration.shuffle,
            num_workers=config.data_loader_configuration.num_workers,
            pin_memory=config.data_loader_configuration.pin_memory
        )
        self.dataloader = dataloader
        self.step = 0

        # Initialize optimizer
        optimizer_class = getattr(optim, config.optimizer_configuration.type)
        self.optimizer = optimizer_class(
            self.nppc_model.parameters(),
            **config.optimizer_configuration.args
        )

    def train(self, n_steps=None, n_epochs=None, checkpoint_dir="checkpoints", save_flag=True):
        """Main training loop using LoopLoader"""
        os.makedirs(checkpoint_dir, exist_ok=True)

        # Initialize loss history
        loss_history = []

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
            reconst_err, objective, log_dict = self.base_step(batch)
            self.optimizer.zero_grad()
            objective.backward()
            self.optimizer.step()

            # Store loss
            loss_history.append(objective.item())

            # Update progress bar
            pbar.set_description(
                f'Loss: {objective.item():.4f}'
            )
            pbar.set_description(
                f'Objective: {objective.item():.4f} | '
                f'Second Moment MSE: {log_dict["second_moment_mse"].mean().item():.4f} | '
                f'Reconstract Error: {reconst_err.mean().item():.4f}'
            )

            self.step += 1

        # Plot loss curve
        self.plot_loss_curve(loss_history)

        if save_flag:
            # Save final checkpoint
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            final_checkpoint_path = os.path.join(
                checkpoint_dir,
                f"checkpoint_final_{timestamp}.pt"
            )

            # Save metrics including loss history
            self._get_and_save_metrics(checkpoint_dir, log_dict, n_epochs, n_steps, timestamp)
            self.save_checkpoint(final_checkpoint_path)

    def plot_loss_curve(self, loss_history):
        """Plot the training loss curve"""
        plt.figure(figsize=(10, 6))
        plt.plot(loss_history)
        plt.xlabel('Steps')
        plt.ylabel('Loss')
        plt.title('Training Loss Over Time')
        plt.grid(True)
        plt.show()

    def base_step(self, batch):
        """
        base step function for training the nppc for the inpainting audio task
        Args:
            batch:

        Returns:

        """
        # firstly we should move the spec into a mag norm log specs:
        # masked_spec, mask, clean_spec = batch
        masked_spec, mask, clean_spec = batch  # ignore masked_spec
        mask = mask.unsqueeze(1).unsqueeze(2)
        mask = mask.expand(-1, 1, clean_spec.shape[2], -1)
        # calculate the mag of the clean and masked specs:
        clean_spec_mag = torch.sqrt(clean_spec[:, 0, :, :] ** 2 + clean_spec[:, 1, :, :] ** 2)
        clean_spec_mag = clean_spec_mag.unsqueeze(1)
        clean_spec_mag_norm_log, _, _ = utils.preprocess_log_magnitude(clean_spec_mag)
        masked_spec_mag_log = clean_spec_mag_norm_log * mask
        # now we will get the pred spec mag from our pretrained restoration model
        w_mat = self.nppc_model(masked_spec_mag_log, mask)  # [B,n_dirs,F,T]
        B, n_dirs, F, T = w_mat.shape
        w_mat_flat = w_mat.reshape(B, n_dirs, -1)  # [B,n_dirs,F*T]
        w_norms = torch.norm(w_mat_flat, dim=-1)  # dim of [B,n_dirs]

        pred_spec_mag_norm_log = self.nppc_model.get_pred_spec_mag_norm(masked_spec_mag_log, mask)
        err = (clean_spec_mag_norm_log - pred_spec_mag_norm_log).flatten(1)  # [B,1,F*T]

        ## Normalizing by the error's norm
        ## -------------------------------
        err_norm = err.norm(dim=1)
        err = err / err_norm[:, None]
        w_norms = w_norms / err_norm[:, None]

        # normalized the directions:
        w_hat_mat = w_mat_flat / (w_norms[:, :, None] + 1e-8)
        ## W hat loss
        ## ----------

        err_proj = torch.einsum('bki,bi->bk', w_hat_mat, err)
        reconst_err = 1 - err_proj.pow(2).sum(dim=1)
        ## W norms loss
        ## ------------
        second_moment_mse = (w_norms.pow(2) - err_proj.detach().pow(2)).pow(2)
        # Compute final objective with adaptive weighting
        objective = self._calculate_final_objective(reconst_err, second_moment_mse)
        # Store logs
        log = {
            'w_mat': w_mat.detach(),
            'err_norm': err_norm.detach(),
            'err_proj': err_proj.detach(),  # Keeping the complex projection for logging if needed
            'w_norms': w_norms.detach(),
            'reconst_err': reconst_err.detach(),
            'second_moment_mse': second_moment_mse.detach(),
            'objective': objective.detach()
        }

        return reconst_err, objective, log

    def save_checkpoint(self, checkpoint_path):
        """
        Save model checkpoint including model state, optimizer state, and training info

        Args:
            checkpoint_path: Path to save checkpoint
        """
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'step': self.step,
        }
        os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
        torch.save(checkpoint, checkpoint_path)
        print(f"Checkpoint saved to {checkpoint_path}")

    def _get_and_save_metrics(self, checkpoint_dir, log_dict, n_epochs, n_steps, timestamp):
        """Save training metrics to JSON file"""
        final_metrics = {
            'timestamp': timestamp,
            'total_steps': self.step,
            'final_loss': log_dict['objective'].item(),
            'training_config': {
                'n_steps': n_steps,
                'n_epochs': n_epochs,
                'learning_rate': self.config.learning_rate,
                'device': self.config.device,
                'batch_size': self.config.dataloader_configuration.batch_size
            }
        }
        metrics_path = os.path.join(
            checkpoint_dir,
            f"metrics_final_{timestamp}.json"
        )
        with open(metrics_path, 'w') as f:
            json.dump(final_metrics, f, indent=4)

    def _calculate_final_objective(self, reconst_err, second_moment_mse):
        second_moment_loss_lambda = -1 + 2 * self.step / self.config.second_moment_loss_grace
        second_moment_loss_lambda = max(min(second_moment_loss_lambda, 1), 1e-6)
        second_moment_loss_lambda *= self.config.second_moment_loss_lambda
        objective = reconst_err.mean() + second_moment_loss_lambda * second_moment_mse.mean()
        return objective
