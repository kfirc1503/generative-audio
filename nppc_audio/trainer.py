import torch
import torch.nn as nn
import torch.optim as optim
import pydantic
import os
from datetime import datetime
import json
from nppc_audio.nppc_model import NPPCModelConfig, NPPCModel
# from nppc_model import NPPCModelConfig
from use_pre_trained_model.model_validator.config.schema import DataConfig, DataLoaderConfig
from dataset import AudioDataset
from FullSubNet_plus.speech_enhance.audio_zen.acoustics.feature import drop_band
from FullSubNet_plus.speech_enhance.audio_zen.acoustics.mask import build_complex_ideal_ratio_mask
from tqdm.auto import tqdm
from nppc.auxil import LoopLoader
from FullSubNet_plus.speech_enhance.audio_zen.acoustics.mask import decompress_cIRM


class OptimizerConfig(pydantic.BaseModel):
    type: str
    args: dict


class NPPCAudioTrainerConfig(pydantic.BaseModel):
    """Configuration for NPPCAudio trainer"""
    nppc_model_configuration: NPPCModelConfig
    data_configuration: DataConfig
    data_loader_configuration: DataLoaderConfig
    optimizer_configuration: OptimizerConfig
    # output_dir: str
    learning_rate: float = 1e-4
    device: str = "cuda"
    save_interval: int = 10
    log_interval: int = 100
    second_moment_loss_lambda: float = 1.0
    second_moment_loss_grace: int = 500


class NPPCAudioTrainer(nn.Module):
    def __init__(self, config: NPPCAudioTrainerConfig):
        super().__init__()
        self.config = config
        ## this is suppose to be the same thing
        # self.nppc_model = self.config.nppc_model_configuration.make_instance()
        self.nppc_model = NPPCModel(self.config.nppc_model_configuration)
        self.device = self.config.device
        # create data loader:
        dataset = AudioDataset(config.data_configuration.dataset)

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

    def train(self, n_steps=None, n_epochs=None, checkpoint_dir="checkpoints"):
        """
        Main training loop using LoopLoader

        Args:
            checkpoint_dir:
            n_steps: Number of training steps (optional)
            n_epochs: Number of training epochs (optional)

        Note: Must provide either n_steps or n_epochs
        """
        os.makedirs(checkpoint_dir, exist_ok=True)

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
            if isinstance(batch, (tuple, list)):
                batch = tuple(x.to(self.device) for x in batch)
            else:
                batch = batch.to(self.device)

            # Forward and backward pass
            objective, log_dict = self.base_step(batch)
            self.optimizer.zero_grad()
            objective.backward()
            self.optimizer.step()

            # Update progress bar
            pbar.set_description(f'Loss: {objective.item():.4f}')
            # Use raw values from log_dict
            pbar.set_description(
                f'Objective: {objective.item():.4f} | '
                f'Second Moment MSE: {log_dict["second_moment_mse"].mean().item():.4f}'
            )

            # # Save checkpoint periodically
            # if self.step % self.config.save_interval == 0:
            #     checkpoint_path = os.path.join(
            #         checkpoint_dir,
            #         f"checkpoint_step_{self.step}.pt"
            #     )
            #     self.save_checkpoint(checkpoint_path)
            #
            self.step += 1

        # Save final checkpoint with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        final_checkpoint_path = os.path.join(
            checkpoint_dir,
            f"checkpoint_final_{timestamp}.pt"
        )

        self._get_and_save_metrics(checkpoint_dir, log_dict, n_epochs, n_steps, timestamp)

        self.save_checkpoint(final_checkpoint_path)

    def _get_and_save_metrics(self, checkpoint_dir, log_dict, n_epochs, n_steps, timestamp):
        # Prepare metrics
        final_metrics = {
            'timestamp': timestamp,
            'total_steps': self.step,
            'final_loss': log_dict['objective'].item(),
            'final_second_moment_mse': log_dict['second_moment_mse'].mean().item(),
            'training_config': {
                'n_steps': n_steps,
                'n_epochs': n_epochs,
                # Add any other relevant configuration
                'learning_rate': self.config.learning_rate,
                'device': self.config.device,
                'snr_range': list(self.config.data_configuration.dataset.snr_range), # convert Tuple to List
                'sub_sample_length_seconds': self.config.data_configuration.dataset.sub_sample_length_seconds,
                'batch_size': self.config.data_loader_configuration.batch_size
            }
        }
        # Save metrics to JSON
        metrics_path = os.path.join(
            checkpoint_dir,
            f"metrics_final_{timestamp}.json"
        )
        with open(metrics_path, 'w') as f:
            json.dump(final_metrics, f, indent=4)

    def base_step(self, batch):
        """
        Base training step adapted from NPPC for audio enhancement using CRM directions.
        The main differences are:
        1. Our PC directions are CRM masks (complex-valued)
        2. We work with complex masks instead of real images
        3. Error is computed in CRM domain

        Args:
            batch: Dictionary containing:
                - noisy_waveform: [B,T]
                - clean_waveform: [B,T]
        Returns:
            tuple: (objective, log_dict)
        """
        model = self.nppc_model
        # Get batch data
        noisy_waveform, clean_waveform = batch
        # Get predicted CRM directions (our PC directions)
        w_mat = model(noisy_waveform)  # [B, n_dirs, 2, F, T]
        B, n_dirs, _, F, T = w_mat.shape
        # Flatten frequency and time dimensions for PC computation
        w_mat_flat = w_mat.reshape(B, n_dirs, 2, -1)  # [B, n_dirs, 2, F*T]
        # get CRMS
        num_groups_in_drop_band = self.config.nppc_model_configuration.audio_pc_wrapper_configuration.multi_direction_configuration.num_groups_in_drop_band

        gt_crm, pred_crm = self._get_true_and_pred_crm(clean_waveform, model, noisy_waveform,
                                                       num_groups_in_drop_band)
        pred_crm_flat = pred_crm.reshape(B, 2, -1)  # [B, 2, F*T]
        gt_crm_flat = gt_crm.reshape(B, 2, -1)  # [B,2,F*T]

        # Compute norms of each CRM direction
        w_norms = torch.norm(w_mat_flat, dim=(2, 3))  # [B, n_dirs]
        # Normalize CRM directions
        w_hat_mat = w_mat_flat / (w_norms[..., None, None] + 1e-8)  # [B, n_dirs, 2, F*T]
        # Compute error in CRM domain
        err = (gt_crm_flat - pred_crm_flat)  # [B, 2, F*T]
        # Normalize error
        err_norm = torch.norm(err, dim=(1, 2))  # [B]
        err = err / (err_norm[:, None, None] + 1e-8)  # [B, 2, F*T]
        w_norms = w_norms / (err_norm[:, None] + 1e-8)
        # Compute projections of error onto normalized CRM directions
        err_proj = torch.sum(w_hat_mat * err[:, None], dim=(2, 3))  # [B, n_dirs]
        # Reconstruction error (how well CRM directions explain the error)
        reconst_err = 1 - err_proj.pow(2).sum(dim=1)  # [B]
        # Second moment loss (align CRM norms with projection magnitudes)
        second_moment_mse = (w_norms.pow(2) - err_proj.detach().pow(2)).pow(2)
        # Compute final objective with adaptive weighting
        objective = self._calculate_final_objective(reconst_err, second_moment_mse)

        # Store logs
        log = {
            'noisy_complex': noisy_waveform,
            'clean_complex': clean_waveform,
            'pred_crm': pred_crm.detach(),
            'w_mat': w_mat.detach(),

            'err_norm': err_norm.detach(),
            'err_proj': err_proj.detach(),
            'w_norms': w_norms.detach(),
            'reconst_err': reconst_err.detach(),
            'second_moment_mse': second_moment_mse.detach(),

            'objective': objective.detach()
        }

        return objective, log

    def save_checkpoint(self, checkpoint_path):
        """
        Save model checkpoint including model state, optimizer state, and training info

        Args:
            checkpoint_path: Path to save checkpoint
        """
        checkpoint = {
            'model_state_dict': self.nppc_model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'config': self.config,
            'step': self.step,
        }

        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
        torch.save(checkpoint, checkpoint_path)
        print(f"Checkpoint saved to {checkpoint_path}")

    def _calculate_final_objective(self, reconst_err, second_moment_mse):
        second_moment_loss_lambda = -1 + 2 * self.step / self.config.second_moment_loss_grace
        second_moment_loss_lambda = max(min(second_moment_loss_lambda, 1), 1e-6)
        second_moment_loss_lambda *= self.config.second_moment_loss_lambda
        objective = reconst_err.mean() + second_moment_loss_lambda * second_moment_mse.mean()
        return objective

    def _get_true_and_pred_crm(self, clean_waveform, model, noisy_waveform, num_groups_in_drop_band):
        nfft = self.config.nppc_model_configuration.stft_configuration.nfft
        hop_length = self.config.nppc_model_configuration.stft_configuration.hop_length
        win_length = self.config.nppc_model_configuration.stft_configuration.win_length

        window = torch.hann_window(win_length).to(self.device)

        clean_complex = torch.stft(clean_waveform, nfft, hop_length=hop_length, win_length=win_length,
                                   window=window, return_complex=True)

        noisy_complex = torch.stft(noisy_waveform, nfft, hop_length=hop_length, win_length=win_length, window=window,
                                   return_complex=True)

        gt_crm = build_complex_ideal_ratio_mask(noisy_complex, clean_complex)  # [B, F, T, 2]
        tmp = gt_crm
        gt_crm = drop_band(
            gt_crm.permute(0, 3, 1, 2),  # [B, 2, F ,T]
            num_groups_in_drop_band
        )
        #gt_crm = decompress_cIRM(gt_crm)
        # not sure if necessary to turn it back to [B,F,T,2]
        # gt_cIRM = gt_cIRM.permute(0,2,3,1) # [B,F,T,2]
        # Get enhanced CRM from model's prediction
        pred_crm = model.get_pred_crm(noisy_waveform)  # [B,2,F,T]
        pred_crm = drop_band(pred_crm, num_groups_in_drop_band)

        # pred_crm = pred_crm.permute(0, 3, 1, 2) # [B,2,F,T]
        return gt_crm, pred_crm
