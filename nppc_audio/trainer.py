import torch
import torch.nn as nn
import numpy as np
import os
from typing import Optional
import pydantic
from pathlib import Path
from nppc_model import NPPCModelConfig, NPPCModel
from use_pre_trained_model.model_validator.config.schema import DataConfig
from dataset import AudioDataset, AudioDataSetConfig
from FullSubNet_plus.speech_enhance.audio_zen.acoustics.feature import mag_phase, drop_band
from FullSubNet_plus.speech_enhance.audio_zen.acoustics.mask import build_complex_ideal_ratio_mask, decompress_cIRM, build_ideal_ratio_mask

class NPPCAudioTrainerConfig(pydantic.BaseModel):
    """Configuration for NPPCAudio trainer"""
    nppc_model_configuration: NPPCModelConfig
    data_configuration: DataConfig
    output_dir: str
    learning_rate: float = 1e-4
    n_epochs: int = 100
    device: str = "cuda"
    save_interval: int = 10
    log_interval: int = 100
    second_moment_loss_lambda: float = 1.0
    second_moment_loss_grace: int = 1000


class NPPCAudioTrainer(nn.Module):
    def __init__(self, config: NPPCAudioTrainerConfig):
        super().__init__()
        self.config = config
        ## this is suppose to be the same thing
        self.nppc_model = self.config.nppc_model_configuration.make_instance()
        # self.nppc_model = NPPCModel(self.nppc_model)
        self.device = self.config.device
        # create data loader:
        dataset = AudioDataset(config.data_config.dataset)

        print(f"Total sample pairs in dataset: {len(dataset)}")

        # Create dataloader
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=config.data_loader.batch_size,  # Adjust based on your GPU memory
            shuffle=config.data_loader.shuffle,
            num_workers=config.data_loader.num_workers,
            pin_memory=config.data_loader.pin_memory
        )
        self.dataloader = dataloader

    def train(self):
        pass

    def base_step(self, batch):
        """
        Base training step adapted from NPPC for audio enhancement using CRM directions.
        The main differences are:
        1. Our PC directions are CRM masks (complex-valued)
        2. We work with complex masks instead of real images
        3. Error is computed in CRM domain

        Args:
            batch: Dictionary containing:
                - noisy_complex: [B, F, T] Complex STFT of noisy speech
                - clean_complex: [B, F, T] Complex STFT of clean speech
        Returns:
            tuple: (objective, log_dict)
        """
        model = self.nppc_model

        # Get batch data
        noisy_complex, clean_complex = batch

        # Get predicted CRM directions (our PC directions)
        w_mat = model(noisy_complex)  # [B, n_dirs, 2, F, T]
        B, n_dirs, _, F, T = w_mat.shape

        # Flatten frequency and time dimensions for PC computation
        w_mat_flat = w_mat.reshape(B, n_dirs, 2, -1)  # [B, n_dirs, 2, F*T]

        # Compute true CRM (target)
        true_crm = clean_complex / (noisy_complex + 1e-8)  # [B, F, T]
        true_crm = torch.stack([true_crm.real, true_crm.imag], dim=1)  # [B, 2, F, T]
        true_crm_flat = true_crm.reshape(B, 2, -1)  # [B, 2, F*T]

        gt_cIRM = build_complex_ideal_ratio_mask(noisy_complex, clean_complex)  # [B, F, T, 2]
        gt_cIRM = drop_band(
            gt_cIRM.permute(0, 3, 1, 2),  # [B, 2, F ,T]
            self.model.module.num_groups_in_drop_band
        )
        gt_cIRM = gt_cIRM.permute(0,2,3,1) # [B,F,T,2]


        # Compute norms of each CRM direction
        w_norms = torch.norm(w_mat_flat, dim=(2, 3))  # [B, n_dirs]

        # Normalize CRM directions
        w_hat_mat = w_mat_flat / (w_norms[..., None, None] + 1e-8)  # [B, n_dirs, 2, F*T]

        # Get enhanced CRM from model's prediction
        pred_crm = model.get_pred_crm(noisy_complex)  # [B, 2, F, T]
        pred_crm_flat = pred_crm.reshape(B, 2, -1)  # [B, 2, F*T]

        # Compute error in CRM domain
        err = (true_crm_flat - pred_crm_flat)  # [B, 2, F*T]

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
        second_moment_loss_lambda = -1 + 2 * model.step / model.second_moment_loss_grace
        second_moment_loss_lambda = max(min(second_moment_loss_lambda, 1), 1e-6)
        second_moment_loss_lambda *= model.second_moment_loss_lambda

        objective = reconst_err.mean() + second_moment_loss_lambda * second_moment_mse.mean()

        # Store logs
        log = {
            'noisy_complex': noisy_complex,
            'clean_complex': clean_complex,
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