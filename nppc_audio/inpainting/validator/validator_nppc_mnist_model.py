import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import pydantic
from pathlib import Path
import torchvision
import numpy as np
from typing import Optional
from nppc_audio.inpainting.networks.unet import LatentEncoder, RestorationWrapper, UNet, LatentEncoderConfig
from nppc_audio.inpainting.nppc.nppc_model import NPPCModelConfig, NPPCModel
from nppc_audio.inpainting.nppc.pc_wrapper import gram_schmidt_to_spec_mag


class NPPCMNISTValidatorConfig(pydantic.BaseModel):
    checkpoint_path: str
    regular_checkpoint_path: str
    device: str = "cuda"
    save_dir: str = "validation_nppc_results"
    nppc_model_configuration: NPPCModelConfig
    nppc_latent_model_configuration: LatentEncoderConfig
    max_dirs_to_plot: int = None
    n_samples: int = 25
    seed: int = 1
    t_range: tuple = (-3, 3)
    n_steps: int = 21



#
# class NPPCMNISTValidatorConfig(pydantic.BaseModel):
#     checkpoint_path: str
#     device: str = "cuda"
#     save_dir: str = "validation_nppc_mnist_results"
#     n_samples: int = 10
#     seed: int = 1
#     t_range: tuple = (-3, 3)
#     n_steps: int = 21


class NPPCMNISTValidator:
    def __init__(self, config: NPPCMNISTValidatorConfig):
        self.config = config
        self.device = config.device

        # Load checkpoint
        checkpoint = torch.load(config.checkpoint_path, map_location="cpu")
        regular_checkpoint = torch.load(config.regular_checkpoint_path, map_location="cpu")

        # Initialize models (you'll need to add your model configurations here)
        self.nppc_latent_model = LatentEncoder(config.nppc_latent_model_configuration)
        self.nppc_model = NPPCModel(self.config.nppc_model_configuration)
        self.pre_trained_restoration_model = self.nppc_model.pretrained_restoration_model
        self.pre_trained_restoration_model.eval()
        self.pre_trained_restoration_model.to(self.device)


        # Load state dict
        self.nppc_latent_model.load_state_dict(checkpoint["model_state_dict"])
        self.nppc_model.load_state_dict(regular_checkpoint["model_state_dict"])


        self.nppc_model.to(self.device)
        self.nppc_model.eval()

        self.nppc_latent_model.to(self.device)
        self.nppc_latent_model.eval()

        # Set up mask
        self.mask = torch.zeros((1, 28, 28)).to(self.device)
        self.mask[:, :20, :] = 1.

        # Load test set
        self.test_set = torchvision.datasets.MNIST(
            root='./', train=False, download=True,
            transform=torchvision.transforms.ToTensor()
        )

    def validate_samples(self):
        """Validate model on multiple samples"""
        # Set random seed
        np.random.seed(self.config.seed)
        samples_list = np.random.randint(0, len(self.test_set), self.config.n_samples)
        t_list = torch.linspace(self.config.t_range[0], self.config.t_range[1],
                                self.config.n_steps).to(self.device)
        # t_list = [0]

        for i in samples_list:
            self.validate_single_sample(i, t_list)

    def validate_single_sample(self, sample_idx: int, t_list: torch.Tensor):
        """Validate model on a single sample"""
        x_org = self.test_set[sample_idx][0][None].to(self.device)
        x_distorted = x_org * (1 - self.mask)

        with torch.no_grad():
            # Get restoration model prediction
            x_restored = self.pre_trained_restoration_model(x_distorted,self.mask)

            # Get latent representations
            # latent_restored, skip_connection_restored = self.pre_trained_restoration_model.net.encoder(x_distorted)
            latent_restored, skip_connection_restored = self.pre_trained_restoration_model.net.encoder(x_restored)
            latent_distored, skip_connection_distored = self.pre_trained_restoration_model.net.encoder(x_distorted)



            x_restored2 = self.pre_trained_restoration_model.net.decoder(latent_distored, skip_connection_distored)
            # let's normalized by x restored 2 norm and multiply by x trestored norm
            x_restored2 = x_restored2 * self.mask + x_distorted


            error = abs(x_restored2 - x_restored)
            # Print min/max values of restorations
            print(f"x_restored min: {x_restored.min():.4f}, max: {x_restored.max():.4f}")
            print(f"x_restored2 min: {x_restored2.min():.4f}, max: {x_restored2.max():.4f}")
            # Get NPPC directions in latent space
            masked_with_pred = torch.cat((x_distorted, x_restored), dim=1)
            broadcasted_mask = self.mask.view(1, 1, 28, 28).expand(x_org.shape[0], 1, 28, 28)

            w_mat_latent, latent_mask  = self.nppc_latent_model(masked_with_pred, broadcasted_mask)
            # w_mat_latent  = self.nppc_latent_model(masked_with_pred)

            latent_mask_flat = latent_mask.flatten(start_dim=1)  # [B, latent_features]
            latent_mask_flat_expanded = latent_mask_flat.unsqueeze(1)  # [B, 1, latent_features]
            latent_mask_flat_broadcasted = latent_mask_flat_expanded.expand(-1, w_mat_latent.shape[1],
                                                                            -1)  # [B, n_dirs, latent_features]

            w_latent_flat = w_mat_latent.flatten(start_dim=2)  # [B, n_dirs, latent_features]
            w_mat_latent_flat = w_latent_flat * latent_mask_flat_broadcasted
            # w_mat_latent_flat = w_latent_flat

            # here need to perform gram smit
            # Flatten and apply Gram-Schmidt normalization using existing function
            # w_mat_latent_flat = w_mat_latent.flatten(start_dim=2)  # [B, n_dirs, latent_features]
            w_mat_latent_flat = gram_schmidt_to_spec_mag(w_mat_latent_flat)
            # back to the originals dims:
            w_mat_latent = w_mat_latent_flat.view(w_mat_latent.shape)


            # Regular model directions
            broadcasted_mask = self.mask.view(1, 1, 28, 28).expand(x_org.shape[0], 1, 28, 28)
            w_mat_regular = self.nppc_model(x_distorted,broadcasted_mask)
            # w_mat_regular_flat = w_mat_regular.flatten(start_dim=2)
            # w_mat_regular_flat = gram_schmidt_to_spec_mag(w_mat_regular_flat)
            # w_mat_regular = w_mat_regular_flat.view(w_mat_regular.shape)


            # Generate variations in latent space for each direction
            variations_per_direction = []
            regular_variations = []
            # Latent model variations
            for dir_idx in range(w_mat_latent.shape[1]):
                direction_variations = []
                for t in t_list:
                    # Add scaled direction to latent representation
                    latent_var = latent_restored + t * w_mat_latent[:, dir_idx, :, :]
                    # Decode back to image space
                    img_var = self.pre_trained_restoration_model.net.decoder(latent_var, skip_connection_distored)
                    img_var = img_var * self.mask + x_distorted

                    # decode_direction = self.pre_trained_restoration_model.net.decoder(w_mat_latent[:,dir_idx], skip_connection_distored)
                    # img_var = x_restored + t * decode_direction * self.mask
                    #

                    direction_variations.append(img_var)
                variations_per_direction.append(torch.stack(direction_variations))

            # Regular model variations
            for dir_idx in range(w_mat_regular.shape[1]):
                direction_variations = []
                for t in t_list:
                    img_var = x_restored + t * w_mat_regular[:, dir_idx] * self.mask
                    # img_var = img_var * self.mask
                    direction_variations.append(img_var)
                regular_variations.append(torch.stack(direction_variations))

        # Plot results
        self._plot_comparison(x_org, x_distorted, x_restored,
                              torch.stack(variations_per_direction),
                              torch.stack(regular_variations),
                              sample_idx)

    def _plot_results(self, x_org, x_distorted, x_restored, w_mat, variations, sample_idx, x_restored2):
        """Plot validation results"""
        fig, axes = plt.subplots(2, 1, figsize=(15, 10))

        # Plot original, distorted, restored, restored2
        axes[0].imshow(torch.cat([x_org[0, 0], x_distorted[0, 0], x_restored[0, 0], x_restored2[0, 0]], dim=1).cpu(),
                       cmap='gray',
                       vmin=0,
                       vmax=1)
        axes[0].set_title("Original | Distorted | Restored | Restored2")
        axes[0].set_xticks([])
        axes[0].set_yticks([])

        # Plot first direction's variations
        first_dir_variations = variations[0]  # [n_steps, B, C, H, W]
        first_dir_variations = first_dir_variations.squeeze(1).squeeze(1)  # [n_steps, H, W]

        # Make sure each variation is properly masked and combined with x_distorted
        masked_variations = []
        for var in first_dir_variations:
            masked_var = var * self.mask[0, 0] + x_distorted[0, 0]
            masked_variations.append(masked_var)

        # Concatenate all steps horizontally
        variations_row = torch.cat(masked_variations, dim=1)  # [H, W*n_steps]

        axes[1].imshow(variations_row.cpu(),
                       cmap='gray',
                       vmin=0,
                       vmax=1)
        axes[1].set_title("Variations along first principal direction")
        axes[1].set_xticks([])
        axes[1].set_yticks([])

        plt.tight_layout()
        plt.show()
        plt.close(fig)



    def _plot_comparison(self, x_org, x_distorted, x_restored, latent_variations, regular_variations, sample_idx):
        """Plot comparison between latent and regular model variations"""
        fig, axes = plt.subplots(3, 1, figsize=(20, 12))

        # Plot original, distorted, restored
        axes[0].imshow(torch.cat([x_org[0, 0], x_distorted[0, 0], x_restored[0, 0]], dim=1).cpu(),
                      cmap='gray', vmin=0, vmax=1)
        axes[0].set_title("Original | Distorted | Restored")
        axes[0].set_xticks([])
        axes[0].set_yticks([])

        # Plot latent model variations
        first_dir_latent = latent_variations[0].squeeze(1).squeeze(1)
        variations_row_latent = torch.cat(list(first_dir_latent), dim=1)
        axes[1].imshow(variations_row_latent.cpu(), cmap='gray', vmin=0, vmax=1)
        axes[1].set_title("Latent Model: Variations along first principal direction")
        axes[1].set_xticks([])
        axes[1].set_yticks([])

        # Plot regular model variations
        first_dir_regular = regular_variations[0].squeeze(1).squeeze(1)
        variations_row_regular = torch.cat(list(first_dir_regular), dim=1)
        axes[2].imshow(variations_row_regular.cpu(), cmap='gray', vmin=0, vmax=1)
        axes[2].set_title("Regular Model: Variations along first principal direction")
        axes[2].set_xticks([])
        axes[2].set_yticks([])

        plt.tight_layout()
        plt.show()
        plt.close(fig)