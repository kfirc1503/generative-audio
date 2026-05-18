import random

from IPython.display import Markdown

import numpy as np
# import tqdm.notebook as tqdm
import plotly.graph_objects as go
import plotly.express as px
import torch
import torch.nn as nn
import torchvision
import pydantic
from tqdm.auto import tqdm  # This will automatically choose the right version


device = 'cuda:0'

restoration_n_steps = 1000
nppc_n_steps = 1000

# restoration_n_steps = 3000
# nppc_n_steps = 3000

batch_size = 256

second_moment_loss_lambda = 1e0
second_moment_loss_grace = 500

mask = torch.zeros((1, 28, 28)).to(device)
mask[:, :20, :] = 1.

n_dirs = 5



def sample_to_width(x, width=1580, padding_size=2):
    n_samples = min((width - padding_size) // (x.shape[-1] + padding_size), x.shape[0])
    indices = np.linspace(0, x.shape[0] - 1, n_samples).astype(int)
    return x[indices]

def imgs_to_grid(imgs, nrows=None, **make_grid_args):
    imgs = imgs.detach().cpu()
    if imgs.ndim == 5:
        nrow = imgs.shape[1]
        imgs = imgs.reshape(imgs.shape[0] * imgs.shape[1], imgs.shape[2], imgs.shape[3], imgs.shape[4])
    elif nrows is None:
        nrow = int(np.ceil(imgs.shape[0] ** 0.5))

    make_grid_args2 = dict(value_range=(0, 1), pad_value=1.)
    make_grid_args2.update(make_grid_args)
    img = torchvision.utils.make_grid(imgs, nrow=nrow, **make_grid_args2).clamp(0, 1)
    return img

def scale_img(x):
    return x / torch.abs(x).flatten(-3).max(-1)[0][..., None, None, None] / 1.5 + 0.5

def tensor_img_to_numpy(x):
    return x.detach().permute(-2, -1, -3).cpu().numpy()

def imshow(img, scale=1, **kwargs):
    if isinstance(img, torch.Tensor):
        img = tensor_img_to_numpy(img)
    img = img.clip(0, 1)

    fig = px.imshow(img, **kwargs).update_layout(
        height=img.shape[0] * scale,
        width=img.shape[1] * scale,
        margin=dict(t=0, b=0, l=0, r=0),
        xaxis_showticklabels=False,
        yaxis_showticklabels=False,
    )
    return fig


class LoopLoader():
    def __init__(self, dataloader, size):
        self.dataloader = dataloader
        self.size = size

    def __len__(self):
        return self.size

    def __iter__(self):
        i = 0
        while (i < self.size):
            for x in self.dataloader:
                if (i >= self.size):
                    break
                yield x
                i += 1


train_set = torchvision.datasets.MNIST(root='./', download=True, train=True, transform=torchvision.transforms.ToTensor())
test_set = torchvision.datasets.MNIST(root='./', train=False, transform=torchvision.transforms.ToTensor())

dataloader = torch.utils.data.DataLoader(
    test_set,
    batch_size=batch_size,
    shuffle=True,
)
test_batch = next(iter(dataloader))







# sub-parts of the U-Net model

import torch
import torch.nn as nn
import torch.nn.functional as F


class double_conv(nn.Module):
    '''(conv => BN => ReLU) * 2'''

    def __init__(self, in_ch, out_ch, leaky_relu=True, dropout=0):
        super(double_conv, self).__init__()
        layers = [
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch)
        ]
        if leaky_relu:
            layers.append(nn.LeakyReLU(0.2))
        else:
            layers.append(nn.ReLU(inplace=True))

        layers.extend([nn.Conv2d(out_ch, out_ch, 3, padding=1),
                       nn.BatchNorm2d(out_ch)])
        if leaky_relu:
            layers.append(nn.LeakyReLU(0.2))
        else:
            layers.append(nn.ReLU(inplace=True))

        if dropout:
            layers.append(nn.Dropout(dropout))
        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv(x)
        return x


class inconv(nn.Module):
    def __init__(self, in_ch, out_ch, leaky_relu=True):
        super(inconv, self).__init__()
        self.conv = double_conv(in_ch, out_ch, leaky_relu)

    def forward(self, x):
        x = self.conv(x)
        return x


class down(nn.Module):
    def __init__(self, in_ch, out_ch, leaky_relu=True, dropout=0):
        super(down, self).__init__()
        self.mpconv = nn.Sequential(
            nn.MaxPool2d(2),
            double_conv(in_ch, out_ch, leaky_relu, dropout)
        )

    def forward(self, x):
        x = self.mpconv(x)
        return x


class up(nn.Module):
    def __init__(self, in_ch, out_ch, bilinear=True, leaky_relu=True, dropout=0):
        super(up, self).__init__()

        #  would be a nice idea if the upsampling could be learned too,
        #  but my machine do not have enough memory to handle all those weights
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        else:
            self.up = nn.ConvTranspose2d(in_ch // 2, in_ch // 2, 2, stride=2)

        self.conv = double_conv(in_ch, out_ch, leaky_relu, dropout)

    def forward(self, x1, x2):
        x1 = self.up(x1)

        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, (diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2))

        # for padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd

        x = torch.cat([x2, x1], dim=1)
        x = self.conv(x)
        return x


class outconv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(outconv, self).__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, 1)

    def forward(self, x):
        x = self.conv(x)
        return x







class Encoder(nn.Module):
    def __init__(self, in_channels=1, dropout=0):
        super(Encoder, self).__init__()
        self.inc = inconv(in_channels, 64)
        self.down1 = down(64, 128)
        self.down2 = down(128, 256)
        self.down3 = down(256, 512, dropout=dropout)
        self.down4 = down(512, 512, dropout=dropout)

    def forward(self, x):
        # Store intermediate outputs for skip connections
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        return x5, [x4, x3, x2, x1]


class Decoder(nn.Module):
    def __init__(self, out_channels=1, dropout=0):
        super(Decoder, self).__init__()
        self.up1 = up(1024, 256, dropout=dropout)  # 512 + 512 = 1024
        self.up2 = up(512, 128, dropout=dropout)  # 256 + 256 = 512
        self.up3 = up(256, 64)  # 128 + 128 = 256
        self.up4 = up(128, 64)  # 64 + 64 = 128
        self.outc = outconv(64, out_channels)

    def forward(self, x, skip_connections):
        x = self.up1(x, skip_connections[0])
        x = self.up2(x, skip_connections[1])
        x = self.up3(x, skip_connections[2])
        x = self.up4(x, skip_connections[3])
        x = self.outc(x)
        return x

class UNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, dropout=0.0):
        super(UNet, self).__init__()
        self.encoder = Encoder(in_channels, dropout=dropout)
        self.decoder = Decoder(out_channels, dropout=dropout)

    def forward(self, x):
        encoded, skip_connections = self.encoder(x)
        decoded = self.decoder(encoded, skip_connections)
        return decoded


class RestorationWrapper(nn.Module):
    def __init__(self, net, mask):
        super().__init__()

        self.net = net
        self.mask = mask

    def forward(self, x):
        x_in = x

        # x = (x - 0.5) / 0.2
        x = self.net(x)
        # x = (x * 0.2) + 0.5

        x = x_in + x * self.mask
        return x





# Assuming you have the double_conv, inconv, down classes defined elsewhere

class LatentEncoder(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, dropout=0.0, n_dirs=5):
        super(LatentEncoder, self).__init__()

        self.n_dirs = n_dirs

        self.inc = inconv(in_channels, 64)
        self.down1 = down(64, 128)
        self.down2 = down(128, 256)
        self.down3 = down(256, 512, dropout=dropout)
        self.down4 = down(512, 512 * n_dirs, dropout=dropout)

    def forward(self, x, mask):
        x1 = self.inc(x)
        # mask1 = F.max_pool2d(self.mask, kernel_size=2)  # Apply max pooling to mask
        x2 = self.down1(x1)
        # mask2 = F.max_pool2d(mask, kernel_size=2)
        x3 = self.down2(x2)
        # mask3 = F.max_pool2d(mask2, kernel_size=2)
        x4 = self.down3(x3)
        # mask4 = F.max_pool2d(mask3, kernel_size=2)
        x5 = self.down4(x4)
        # mask5 = F.max_pool2d(mask4, kernel_size=2)  # Final mask in latent space

        # Reshape to separate n_dirs dimension using C // n_dirs
        B, C, H, W = x5.shape
        x5 = x5.view(B, self.n_dirs, C // self.n_dirs, H, W)

        return x5, mask  # Return latent representation and mask

def gram_schmidt(x):
    x_shape = x.shape
    x = x.flatten(2)

    x_orth = []
    proj_vec_list = []
    for i in range(x.shape[1]):
        w = x[:, i, :]
        for w2 in proj_vec_list:
            w = w - w2 * torch.sum(w * w2, dim=-1, keepdim=True)
        w_hat = w.detach() / w.detach().norm(dim=-1, keepdim=True)

        x_orth.append(w)
        proj_vec_list.append(w_hat)

    x_orth = torch.stack(x_orth, dim=1).view(*x_shape)
    return x_orth





restoration_net = RestorationWrapper(UNet(), mask=mask)
restoration_net.to(device)
restoration_net.train()
restoration_optimizer = torch.optim.Adam(restoration_net.parameters(), lr=1e-4, betas=(0.9, 0.999))
restoration_step = 0

nppc_net = LatentEncoder(in_channels=2, out_channels=5, dropout=0.0, n_dirs=5)
nppc_net.to(device)
nppc_net.train()
nppc_optimizer = torch.optim.Adam(nppc_net.parameters(), lr=1e-4, betas=(0.9, 0.999))
nppc_step = 0



dataloader = torch.utils.data.DataLoader(
    train_set,
    batch_size=batch_size,
    shuffle=True,
)

restoration_objective_log = []
for batch in tqdm(LoopLoader(dataloader, restoration_n_steps)):
    x_org = batch[0].to(device)
    x_distorted = x_org * (1 - mask)

    x_restored = restoration_net(x_distorted)
    err = x_org - x_restored
    objective = err.pow(2).flatten(1).mean()

    restoration_optimizer.zero_grad()
    objective.backward()
    restoration_optimizer.step()
    restoration_step += 1

    if restoration_step % 100:
        restoration_objective_log.append(objective.detach().item())





def calculate_final_objective(reconst_err, second_moment_mse, step):
    second_moment_loss_lambda = -1 + 2 * step / second_moment_loss_grace
    second_moment_loss_lambda = max(min(second_moment_loss_lambda, 1), 1e-6)
    # second_moment_loss_lambda = 1

    second_moment_loss_lambda *= second_moment_loss_lambda
    objective = reconst_err.mean() + second_moment_loss_lambda * second_moment_mse.mean()
    return objective



def latent_space_nppc_step(batch,mask, restoration_net,latent_encoder, step):
    """
    Perform NPPC in latent space instead of output space.
    Args:
        batch: input batch (masked_spec, mask, clean_spec)

    Returns:
        reconst_err_latent, objective_latent, log
    """
    # Step 1: Preprocessing data
    x_org = batch[0].to(device)
    x_distorted = x_org * (1 - mask)
    broadcasted_mask = mask.view(1, 1, 28, 28).expand(x_org.shape[0], 1, 28, 28)
    # print(f"broadcasted mask shape is {broadcasted_mask.shape}")

    # Step 2: Encoding into latent space
    with torch.no_grad():
        # these latent guys need to be multiply by latent_mask later  !
        latent_clean , _  = restoration_net.net.encoder(x_org)
        latent_pred , _ = restoration_net.net.encoder(x_distorted)
        x_pred = restoration_net(x_distorted)
        # print(f"x_pred shape is {x_pred.shape}")



    # Step 3: Latent Error computation
    latent_err = latent_clean - latent_pred
    # print(f"latent_err shape is {latent_err.shape}")
    latent_err_flat = latent_err.flatten(start_dim=1)  # [B, latent_features]
    # print(latent_err_flat[0,:10])
    # Step 4: Predict latent directions
    # w_latent = self.nppc_latent_model(masked_spec_norm_log, mask)
    x_distorted_with_pred = torch.cat(
        (x_distorted, x_pred),
        dim=1
    )

    w_latent, latent_mask = latent_encoder(x_distorted_with_pred, broadcasted_mask)
    # print(f"w_latent shape is {w_latent.shape}")
    # print(f"latent_mask shape is {latent_mask.shape}")
    # latent_mask_flat = latent_mask.flatten(start_dim=1)  # [B, latent_features]
    # latent_err_flat = latent_err_flat * latent_mask_flat

    # latent_mask_flat_expanded = latent_mask_flat.unsqueeze(1)
    # latent_mask_flat_broadcasted = latent_mask_flat_expanded.expand(-1, w_latent.shape[1], -1)

    w_latent_flat = w_latent.flatten(start_dim=2)  # [B, n_dirs, latent_features]
    # broadcast the mask to w_latent_flat_shape
    # w_latent_flat = w_latent_flat * latent_mask_flat_broadcasted

    # Step 5: Gram-Schmidt normalization (latent space)
    # still not implemented

    # print(w_latent_flat[0,0,:10])
    w_latent_flat = gram_schmidt(w_latent_flat)
    # print(w_latent_flat[0,0,:10])
    w_norms_latent = w_latent_flat.norm(dim=2) + 1e-6
    w_hat_latent = w_latent_flat / w_norms_latent[:,:, None]
    # print(w_hat_latent[0,0,:10])
    # Step 6: Project latent error
    latent_err_norm = latent_err_flat.norm(dim=1) + 1e-6
    # print(latent_err_norm[:10])
    latent_err_normalized = latent_err_flat / latent_err_norm[:, None]
    w_norms_latent_normalized = w_norms_latent / latent_err_norm[:, None]

    latent_err_proj = torch.einsum('bki,bi->bk', w_hat_latent, latent_err_normalized)
    # print(latent_err_proj[:10,0])
    # Step 7: Latent reconstruction and variance losses
    reconst_err_latent = 1 - latent_err_proj.pow(2).sum(dim=1)
    second_moment_mse_latent = (w_norms_latent_normalized.pow(2) - latent_err_proj.detach().pow(2)).pow(2)


    # Step 8: Final combined loss
    objective_latent = calculate_final_objective(
        reconst_err_latent,
        second_moment_mse_latent,
        step
    )

    # Logging dictionary
    log = {
        'latent_err_norm': latent_err_norm.detach(),
        'latent_err_proj': latent_err_proj.detach(),
        'w_norms': w_norms_latent.detach(),
        'reconst_err': reconst_err_latent.detach(),
        'second_moment_mse': second_moment_mse_latent.detach(),
        'objective': objective_latent.detach(),
        'w_latent': w_latent.detach()
    }

    return reconst_err_latent, objective_latent,second_moment_mse_latent, log


# prompt: okay gemini i want you to now write the traning nppc model just like under be but we our new function that we wrote above, latent space step, i want to write the training with our new method

import numpy as np
# **Train NPPC model**
nppc_objective_log = []
nppc_second_moment_mse_log = []
restoration_net.eval()
nppc_net.train()

for batch in tqdm(LoopLoader(dataloader, nppc_n_steps)):
    reconst_err_latent, objective_latent, second_moment_mse_latent, log = latent_space_nppc_step(batch, mask, restoration_net, nppc_net, nppc_step)

    nppc_optimizer.zero_grad()
    objective_latent.backward()
    nppc_optimizer.step()
    nppc_step += 1

    if nppc_step % 100 == 0:
        nppc_objective_log.append(objective_latent.detach().item())
        nppc_second_moment_mse_log.append(second_moment_mse_latent.detach().mean().item())
        # Print objective and second moment loss
    if nppc_step % 100 == 0:  # Print every 100 steps
        print(f"Step: {nppc_step}, Objective: {objective_latent.item():.4f}, Second Moment Loss: {second_moment_mse_latent.mean().item():.4f} , the reconstrat error: {reconst_err_latent.mean().item()}")


# go.Figure(data=[go.Scatter(mode='lines', x=np.arange(len(nppc_objective_log)) * 100, y=nppc_objective_log)],
#           layout=go.Layout(yaxis_title='Objective', xaxis_title='step', height=400, width=550, margin=dict(t=0, b=20, l=20, r=0)),
#           ).show()

# go.Figure(data=[go.Scatter(mode='lines', x=np.arange(len(nppc_second_moment_mse_log)) * 100, y=nppc_second_moment_mse_log)],
#           layout=go.Layout(yaxis_title='Second Moment MSE', xaxis_title='step', height=400, width=550, margin=dict(t=0, b=20, l=20, r=0)),
#           ).show()
