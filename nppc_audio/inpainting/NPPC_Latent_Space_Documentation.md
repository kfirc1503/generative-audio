# Latent Space NPPC for Speech Inpainting

We learn PC directions in the latent space of a pre-trained restoration model instead of directly in spectrogram space.

## Data Preparation

```
┌────────────────────────────────────────────────────────────┐
│                      Input Preparation                     │
└────────────────────────────────────────────────────────────┘

like in the original method we passing the stft in binary mask, 1 represent the known areas, 0 represent the inpainting area 


   Clean Audio          Binary Mask
   "parrots..."         (1=keep, 0=missing)
       │                      │
       │ STFT                 │
       ▼                      ▼
   ┌─────────┐          ┌─────────┐
   │ ███████ │    ×     │ 1110011 │  = Masked Spec
   │ ███████ │          │ 1110011 │    (zeros in hole)
   │ ███████ │          │ 1110011 │
   └─────────┘          └────▲▲───┘
   Spectrogram          0 represents
                        missing region
```

## Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                      Latent Space NPPC Model                        │
└─────────────────────────────────────────────────────────────────────┘

unlike the regular model here we build only encoder (becasue we are working on the latent space this time)
as for input we first take the masked spec and the pred one (passing the pred on the pre trained restoration model like the original method)
then we are passing to the latent model 
in the latent model we calucalte the direction matrix and then we apply masking on the latent space and apply gram schmidt 

   masked + pred (output space)
   [B,1,F,T]  [B,1,F,T]
        │         │
        │         │
        ▼         ▼
   ┌──────────────────────┐
   │  Restoration Model   │
   │ Encoder (pre trained)│
   │                      │
   │  Encode to latent    │
   └──────────┬───────────┘
              │
              ▼
   masked_latent + pred_latent        mask (output space)
   [B,C,H,W]     [B,C,H,W]            [B,1,F,T]
              │                             │
              │ Concat: [B, 2C, H, W]       │
              │                             │
              └──────────┬──────────────────┘
                         │
                         ▼
              ┌────────────────────────┐
              │  Latent Encoder        │
              │  (trainable)           │
              │                        │
              │  Input 1: concatenated │
              │           latents      │
              │  Input 2: mask         │
              │           (separately) │
              │                        │
              │  Just a U-Net Encoder  │
              └──────────┬─────────────┘
                         │
              ┌──────────┴──────────┐
              │                     │
              ▼                     ▼
       W_latent              latent_mask
    [B,n_dirs,C,H,W]        [B,1,H,W]
              │                     │
              └──────────┬──────────┘
                         │
                         ▼
              ┌──────────────────────┐
              │ Apply Latent Mask    │
              │ W * latent_mask      │
              └──────────┬───────────┘
                         │
                         ▼
              ┌──────────────────────┐
              │ Gram-Schmidt         │
              │ Orthogonalization    │
              └──────────┬───────────┘
                         │
                         ▼
              W_latent (masked & orthogonal)
              [B,n_dirs,C,H,W]
```

## Training

```
┌────────────────────────────────────────────────────────────┐
│                    Training Flow                           │
└────────────────────────────────────────────────────────────┘
in the training section unlike the orginial method , in here i am calclating the loss in the latent space 
(whole total loss directions and norm correspoding to the latent clean stft)

Training Batch: (masked_spec, pred_spec, clean_spec)
                           ↓
              
              [1] Latent Encoder (trainable)
                  Input: [masked, pred] + mask
                  (in output space [B,2,F,T])
                           ↓
                  W_latent [B,n_dirs,C,H,W]
                  latent_mask [B,1,H,W]
                           ↓

              [2] Apply Latent Mask
                  W_latent * latent_mask
                           ↓

              [3] Gram-Schmidt Orthogonalization
                           ↓
                  W_latent (masked & orthogonal)
                           ↓

              [4] Encode clean & pred to Latent
                  (U-Net Encoder - frozen)
                           ↓
                  latent_clean [B,C,H,W]
                  latent_pred [B,C,H,W]
                           ↓

              [5] Compute Latent Error
                  latent_err = latent_clean - latent_pred
                           ↓

              [6] NPPC Loss (in latent space)
                  Project W_latent onto latent_err
                  Loss = (1-Σproj²) + λ·||W||²
                           ↓

              [7] Backprop
                  Train Latent Encoder only
```

## Validation

```
┌────────────────────────────────────────────────────────────┐
│                    Validation Flow                         │
└────────────────────────────────────────────────────────────┘
in the validation i do the same thing like the training , i am taking the test sample 
i am encoding the pred sample and the masked sample to the latent space and then apply the nppc latent encoder.
i am getting both of the latent mask and the latent direction. then unlike the regular method i am creating the combination of the 
pred + directions in the latent space, then for each combantion for exmaple x = a * w_0 + latent_pred i am using the decoder of the restoration model.
and i am plotting the spectogram output for now. 



Test Sample: (masked_spec, mask, clean_spec)
                    │
                    ▼
         ┌─────────────────────┐
         │  U-Net (frozen)     │
         └──────────┬──────────┘
                    │
                    ▼
               x̂ [1,1,F,T]
                    │
        ┌───────────┴────────────┐
        │                        │
        │ Encode to Latent       │
        ▼                        ▼
   ┌─────────────┐    ┌──────────────────────────────┐
   │ latent_pred │    │ Encode [masked, x̂]          │
   │ [1,C,H,W]   │    │ to latent space              │
   └──────┬──────┘    └──────────┬───────────────────┘
          │                      │
          │                      │ concatenated latents
          │                      │ [1, 2C, H, W]
          │                      │
          │           ┌──────────┴───────────────────┐
          │           │                              │
          │           │  Latent Encoder (trained)    │
          │           │                              │
          │           │  Receives mask as input      │
          │           │  (uses it internally)        │
          │           │          ↓                   │
          │           │  Outputs: W_latent           │
          │           │           latent_mask        │
          │           └──────────┬───────────────────┘
          │                      │
          │                      ▼
          │            W_latent [1,K,C,H,W]
          │            latent_mask [1,1,H,W]
          │                      │
          │                      ▼
          │           ┌──────────────────────┐
          │           │  Apply Latent Mask   │
          │           │  W * latent_mask     │
          │           └──────────┬───────────┘
          │                      │
          │                      ▼
          │           ┌──────────────────────┐
          │           │  Gram-Schmidt        │
          │           │  Orthogonalization   │
          │           └──────────┬───────────┘
          │                      │
          │                      ▼
          │            W_latent (masked & orthogonal)
          │            [1,K,C,H,W]
          │                      │
          └──────────┬───────────┘
                     │
                     │ For each PC i:
                     │ For α ∈ [-3, -2.5, ... 3]:
                     │
                     ▼
      ┌─────────────────────────────────┐
      │                  │
      │                                │
      │  latent_var = latent_pred       │
      │             + α·W_latent[i]     │
      │  [1, C, H, W]                   │
      │                                 │
      │  Add direction in latent, THEN  │
      │  decode to output space         │
      └──────────────┬──────────────────┘
                     │
                     ▼
         ┌───────────────────────────┐
         │  Restoration Model        │
         │  Decoder (frozen)         │
         │                           │
         │  Uses original U-Net      │
         │  decoder with skip        │
         │  connections              │
         │                           │
         │  Decode latent_var        │
         │  back to output space     │
         └───────────┬───────────────┘
                     │
                     ▼
         ┌────────────────────────┐
         │  x_var [1, 1, F, T]    │
         │                        │
         │  Different plausible   │
         │  reconstructions       │
         └──────────┬─────────────┘
                    │
                    ▼
              ┌──────────────┐
              │ Spectrogram  │
              │ Visualization│
              │              │
              └──────────────┘

Note: We modify the latent first, then decode.
```

## Key Differences

The main difference between latent space and regular output space NPPC is where we work. In the regular approach, we predict PC directions directly on spectrograms and add variations there. Here, we work entirely in the latent space of a pre-trained restoration model. We predict PC directions in latent space, apply masking and Gram-Schmidt there, and only decode back to spectrograms when generating final variations. This gives us better structure since the latent space already captures meaningful audio features.


