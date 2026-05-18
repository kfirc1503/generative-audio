# Quick Training Commands

## Prerequisites

Install ffmpeg first:
```bash
conda install -c conda-forge ffmpeg
# OR
sudo apt install ffmpeg
```

---

## Training Regular NPPC Model

### 1. Navigate to training directory:
```bash
cd /storage/kfir/repos/generative-audio-workspace/nppc_audio/inpainting/scripts/train
```

### 2. Set environment variables and run:
```bash
export PYTHONPATH=/storage/kfir/repos/generative-audio-workspace:$PYTHONPATH
export MPLBACKEND=Agg
nohup python train_nppc_model.py > training_output.log 2>&1 &
```

### 3. Monitor training:
```bash
tail -f training_output.log
```

---

## Training Latent NPPC Model

### 1. Edit `nppc_audio/inpainting/trainer/nppc_trainer.py`:

**Uncomment these lines (~119-124):**
```python
# Option 2: Optimize latent NPPC model
optimizer_class = getattr(optim, config.optimizer_configuration.type)
self.optimizer = optimizer_class(
    self.nppc_latent_model.parameters(),
    **config.optimizer_configuration.args,
)
```

**Comment out Option 1 (~112-116)**

**Change training step (~188):**
```python
# Comment out:
# reconst_err, objective, log_dict = self.base_step(batch)

# Uncomment:
reconst_err, objective, log_dict = self.latent_space_nppc_step(batch)
```

**Change checkpoint saving (~835):**
```python
# Comment out Option 1, uncomment Option 2:
checkpoint = {
    'model_state_dict': self.nppc_latent_model.state_dict(),
    'optimizer_state_dict': self.optimizer.state_dict(),
    'step': self.step,
}
```

### 2. Edit `nppc_audio/inpainting/scripts/train/config/config_nppc.yaml`:

**Change artifact name (line 19):**
```yaml
wandb_artifact_name: "nppc_latent_inpainting_model"
```

**Change tags (line 18):**
```yaml
wandb_tags:
  - "128ms_gap"
  - "use_vad"
  - "2.044sec_audio_len"
  - "libri_speech"
  - "nppc_latent_space"  # Uncomment this
```

### 3. Run training (same as regular):
```bash
cd /storage/kfir/repos/generative-audio-workspace/nppc_audio/inpainting/scripts/train
export PYTHONPATH=/storage/kfir/repos/generative-audio-workspace:$PYTHONPATH
export MPLBACKEND=Agg
nohup python train_nppc_model.py > training_latent_output.log 2>&1 &
```

### 4. Monitor:
```bash
tail -f training_latent_output.log
```

---

## Quick Commands Reference

**View live logs:**
```bash
tail -f training_output.log
```

**Check if running:**
```bash
ps aux | grep train_nppc_model.py
```

**Stop training:**
```bash
kill -TERM <PID>
```

**Find PID:**
```bash
ps aux | grep train_nppc_model.py
```
(Second column is the PID)

**Get wandb link:**
```bash
grep "View run at" training_output.log
```

---

## One-Liner (All commands combined)

```bash
cd /storage/kfir/repos/generative-audio-workspace/nppc_audio/inpainting/scripts/train && export PYTHONPATH=/storage/kfir/repos/generative-audio-workspace:$PYTHONPATH && export MPLBACKEND=Agg && nohup python train_nppc_model.py > training_output.log 2>&1 &
```

