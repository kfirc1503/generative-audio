# Training NPPC Models with nohup - Complete Guide

This guide provides step-by-step instructions for running NPPC model training in the background using `nohup`.

---

## 📋 Prerequisites

### 1. Install ffmpeg (Required for Audio Generation)

**Ubuntu/Debian:**
```bash
sudo apt update
sudo apt install ffmpeg
```

**CentOS/RHEL:**
```bash
sudo yum install ffmpeg
```

**Using Conda (Recommended if in conda environment):**
```bash
conda install -c conda-forge ffmpeg
```

**Verify Installation:**
```bash
ffmpeg -version
```

You should see version information printed.

---

## 🚀 Running Training with nohup

### Step 1: Navigate to Training Directory

```bash
cd /storage/kfir/repos/generative-audio-workspace/nppc_audio/inpainting/scripts/train
```

### Step 2: Set Required Environment Variables

```bash
# Tell Python where to find modules
export PYTHONPATH=/storage/kfir/repos/generative-audio-workspace:$PYTHONPATH

# Set matplotlib backend for non-interactive plotting (critical for nohup!)
export MPLBACKEND=Agg
```

**Why these are needed:**
- `PYTHONPATH`: Ensures Python can find all project modules
- `MPLBACKEND=Agg`: Prevents matplotlib from trying to open display windows (which would crash in nohup)

### Step 3: Run Training with nohup

**For Regular NPPC Model:**
```bash
nohup python train_nppc_model.py > training_output.log 2>&1 &
```

**One-liner (all commands combined):**
```bash
cd /storage/kfir/repos/generative-audio-workspace/nppc_audio/inpainting/scripts/train && \
export PYTHONPATH=/storage/kfir/repos/generative-audio-workspace:$PYTHONPATH && \
export MPLBACKEND=Agg && \
nohup python train_nppc_model.py > training_output.log 2>&1 &
```

**What this does:**
- `nohup`: Runs the command immune to hangups (keeps running after you logout)
- `python train_nppc_model.py`: Runs the training script
- `> training_output.log`: Redirects stdout to log file
- `2>&1`: Redirects stderr to the same log file
- `&`: Runs the process in the background

### Step 4: Get Process ID

After running the command, you'll see output like:
```
[1] 12345
```

The number `12345` is your **process ID (PID)**. Save this!

---

## 📊 Monitoring Training

### View Live Logs

```bash
tail -f training_output.log
```

Press `Ctrl+C` to stop viewing (training continues running).

### Check if Training is Still Running

```bash
ps aux | grep train_nppc_model.py
```

Or use the PID:
```bash
ps -p 12345
```

### View Last N Lines of Log

```bash
tail -n 100 training_output.log
```

### Search Logs for Specific Information

**Find loss values:**
```bash
grep "Objective" training_output.log
```

**Find wandb links:**
```bash
grep "wandb" training_output.log
```

**Find errors:**
```bash
grep -i "error" training_output.log
```

---

## ⚙️ Training Configuration

### Before Running Training

1. **Edit config file:** `nppc_audio/inpainting/scripts/train/config/config_nppc.yaml`

2. **Key settings to check:**
   - `checkpoint_dir`: Where checkpoints will be saved
   - `n_epochs`: Number of training epochs
   - `wandb_artifact_name`: Name for wandb artifact
   - `wandb_tags`: Tags for the run
   - `clean_path`: Path to training data

### Choose Training Mode

Edit `nppc_audio/inpainting/trainer/nppc_trainer.py` around lines 116-190:

**Option 1: Regular NPPC (Default)**
```python
# Line ~116: Optimizer
self.optimizer = torch.optim.Adam(self.nppc_model.parameters(), lr=lr)

# Line ~180: Training step
reconst_err, objective, log = self.base_step(batch)

# Line ~728: Save checkpoint
checkpoint = {
    'model_state_dict': self.nppc_model.state_dict(),
    ...
}
```

**Option 2: Latent NPPC**
- Uncomment optimizer for `nppc_latent_model` (line ~116-120)
- Use `self.latent_space_nppc_step(batch)` (line ~183)
- Save `nppc_latent_model.state_dict()` (line ~733)
- Change `wandb_artifact_name` in config to `nppc_latent_inpainting_model`

---

## 🛑 Stopping Training

### Graceful Stop (Recommended)

```bash
kill -TERM <PID>
```

### Force Stop (if graceful doesn't work)

```bash
kill -9 <PID>
```

### Find PID if you forgot it

```bash
ps aux | grep train_nppc_model.py
```

The second column is the PID.

---

## 📁 Output Locations

### Checkpoints
Saved to the path specified in `checkpoint_dir` in your config:
```
/storage/kfir/data/inpainting/nppc_model/checkpoint/
```

### Wandb Logs
Local wandb data stored in:
```
/storage/kfir/repos/generative-audio-workspace/nppc_audio/inpainting/scripts/train/wandb/
```

Online dashboard:
```
https://wandb.ai/kfirc-tel-aviv-university/generative-audio
```

### Training Output Log
```
/storage/kfir/repos/generative-audio-workspace/nppc_audio/inpainting/scripts/train/training_output.log
```

---

## ⚠️ Common Issues

### Issue 1: "ModuleNotFoundError"

**Solution:** Make sure `PYTHONPATH` is set correctly:
```bash
export PYTHONPATH=/storage/kfir/repos/generative-audio-workspace:$PYTHONPATH
```

### Issue 2: "FileNotFoundError: [Errno 2] No such file or directory: 'ffmpeg'"

**Solution:** Install ffmpeg (see Prerequisites section above)

### Issue 3: Matplotlib Display Error

**Solution:** Set matplotlib backend:
```bash
export MPLBACKEND=Agg
```

### Issue 4: Training Crashes During Validation

**Check:**
- Validation dataset path is correct
- Validation dataloader is properly configured
- Enough GPU memory available

### Issue 5: Process Terminated Unexpectedly

**Check logs for:**
```bash
tail -n 200 training_output.log
```

Look for:
- CUDA out of memory errors → Reduce batch size
- File not found errors → Check dataset paths
- Import errors → Check PYTHONPATH

---

## 🔄 Running Multiple Training Jobs

### Start Second Job in Different Directory

```bash
# Terminal 1: Regular NPPC
cd /storage/kfir/repos/generative-audio-workspace/nppc_audio/inpainting/scripts/train
export PYTHONPATH=/storage/kfir/repos/generative-audio-workspace:$PYTHONPATH
export MPLBACKEND=Agg
nohup python train_nppc_model.py > training_regular.log 2>&1 &

# Terminal 2: Latent NPPC (after changing config)
cd /storage/kfir/repos/generative-audio-workspace/nppc_audio/inpainting/scripts/train
export PYTHONPATH=/storage/kfir/repos/generative-audio-workspace:$PYTHONPATH
export MPLBACKEND=Agg
nohup python train_nppc_model.py > training_latent.log 2>&1 &
```

### Monitor Multiple Jobs

```bash
# View all Python processes
ps aux | grep python

# View specific logs
tail -f training_regular.log
tail -f training_latent.log
```

---

## 📊 Checking Training Progress

### Via Wandb (Recommended)

1. Get wandb link from logs:
```bash
grep "View run at" training_output.log
```

2. Open the URL in browser

3. Monitor:
   - Loss curves
   - Learning rate
   - Sample reconstructions
   - System metrics (GPU usage, etc.)

### Via Logs

```bash
# Watch live progress
tail -f training_output.log

# Count epochs completed
grep "Epoch" training_output.log | tail -1

# Check recent loss values
grep "Objective" training_output.log | tail -20
```

### Via Checkpoints

```bash
# List saved checkpoints
ls -lth /storage/kfir/data/inpainting/nppc_model/checkpoint/

# Most recent checkpoint
ls -lt /storage/kfir/data/inpainting/nppc_model/checkpoint/ | head -2
```

---

## 🎯 Quick Reference Commands

### Start Training
```bash
cd /storage/kfir/repos/generative-audio-workspace/nppc_audio/inpainting/scripts/train && \
export PYTHONPATH=/storage/kfir/repos/generative-audio-workspace:$PYTHONPATH && \
export MPLBACKEND=Agg && \
nohup python train_nppc_model.py > training_output.log 2>&1 &
```

### Monitor Training
```bash
tail -f training_output.log
```

### Check Status
```bash
ps aux | grep train_nppc_model.py
```

### Stop Training
```bash
kill -TERM <PID>
```

---

## 📝 Best Practices

1. **Always set environment variables** before running nohup
2. **Monitor initial logs** for first few minutes to catch early errors
3. **Save PID** after starting training
4. **Use descriptive log names** if running multiple experiments
5. **Check wandb dashboard** regularly during training
6. **Verify checkpoint saving** after first epoch completes
7. **Keep config files** organized (use git or save copies)

---

## 🆘 Emergency Recovery

If training crashes and you need to resume:

1. **Find latest checkpoint:**
```bash
ls -lt /storage/kfir/data/inpainting/nppc_model/checkpoint/ | head -5
```

2. **Update config to load checkpoint** (if trainer supports it)

3. **Check logs for error:**
```bash
tail -n 500 training_output.log
```

4. **Fix issue and restart training**

---

## ✅ Validation After Training

Once training completes, validate the model:

```bash
cd /storage/kfir/repos/generative-audio-workspace/nppc_audio/inpainting/scripts/validator

export PYTHONPATH=/storage/kfir/repos/generative-audio-workspace:$PYTHONPATH

# For regular NPPC
python validate_nppc_model.py

# For latent NPPC
python validate_nppc_latent_model.py

# For comparison
python validate_comparison.py
```

---

pkill -9 -f train_nppc_model.py && \
cd /storage/kfir/repos/generative-audio-workspace/nppc_audio/inpainting/scripts/train && \
export PYTHONPATH=/storage/kfir/repos/generative-audio-workspace:$PYTHONPATH && \
export MPLBACKEND=Agg && \
nohup python train_nppc_model.py > training_output.log 2>&1 & \
echo "Training restarted with PID $!"


**Last Updated:** December 2024

