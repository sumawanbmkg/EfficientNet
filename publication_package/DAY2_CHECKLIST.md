# Day 2 Checklist - ConvNeXt Training

**Date**: 19 Februari 2026  
**Goal**: Prepare dataset & start ConvNeXt-Tiny training  
**Estimated Time**: 4-5 jam (+ overnight training)

---

## ☀️ MORNING SESSION (2-3 hours)

### ☕ Task 1: Read ConvNeXt Paper (30 menit)
- [ ] Download paper: Liu et al. (2022) "A ConvNet for the 2020s"
- [ ] Read Abstract, Introduction, Method (Sections 1-3)
- [ ] Note key architectural features:
  - [ ] Depthwise convolutions
  - [ ] Inverted bottleneck
  - [ ] LayerNorm instead of BatchNorm
  - [ ] GELU activation
- [ ] Prepare citation for paper

**Why**: Understanding architecture helps explain results in Section 3.5

---

### 🗂️ Task 2: Check Dataset Status (30 menit)
- [ ] Verify dataset exists:
  ```bash
  ls data/train/ data/val/ data/test/
  ```
- [ ] Check dataset structure:
  ```bash
  python check_dataset.py
  ```
- [ ] Verify class distribution:
  - [ ] Normal: ~888 samples
  - [ ] Medium: ~1,036 samples
  - [ ] Large: ~28 samples
  - [ ] Moderate: ~20 samples

**If dataset missing**:
```bash
python prepare_dataset.py --model convnext --augmentation moderate
```

---

### 🔧 Task 3: Configure Training (1 hour)
- [ ] Create training config file: `config_convnext.yaml`
  ```yaml
  model: convnext_tiny
  pretrained: true
  num_classes_magnitude: 4
  num_classes_azimuth: 9
  
  training:
    epochs: 50
    batch_size: 32
    learning_rate: 1e-4
    optimizer: adam
    loss: focal_loss
    early_stopping_patience: 10
  
  data:
    train_path: data/train/
    val_path: data/val/
    test_path: data/test/
    augmentation: moderate
  ```

- [ ] Test training script (1 epoch dry run):
  ```bash
  python train_convnext.py --config config_convnext.yaml --epochs 1 --dry-run
  ```

- [ ] Verify no errors:
  - [ ] Model loads correctly
  - [ ] Data loads correctly
  - [ ] Forward pass works
  - [ ] Loss computes correctly
  - [ ] Backward pass works

---

### 🧪 Task 4: Setup Monitoring (30 menit)
- [ ] Create logging directory:
  ```bash
  mkdir -p logs/convnext_training/
  ```

- [ ] Setup TensorBoard (optional):
  ```bash
  tensorboard --logdir logs/convnext_training/ --port 6006
  ```

- [ ] Create training log file:
  ```bash
  touch logs/convnext_training/training.log
  ```

- [ ] Test logging:
  ```bash
  python train_convnext.py --epochs 1 --log-dir logs/convnext_training/
  ```

---

## 🌆 AFTERNOON SESSION (2-3 hours)

### 🚀 Task 5: Start Full Training (30 menit setup)
- [ ] Final verification:
  - [ ] Dataset ready
  - [ ] Config file correct
  - [ ] Logging setup
  - [ ] Disk space available (>10 GB)

- [ ] Start training:
  ```bash
  python train_convnext.py \
    --config config_convnext.yaml \
    --epochs 50 \
    --batch-size 32 \
    --lr 1e-4 \
    --log-dir logs/convnext_training/ \
    > logs/convnext_training/training.log 2>&1 &
  ```

- [ ] Get process ID:
  ```bash
  echo $! > logs/convnext_training/pid.txt
  ```

**Expected Runtime**: 6-8 hours (overnight)

---

### 📊 Task 6: Monitor First Epochs (1 hour)
- [ ] Wait for first epoch to complete (~10-15 minutes)

- [ ] Check training log:
  ```bash
  tail -f logs/convnext_training/training.log
  ```

- [ ] Verify metrics:
  - [ ] Training loss decreasing
  - [ ] Validation loss reasonable
  - [ ] No NaN or Inf values
  - [ ] GPU/CPU utilization normal

- [ ] Check first epoch results:
  ```
  Epoch 1/50
  Train Loss: ~2.5-3.0 (expected)
  Val Loss: ~2.8-3.5 (expected)
  Train Acc: ~40-50% (expected)
  Val Acc: ~35-45% (expected)
  Time: ~10-15 min/epoch
  ```

**If issues**:
- Loss too high (>5.0): Check data normalization
- Loss NaN: Reduce learning rate to 1e-5
- Too slow (>30 min/epoch): Reduce batch size to 16

---

### 📝 Task 7: Document Configuration (30 menit)
- [ ] Create training documentation:
  ```markdown
  # ConvNeXt-Tiny Training Log
  
  **Start Time**: [timestamp]
  **Config**: config_convnext.yaml
  **Expected Completion**: [timestamp + 8 hours]
  
  ## Hyperparameters
  - Model: ConvNeXt-Tiny (28.6M params)
  - Epochs: 50
  - Batch Size: 32
  - Learning Rate: 1e-4
  - Optimizer: Adam
  - Loss: Focal Loss (γ=2)
  
  ## First Epoch Results
  - Train Loss: [value]
  - Val Loss: [value]
  - Train Acc: [value]
  - Val Acc: [value]
  - Time: [value] min
  
  ## Notes
  - [Any observations]
  ```

- [ ] Save to: `logs/convnext_training/README.md`

---

## 🌙 EVENING SESSION (1 hour)

### ✅ Task 8: Verification Before Bed (30 menit)
- [ ] Check training is still running:
  ```bash
  ps aux | grep train_convnext.py
  ```

- [ ] Check latest log entries:
  ```bash
  tail -20 logs/convnext_training/training.log
  ```

- [ ] Verify progress:
  - [ ] Multiple epochs completed (at least 3-5)
  - [ ] Loss trending downward
  - [ ] No errors or warnings
  - [ ] Checkpoints being saved

- [ ] Check disk space:
  ```bash
  df -h
  ```

- [ ] Estimate completion time:
  ```
  Time per epoch: [X] minutes
  Epochs remaining: [50 - current]
  Estimated completion: [calculation]
  ```

---

### 📚 Task 9: Prepare for Tomorrow (30 menit)
- [ ] Read `DAY3_CHECKLIST.md` (if exists)

- [ ] Review evaluation script:
  ```bash
  cat evaluate.py
  ```

- [ ] Prepare evaluation checklist:
  - [ ] Test set path
  - [ ] Metrics to compute
  - [ ] Figures to generate
  - [ ] Comparison with EfficientNet

- [ ] Set alarm for tomorrow:
  - [ ] Check training completion first thing

---

## 📊 EXPECTED OUTCOMES

### By End of Day 2
- [x] ✅ ConvNeXt paper read and understood
- [x] ✅ Dataset verified and ready
- [x] ✅ Training configuration tested
- [x] ✅ Full training started (running overnight)
- [x] ✅ First 3-5 epochs completed successfully
- [x] ✅ Monitoring and logging setup

### Overnight (6-8 hours)
- [ ] ⏳ Training continues (epochs 5-50)
- [ ] ⏳ Checkpoints saved every 5 epochs
- [ ] ⏳ Best model saved based on validation loss
- [ ] ⏳ Training log updated continuously

### Tomorrow Morning (Day 3)
- [ ] ⏳ Training completed (50 epochs)
- [ ] ⏳ Best model checkpoint available
- [ ] ⏳ Training curves generated
- [ ] ⏳ Ready for evaluation

---

## 🚨 TROUBLESHOOTING

### Issue 1: Dataset Not Found
```bash
# Check current directory
pwd

# List data directory
ls -la data/

# If missing, prepare dataset
python prepare_dataset.py --model convnext
```

### Issue 2: Out of Memory
```bash
# Reduce batch size
python train_convnext.py --batch-size 16  # or 8

# Or use gradient accumulation
python train_convnext.py --batch-size 16 --accumulation-steps 2
```

### Issue 3: Training Too Slow
```bash
# Check CPU/GPU usage
top  # or htop

# Reduce number of workers
python train_convnext.py --num-workers 2

# Or use smaller model
python train_convnext.py --model convnext_nano
```

### Issue 4: Training Crashes
```bash
# Check log for errors
tail -50 logs/convnext_training/training.log

# Restart from checkpoint
python train_convnext.py --resume logs/convnext_training/checkpoint_last.pth
```

### Issue 5: Loss Not Decreasing
```bash
# Reduce learning rate
python train_convnext.py --lr 1e-5

# Or use learning rate scheduler
python train_convnext.py --scheduler cosine
```

---

## 📞 QUICK COMMANDS

### Check Training Status
```bash
# Is training running?
ps aux | grep train_convnext.py

# Latest log entries
tail -f logs/convnext_training/training.log

# Current epoch
grep "Epoch" logs/convnext_training/training.log | tail -1

# Disk space
df -h
```

### Stop Training (if needed)
```bash
# Get process ID
cat logs/convnext_training/pid.txt

# Kill process
kill [PID]

# Or force kill
kill -9 [PID]
```

### Resume Training (if stopped)
```bash
# From last checkpoint
python train_convnext.py --resume logs/convnext_training/checkpoint_last.pth
```

---

## ✅ END OF DAY CHECKLIST

Before going to bed, verify:
- [ ] ✅ Training is running
- [ ] ✅ At least 3-5 epochs completed
- [ ] ✅ Loss is decreasing
- [ ] ✅ No errors in log
- [ ] ✅ Checkpoints being saved
- [ ] ✅ Disk space sufficient
- [ ] ✅ Estimated completion time noted

---

## 🎯 SUCCESS CRITERIA

### Minimum Success
- [x] ✅ Training started successfully
- [x] ✅ First epoch completed without errors
- [x] ✅ Monitoring setup working

### Target Success
- [x] ✅ 5+ epochs completed by end of day
- [x] ✅ Loss trending downward
- [x] ✅ Validation accuracy >50%

### Excellent Success
- [x] ✅ 10+ epochs completed by end of day
- [x] ✅ Validation accuracy >70%
- [x] ✅ Training curves look healthy

---

**Timeline**: Day 2 of 21  
**Progress**: 5% → 10% (estimated)  
**Next**: Day 3 - Evaluation & Comparison

---

*Good luck with training! Check progress tomorrow morning.*
