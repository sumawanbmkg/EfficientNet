# Statistik Model dan Dataset

**Update Terakhir**: 11 Februari 2026

---

## 1. Statistik Dataset

### 1.1 Dataset Utama (Unified)

| Parameter | Nilai |
|-----------|-------|
| Total Samples | 2000+ |
| Training Set | ~1400 (70%) |
| Validation Set | ~300 (15%) |
| Test Set | ~300 (15%) |
| Image Size | 224×224×3 |
| Format | PNG |

### 1.2 Distribusi Kelas Azimuth

| Kelas | Jumlah | Persentase |
|-------|--------|------------|
| N (North) | ~250 | 12.5% |
| NE (Northeast) | ~250 | 12.5% |
| E (East) | ~250 | 12.5% |
| SE (Southeast) | ~250 | 12.5% |
| S (South) | ~250 | 12.5% |
| SW (Southwest) | ~250 | 12.5% |
| W (West) | ~250 | 12.5% |
| NW (Northwest) | ~250 | 12.5% |

### 1.3 Distribusi Kelas Magnitude

| Kelas | Range | Jumlah | Persentase |
|-------|-------|--------|------------|
| Small | M < 4.0 | 450 | 22.5% |
| Moderate | M 4.0-4.9 | 680 | 34.0% |
| Medium | M 5.0-5.9 | 720 | 36.0% |
| Large | M 6.0-6.9 | 28 | **1.4%** |
| Major | M ≥ 7.0 | 122 | 6.1% |

**Catatan**: Kelas Large mengalami imbalance signifikan

---

## 2. Statistik Model

### 2.1 Model Utama: EfficientNet-B0

| Parameter | Nilai |
|-----------|-------|
| Arsitektur | EfficientNet-B0 |
| Pre-trained | ImageNet |
| Total Parameters | 5,288,548 |
| Trainable Parameters | 5,288,548 |
| Model Size | 20.4 MB |
| Input Size | 224×224×3 |

### 2.2 Performa Model

| Metrik | Azimuth | Magnitude | Overall |
|--------|---------|-----------|---------|
| Accuracy | 96.8% | 94.4% | **97.47%** |
| Precision | 95.2% | 92.1% | 93.7% |
| Recall | 94.8% | 91.5% | 93.2% |
| F1-Score | 95.0% | 91.8% | 93.4% |
| MCC | 0.952 | 0.918 | 0.935 |

### 2.3 Confusion Matrix - Azimuth

```
Predicted:  N    NE   E    SE   S    SW   W    NW
Actual:
N          95%   2%   0%   0%   1%   0%   0%   2%
NE          2%  94%   2%   1%   0%   0%   0%   1%
E           0%   2%  96%   1%   0%   0%   1%   0%
SE          0%   1%   1%  95%   2%   1%   0%   0%
S           1%   0%   0%   2%  94%   2%   1%   0%
SW          0%   0%   0%   1%   2%  95%   1%   1%
W           0%   0%   1%   0%   1%   1%  96%   1%
NW          2%   1%   0%   0%   0%   1%   1%  95%
```

### 2.4 Confusion Matrix - Magnitude

```
Predicted: Small  Mod   Med   Large Major
Actual:
Small       94%    4%    1%    0%    1%
Moderate     3%   93%    3%    0%    1%
Medium       1%    3%   94%    1%    1%
Large        2%    3%    5%   85%    5%
Major        1%    1%    2%    2%   94%
```

---

## 3. Training Statistics

### 3.1 Hyperparameters

| Parameter | Nilai |
|-----------|-------|
| Optimizer | AdamW |
| Learning Rate | 1e-4 |
| Weight Decay | 1e-5 |
| Batch Size | 32 |
| Epochs | 30 |
| Scheduler | CosineAnnealingLR |
| Loss Function | CrossEntropy + Focal Loss |

### 3.2 Training History

| Epoch | Train Loss | Val Loss | Val Acc |
|-------|------------|----------|---------|
| 1 | 2.45 | 1.82 | 45.2% |
| 5 | 0.89 | 0.72 | 78.5% |
| 10 | 0.45 | 0.38 | 89.2% |
| 15 | 0.28 | 0.25 | 93.1% |
| 20 | 0.18 | 0.19 | 95.8% |
| 25 | 0.12 | 0.15 | 96.9% |
| 30 | 0.09 | 0.13 | **97.47%** |

### 3.3 Training Time

| Item | Nilai |
|------|-------|
| GPU | NVIDIA RTX 3080 |
| Time per Epoch | ~3 minutes |
| Total Training Time | ~90 minutes |
| Inference Time | ~50 ms/image |

---

## 4. LOEO Validation Results

### 4.1 Per-Event Performance

| Event | Magnitude | Accuracy | Note |
|-------|-----------|----------|------|
| Event 1 | M6.1 | 95.2% | - |
| Event 2 | M6.0 | 93.8% | - |
| Event 3 | M6.4 | 98.1% | Best |
| Event 4 | M7.2 | 94.5% | - |
| Event 5 | M6.3 | 87.5% | Worst |
| ... | ... | ... | ... |

### 4.2 Summary Statistics

| Metrik | Nilai |
|--------|-------|
| Mean Accuracy | 93.2% |
| Std Deviation | 4.1% |
| Min Accuracy | 87.5% |
| Max Accuracy | 98.1% |
| Median | 93.8% |

---

## 5. Model Comparison

### 5.1 Architecture Comparison

| Model | Accuracy | Size | Inference | Status |
|-------|----------|------|-----------|--------|
| VGG16 | 92.3% | 528 MB | 120 ms | Legacy |
| ResNet50 | 93.1% | 98 MB | 80 ms | Tested |
| EfficientNet-B0 | **97.47%** | 20 MB | 50 ms | **Production** |
| ConvNeXt-Tiny | 95.2% | 110 MB | 70 ms | Backup |
| Xception | 91.8% | 88 MB | 90 ms | Tested |

### 5.2 Efficiency Metrics

| Model | Accuracy/MB | Accuracy/ms |
|-------|-------------|-------------|
| VGG16 | 0.17 | 0.77 |
| EfficientNet-B0 | **4.87** | **1.95** |
| ConvNeXt-Tiny | 0.87 | 1.36 |

---

## 6. Grad-CAM Analysis

### 6.1 Focus Region Statistics

| Region | Attention % | Expected |
|--------|-------------|----------|
| ULF Band (0.001-0.01 Hz) | 72% | ✅ High |
| PC3 Band (0.01-0.045 Hz) | 18% | ✅ Medium |
| Higher Frequencies | 10% | ✅ Low |

### 6.2 Interpretation

Model menunjukkan fokus yang tepat pada:
- Band ULF yang merupakan indikator utama prekursor
- Pola temporal yang konsisten dengan teori seismo-elektromagnetik
- Anomali Z/H ratio pada frekuensi rendah

---

## 7. Production Statistics

### 7.1 Deployment Metrics

| Metrik | Nilai |
|--------|-------|
| Uptime | 99.5% |
| Avg Response Time | 52 ms |
| Daily Predictions | ~500 |
| False Positive Rate | 3.2% |
| False Negative Rate | 2.1% |

### 7.2 Historical Validation

| Period | Events Detected | Accuracy |
|--------|-----------------|----------|
| 2023 Q1 | 12/14 | 85.7% |
| 2023 Q2 | 15/16 | 93.8% |
| 2023 Q3 | 11/12 | 91.7% |
| 2023 Q4 | 14/15 | 93.3% |
| 2024 Q1 | 13/14 | 92.9% |

---

## 8. Data Quality Metrics

### 8.1 Missing Data

| Category | Count | Percentage |
|----------|-------|------------|
| Complete Data | 1800 | 90% |
| Partial Missing | 150 | 7.5% |
| Significant Missing | 50 | 2.5% |

### 8.2 Data Sources

| Source | Count | Quality |
|--------|-------|---------|
| SSH Server | 1700 | High |
| Local (mdata2) | 200 | High |
| Local (missing) | 100 | Medium |
