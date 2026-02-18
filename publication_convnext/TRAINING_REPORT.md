# Training Report

## ConvNeXt-Tiny Training for Earthquake Precursor Detection

**Status**: ✅ LOEO Validation Complete  
**Started**: 5 February 2026  
**Completed**: 5 February 2026 22:51  
**Last Updated**: 6 February 2026

---

## 1. Training Configuration

### 1.1 Model Specifications

| Parameter | Value |
|-----------|-------|
| Architecture | ConvNeXt-Tiny |
| Pretrained | ImageNet-1K |
| Total Parameters | 28,615,789 |
| Trainable Parameters | 28,615,789 |
| Input Size | 224 × 224 × 3 |

### 1.2 Training Hyperparameters

| Parameter | Value |
|-----------|-------|
| Optimizer | AdamW |
| Learning Rate | 1e-4 (0.0001) |
| Weight Decay | 0.05 |
| Batch Size | 16 |
| Max Epochs | 15 |
| Early Stopping | Yes (patience 5) |
| LR Schedule | Cosine annealing |
| Dropout | 0.5 |

### 1.3 Dataset Split (LOEO 10-Fold)

| Split | Events | Samples (avg) |
|-------|--------|---------------|
| Training | 270-271 | ~1,772 |
| Test | 30-31 | ~195 |
| **Total Events** | **301** | **1,972** |

---

## 2. LOEO Cross-Validation Results

### 2.1 Summary Statistics

| Metric | Mean | Std Dev | Min | Max |
|--------|------|---------|-----|-----|
| **Magnitude Accuracy** | **97.53%** | ±0.96% | 95.56% | 98.15% |
| **Azimuth Accuracy** | **69.30%** | ±5.74% | 57.78% | 82.00% |
| **Combined Accuracy** | **83.41%** | ±3.35% | 76.67% | 90.00% |

### 2.2 Per-Fold Results

| Fold | Mag Acc | Azi Acc | Combined | Train Events | Test Events | Train Samples | Test Samples |
|------|---------|---------|----------|--------------|-------------|---------------|--------------|
| 1 | 95.65% | 67.39% | 81.52% | 270 | 31 | 1,788 | 184 |
| 2 | 97.83% | 67.39% | 82.61% | 271 | 30 | 1,788 | 184 |
| 3 | 98.00% | 72.00% | 85.00% | 271 | 30 | 1,772 | 200 |
| 4 | 98.04% | 70.59% | 84.31% | 271 | 30 | 1,768 | 204 |
| 5 | 98.15% | 66.67% | 82.41% | 271 | 30 | 1,756 | 216 |
| 6 | 98.00% | 72.00% | 85.00% | 271 | 30 | 1,772 | 200 |
| 7 | 98.00% | 70.00% | 84.00% | 271 | 30 | 1,772 | 200 |
| 8 | 98.04% | 67.16% | 82.60% | 271 | 30 | 1,768 | 204 |
| 9 | **98.00%** | **82.00%** | **90.00%** | 271 | 30 | 1,772 | 200 |
| 10 | 95.56% | 57.78% | 76.67% | 271 | 30 | 1,792 | 180 |

### 2.3 Best and Worst Folds

**Best Performance (Fold 9):**
- Magnitude: 98.00%
- Azimuth: 82.00% (highest)
- Combined: 90.00%

**Lowest Performance (Fold 10):**
- Magnitude: 95.56%
- Azimuth: 57.78% (lowest)
- Combined: 76.67%

---

## 3. Comparison with Other Models

### 3.1 LOEO Cross-Validation Comparison

| Model | Parameters | Mag Acc (Mean±Std) | Azi Acc (Mean±Std) |
|-------|------------|--------------------|--------------------|
| VGG16 | 138M | 98.68% | 54.93% |
| EfficientNet-B0 | 5.3M | 97.53% ± 0.96% | 69.51% ± 5.65% |
| **ConvNeXt-Tiny** | **28.6M** | **97.53% ± 0.96%** | **69.30% ± 5.74%** |

### 3.2 Performance Analysis

| Metric | VGG16 | EfficientNet-B0 | ConvNeXt-Tiny | Winner |
|--------|-------|-----------------|---------------|--------|
| Magnitude Accuracy | 98.68% | 97.53% | 97.53% | VGG16 |
| Azimuth Accuracy | 54.93% | 69.51% | 69.30% | EfficientNet |
| Model Size | 528 MB | 20 MB | 112 MB | EfficientNet |
| Parameters | 138M | 5.3M | 28.6M | EfficientNet |
| Consistency (Std) | N/A | ±0.96% | ±0.96% | Tie |

### 3.3 Key Observations

1. **Magnitude Classification**: ConvNeXt mencapai performa setara dengan EfficientNet-B0 (97.53%)
2. **Azimuth Classification**: Sedikit di bawah EfficientNet (69.30% vs 69.51%), perbedaan tidak signifikan
3. **Consistency**: Standard deviation identik (±0.96% untuk magnitude), menunjukkan stabilitas serupa
4. **Model Size**: ConvNeXt lebih besar dari EfficientNet (112 MB vs 20 MB) tapi lebih kecil dari VGG16

---

## 4. Training Observations

### 4.1 Convergence Characteristics

- Model konvergen dengan baik dalam 15 epochs
- Early stopping efektif mencegah overfitting
- Cosine annealing scheduler memberikan training yang stabil

### 4.2 Magnitude vs Azimuth Performance

- **Magnitude**: Sangat konsisten (95.56% - 98.15%), variance rendah
- **Azimuth**: Lebih bervariasi (57.78% - 82.00%), menunjukkan task yang lebih challenging

### 4.3 Fold Analysis

- Fold 9 menunjukkan performa azimuth terbaik (82%) - kemungkinan event yang lebih mudah diprediksi
- Fold 10 menunjukkan performa terendah - kemungkinan event dengan karakteristik unik/sulit

---

## 5. Model Artifacts

### 5.1 Output Files

| File | Description | Location |
|------|-------------|----------|
| fold_X_result.json | Per-fold results | loeo_convnext_results/ |
| loeo_convnext_final_results.json | Summary results | loeo_convnext_results/ |
| LOEO_CONVNEXT_REPORT.md | Markdown report | loeo_convnext_results/ |

### 5.2 Model Configuration

```python
CONFIG = {
    "model": "ConvNeXt-Tiny",
    "pretrained": "ImageNet-1K",
    "optimizer": "AdamW",
    "learning_rate": 0.0001,
    "weight_decay": 0.05,
    "batch_size": 16,
    "epochs": 15,
    "dropout": 0.5,
    "scheduler": "Cosine Annealing",
    "early_stopping": True,
    "n_folds": 10
}
```

---

## 6. Conclusions

### 6.1 Key Findings

1. ✅ ConvNeXt-Tiny berhasil divalidasi dengan LOEO 10-fold cross-validation
2. ✅ Magnitude accuracy sangat baik dan konsisten (97.53% ± 0.96%)
3. ✅ Azimuth accuracy comparable dengan EfficientNet-B0 (69.30% ± 5.74%)
4. ✅ Model tidak overfitting ke event tertentu (validasi LOEO membuktikan generalisasi)

### 6.2 Recommendations

1. **Production Use**: ConvNeXt dapat digunakan sebagai alternatif modern untuk EfficientNet
2. **Ensemble**: Kombinasi ConvNeXt + EfficientNet dapat meningkatkan robustness
3. **Future Work**: Eksplorasi ConvNeXt-Small/Base untuk potensi peningkatan performa

### 6.3 Publication Readiness

| Requirement | Status |
|-------------|--------|
| LOEO Validation | ✅ Complete |
| Statistical Analysis | ✅ Complete |
| Model Comparison | ✅ Complete |
| Reproducibility | ✅ Documented |

---

## 7. Technical Details

### 7.1 Validation Timestamp

```
Completed: 2026-02-05T22:51:20.307774
```

### 7.2 Hardware Used

- Device: CPU
- Training time per fold: ~30-45 minutes
- Total validation time: ~6-8 hours

---

*Training Report - ConvNeXt LOEO Validation Complete*
