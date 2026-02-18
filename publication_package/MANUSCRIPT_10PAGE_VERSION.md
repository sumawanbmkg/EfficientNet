# Deep Learning-Based Earthquake Precursor Detection: VGG16 vs EfficientNet Comparison

**Target: ≤10 halaman (≤3000 kata)**

---

## Abstract (~200 kata)

Earthquake prediction remains challenging in geophysics. This study presents a deep learning approach for earthquake precursor detection from geomagnetic spectrogram data, comparing VGG16 and EfficientNet-B0 architectures for multi-task magnitude (4 classes) and azimuth (9 classes) prediction. Our dataset comprises 256 unique earthquake events (M4.0-7.0+) from Indonesian geomagnetic stations (2018-2025), generating 1,972 samples through temporal windowing. VGG16 achieved 98.68% magnitude and 54.93% azimuth accuracy, while EfficientNet-B0 achieved 94.37% magnitude and 57.39% azimuth accuracy with 26× fewer parameters. Leave-One-Event-Out validation confirmed robust generalization (<5% performance drop). Grad-CAM visualizations revealed focus on ULF bands (0.001-0.01 Hz), consistent with precursor theory. EfficientNet-B0 offers optimal balance for real-time deployment.

**Keywords**: Earthquake precursor, Deep learning, Geomagnetic data, VGG16, EfficientNet, Multi-task learning

---

## 1. Introduction (~400 kata)

Earthquake prediction is a grand challenge in geophysics. Among precursor signals, Ultra-Low Frequency (ULF) geomagnetic anomalies show promising correlations with seismic activity (Hayakawa et al., 2015). These signals (0.001-1 Hz) originate from stress-induced electromagnetic emissions before earthquakes.

Deep learning advances enable automated precursor detection. Previous studies applied CNNs to spectrogram representations with varying success. However, comprehensive comparisons between architectures for this application remain limited.

This study compares VGG16 (Simonyan & Zisserman, 2014) and EfficientNet-B0 (Tan & Le, 2019) for earthquake precursor detection. VGG16 represents classical deep CNN architecture, while EfficientNet uses compound scaling for efficiency. Our contributions:

1. First comprehensive VGG16 vs EfficientNet comparison for geomagnetic precursor detection
2. Multi-task learning for simultaneous magnitude and azimuth prediction
3. Rigorous LOEO validation ensuring generalization to unseen events
4. Grad-CAM analysis demonstrating physically meaningful feature learning

---

## 2. Dataset and Methods (~600 kata)

### 2.1 Dataset

Data comprises ULF geomagnetic recordings from 25 Indonesian BMKG stations. We selected 256 earthquake events (M4.0-7.0+, 2018-2025):
- Moderate (M4.0-4.9): 20 events
- Medium (M5.0-5.9): 1,036 samples  
- Large (M6.0+): 28 samples
- Normal (non-precursor): 888 samples

Total: 1,972 spectrogram samples after temporal windowing (4.2× factor).

### 2.2 Preprocessing

Raw geomagnetic data (H, D, Z components) processed:
1. Bandpass filtering: 0.001-0.5 Hz
2. Hourly segmentation: 6-hour windows before events
3. STFT spectrogram generation
4. Resize to 224×224 pixels
5. ImageNet normalization

### 2.3 Model Architectures

**VGG16**: 16-layer CNN with 138M parameters (528 MB). Multi-task head: FC(4096→512→4) for magnitude, FC(4096→512→9) for azimuth.

**EfficientNet-B0**: Compound-scaled CNN with 5.3M parameters (20 MB). Multi-task head: FC(1280→512→4/9) with dropout 0.444.

### 2.4 Training

- Optimizer: Adam (VGG16: lr=1e-4, EfficientNet: lr=9.89e-4)
- Loss: Focal Loss (γ=2) for class imbalance
- Batch size: 32
- Early stopping: 10 epochs patience
- Data split: 70/15/15 (train/val/test), fixed seed

### 2.5 Validation

Leave-One-Event-Out (LOEO) 10-fold cross-validation ensures no event overlap between train/test sets, validating true generalization.

---

## 3. Results (~600 kata)

### 3.1 Performance Comparison

| Model | Mag Acc | Azi Acc | Params | Size | Speed |
|-------|---------|---------|--------|------|-------|
| VGG16 | **98.68%** | 54.93% | 138M | 528 MB | 125 ms |
| EfficientNet-B0 | 94.37% | **57.39%** | 5.3M | 20 MB | 50 ms |

VGG16 achieves higher magnitude accuracy (+4.31%), while EfficientNet-B0 shows better azimuth accuracy (+2.46%) with 26× smaller model and 2.5× faster inference.

### 3.2 LOEO Validation

| Method | Magnitude | Azimuth |
|--------|-----------|---------|
| Random Split | 98.68% | 54.93% |
| LOEO (10-fold) | 94.23% ± 2.1% | 52.18% ± 3.4% |
| Performance Drop | 4.45% | 2.75% |

Both drops <5%, confirming robust generalization to unseen earthquake events.

### 3.3 Per-Class Analysis

**Magnitude Classification:**
- Normal: 100% (both models)
- Medium: 98.5% (VGG16), 94.2% (EfficientNet)
- Large: 96.4% (VGG16), 92.9% (EfficientNet)
- Moderate: 85.0% (VGG16), 80.0% (EfficientNet)

**Azimuth Classification:**
- N direction: 72.3% (VGG16), 75.8% (EfficientNet)
- Adjacent directions show confusion (expected for 9-class task)

### 3.4 Grad-CAM Analysis

Both models focus on:
1. ULF frequency bands (0.001-0.01 Hz) - consistent with precursor theory
2. Temporal evolution patterns in 6-hour windows
3. Magnitude-dependent signal intensity

This confirms physically meaningful feature learning rather than spurious correlations.

---

## 4. Discussion (~400 kata)

### 4.1 Architecture Trade-offs

VGG16 achieves maximum accuracy (98.68%) but requires significant resources (528 MB, 125 ms). EfficientNet-B0 offers 94.37% accuracy with 26× smaller footprint, suitable for edge deployment and real-time monitoring.

### 4.2 Azimuth Challenge

Both models show lower azimuth accuracy (~55%) compared to magnitude (~97%). This reflects the inherent difficulty of 9-class directional classification from geomagnetic signals. However, 55% significantly exceeds random baseline (11.1%), demonstrating meaningful directional learning.

### 4.3 Temporal Windowing

Our 4.2× windowing factor is consistent with literature (Han 2020: 4×, Akhoondzadeh 2022: 4×). LOEO validation confirms this captures genuine temporal evolution rather than causing overfitting.

### 4.4 Limitations

1. Limited Large/Moderate class samples (28/20 events)
2. Regional specificity (Indonesia)
3. Azimuth accuracy requires improvement
4. Normal class selection criteria (Kp < 2) may be optimistic

### 4.5 Deployment Recommendation

For production systems, EfficientNet-B0 is recommended due to:
- Acceptable accuracy (94.37% magnitude)
- Small footprint (20 MB)
- Fast inference (50 ms)
- Suitable for real-time monitoring

---

## 5. Conclusion (~200 kata)

This study compared VGG16 and EfficientNet-B0 for earthquake precursor detection from geomagnetic spectrograms. Key findings:

1. VGG16 achieves 98.68% magnitude accuracy (best overall)
2. EfficientNet-B0 achieves 94.37% with 26× smaller model
3. LOEO validation confirms <5% generalization drop
4. Grad-CAM reveals physically meaningful ULF band focus

EfficientNet-B0 offers optimal balance for operational early warning systems, enabling real-time precursor detection with minimal computational resources.

**Future work**: Expand dataset with more Large events, improve azimuth classification, and deploy real-time monitoring system.

---

## References (~20 citations)

1. Hayakawa, M., et al. (2015). ULF/ELF electromagnetic phenomena for earthquake prediction. PEPI.
2. Tan, M., & Le, Q. (2019). EfficientNet: Rethinking model scaling. ICML.
3. Simonyan, K., & Zisserman, A. (2014). Very deep convolutional networks. arXiv.
4. Han, P., et al. (2020). Machine learning for earthquake prediction. JGR.
5. Akhoondzadeh, M. (2022). CNN-based earthquake precursor detection. NHESS.
[... 15 more references ...]

---

## Word Count Summary

| Section | Words |
|---------|-------|
| Abstract | 200 |
| Introduction | 400 |
| Dataset & Methods | 600 |
| Results | 600 |
| Discussion | 400 |
| Conclusion | 200 |
| **Total** | **~2400** |

**Estimasi halaman**: ~8 halaman (single column, 300 kata/hal)

✅ **SESUAI DENGAN BATAS 10 HALAMAN**

---

## Figures (4-5 figures)

1. **Fig 1**: System architecture diagram
2. **Fig 2**: Confusion matrices (VGG16 vs EfficientNet)
3. **Fig 3**: Grad-CAM comparison
4. **Fig 4**: Model comparison chart (accuracy vs size)
5. **Fig 5**: LOEO validation results (optional)

## Tables (3-4 tables)

1. **Table 1**: Dataset statistics
2. **Table 2**: Model comparison
3. **Table 3**: LOEO validation results
4. **Table 4**: Per-class performance (optional, bisa di supplementary)

---

## Catatan untuk Penyesuaian

Jika masih melebihi 10 halaman setelah formatting:

**Opsi pemangkasan tambahan:**
1. Pindahkan per-class analysis ke Supplementary Materials
2. Kurangi Related Work (gabung ke Introduction)
3. Ringkas Discussion menjadi 3 poin utama
4. Batasi references ke 15-20 citations

**Opsi jika jurnal mengizinkan Supplementary:**
- Pindahkan detail metodologi ke Supplementary
- Pindahkan ablation studies ke Supplementary
- Pindahkan extended Grad-CAM analysis ke Supplementary
