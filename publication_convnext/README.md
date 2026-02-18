# ConvNeXt Publication Package

## Earthquake Precursor Detection using ConvNeXt: A Modern Convolutional Approach

**Status**: ✅ LOEO Validation Complete  
**Last Updated**: 6 February 2026

---

## 📁 Package Contents

```
publication_convnext/
├── README.md                          # This file
├── MANUSCRIPT_DRAFT.md                # Main paper draft
├── SUPPLEMENTARY_MATERIALS.md         # Supplementary information
├── METHODOLOGY.md                     # Detailed methodology
├── MODEL_ARCHITECTURE.md              # ConvNeXt architecture details
├── TRAINING_REPORT.md                 # Training results and analysis
├── COMPARISON_WITH_OTHER_MODELS.md    # VGG16 vs EfficientNet vs ConvNeXt
├── FIGURES_LIST.md                    # List of figures for paper
├── COVER_LETTER.md                    # Journal submission cover letter
├── REVIEWER_RESPONSE_TEMPLATE.md      # Template for reviewer responses
├── TARGET_JOURNALS.md                 # Recommended journals
├── HIGHLIGHTS.md                      # Paper highlights
├── ABSTRACT.md                        # Paper abstract
├── KEYWORDS.md                        # Keywords for indexing
└── scripts/
    ├── generate_convnext_figures.py   # Generate paper figures
    ├── generate_gradcam_convnext.py   # Grad-CAM visualizations
    ├── train_loeo_convnext.py         # LOEO validation
    └── evaluate_convnext.py           # Model evaluation
```

---

## 🎯 Research Highlights

1. **First application of ConvNeXt** for earthquake precursor detection from ULF geomagnetic signals
2. **Modern CNN architecture** incorporating Vision Transformer design principles
3. **Multi-task learning** for simultaneous magnitude and azimuth classification
4. **Comprehensive comparison** with VGG16, EfficientNet-B0, and Xception
5. **Rigorous validation** using LOEO 10-fold cross-validation

---

## 📊 Model Specifications

| Specification | Value |
|---------------|-------|
| Architecture | ConvNeXt-Tiny |
| Parameters | 28.6M |
| Input Size | 224×224×3 |
| Pretrained | ImageNet-1K |
| Framework | PyTorch 2.x |

---

## 📈 Final Results (LOEO 10-Fold Cross-Validation)

| Metric | Result | Status |
|--------|--------|--------|
| Magnitude Accuracy | **97.53% ± 0.96%** | ✅ Complete |
| Azimuth Accuracy | **69.30% ± 5.74%** | ✅ Complete |
| Best Fold (Mag) | 98.15% (Fold 5) | ✅ |
| Best Fold (Azi) | 82.00% (Fold 9) | ✅ |
| Worst Fold (Mag) | 95.56% (Fold 10) | ✅ |
| Worst Fold (Azi) | 57.78% (Fold 10) | ✅ |

---

## 🔬 Key Innovations

### 1. ConvNeXt Architecture Advantages
- **Patchify stem**: 4×4 non-overlapping convolution (like ViT)
- **Inverted bottleneck**: Efficient feature extraction
- **Large kernels**: 7×7 depthwise convolutions
- **Layer normalization**: Better training stability
- **GELU activation**: Smoother gradients

### 2. Multi-Task Learning Design
- Shared backbone for feature extraction
- Separate classification heads for magnitude and azimuth
- Weighted loss function (magnitude prioritized)

### 3. Training Optimizations
- AdamW optimizer with weight decay 0.05
- Cosine annealing scheduler
- Dropout 0.5 for regularization
- Class weighting for imbalanced data

---

## 📝 Publication Timeline

| Phase | Status | Date |
|-------|--------|------|
| Model Training | ✅ Complete | 5 Feb 2026 |
| LOEO Validation | ✅ Complete | 5 Feb 2026 |
| Documentation Update | ✅ Complete | 6 Feb 2026 |
| LOSO Validation | ⏳ Pending | TBD |
| Grad-CAM Analysis | ⏳ Pending | TBD |
| Manuscript Finalization | ⏳ Pending | TBD |
| Journal Submission | ⏳ Pending | TBD |

---

## 🚀 Quick Start

### 1. Check LOEO Results
```bash
# View final results
type loeo_convnext_results\loeo_convnext_final_results.json

# View report
type loeo_convnext_results\LOEO_CONVNEXT_REPORT.md
```

### 2. Generate Figures (After Training)
```bash
python publication_convnext/scripts/generate_convnext_figures.py
```

### 3. Generate Grad-CAM
```bash
python publication_convnext/scripts/generate_gradcam_convnext.py
```

---

## 📊 Model Comparison Summary

| Model | Parameters | Mag Acc (LOEO) | Azi Acc (LOEO) |
|-------|------------|----------------|----------------|
| VGG16 | 138M | 98.68% | 54.93% |
| EfficientNet-B0 | 5.3M | 97.53% ± 0.96% | 69.51% ± 5.65% |
| **ConvNeXt-Tiny** | **28.6M** | **97.53% ± 0.96%** | **69.30% ± 5.74%** |

---

## 📚 References

1. Liu, Z., et al. (2022). "A ConvNet for the 2020s." CVPR 2022.
2. Hayakawa, M., et al. (2015). "ULF/ELF electromagnetic phenomena for short-term earthquake prediction."
3. Previous work: VGG16 and EfficientNet-B0 for earthquake precursor detection.

---

## 👥 Authors

- Earthquake Prediction Research Team
- BMKG (Badan Meteorologi, Klimatologi, dan Geofisika)

---

## 📧 Contact

For questions about this publication package, please contact the research team.
