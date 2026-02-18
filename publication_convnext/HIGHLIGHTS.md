# Research Highlights

## ConvNeXt for Earthquake Precursor Detection

**Last Updated**: 6 February 2026  
**Status**: ✅ LOEO Validation Complete

---

## Paper Highlights (for journal submission)

- First application of ConvNeXt architecture for earthquake precursor detection from ULF signals
- Multi-task learning framework for simultaneous magnitude and azimuth classification
- **97.53% ± 0.96%** magnitude accuracy via LOEO 10-fold cross-validation
- **69.30% ± 5.74%** azimuth accuracy, comparable to EfficientNet-B0
- Comprehensive comparison with VGG16, EfficientNet-B0 architectures
- Modern CNN design incorporating Vision Transformer principles for improved feature extraction

---

## Key Findings

### 1. Architecture Innovation
ConvNeXt-Tiny successfully adapts modern CNN design principles (patchify stem, 7×7 kernels, Layer Normalization, GELU activation) for geophysical signal classification.

### 2. Performance Results (LOEO 10-Fold Cross-Validation)

| Metric | Result |
|--------|--------|
| Magnitude Classification | **97.53% ± 0.96%** |
| Azimuth Classification | **69.30% ± 5.74%** |
| Best Fold (Combined) | **90.00%** (Fold 9) |
| Worst Fold (Combined) | **76.67%** (Fold 10) |

**MCC Analysis (vs Random Guessing):**
| Task | Random Baseline | Model Accuracy | Improvement | Est. MCC |
|------|-----------------|----------------|-------------|----------|
| Magnitude (4 classes) | 25.00% | 97.53% | 3.9x | ~0.97 |
| Azimuth (9 classes) | 11.11% | 69.30% | **6.2x** | **~0.69** |

*Note: Azimuth accuracy of 69.30% represents 6.2x improvement over random guessing, with MCC ≈ 0.69 indicating substantial predictive capability.*

### 3. Model Comparison

| Model | Parameters | Mag Acc (LOEO) | Azi Acc (LOEO) |
|-------|------------|----------------|----------------|
| VGG16 | 138M | 98.68% | 54.93% |
| EfficientNet-B0 | 5.3M | 97.53% ± 0.96% | 69.51% ± 5.65% |
| **ConvNeXt-Tiny** | **28.6M** | **97.53% ± 0.96%** | **69.30% ± 5.74%** |

ConvNeXt offers a modern alternative to established architectures with:
- Identical magnitude accuracy to EfficientNet-B0
- Comparable azimuth accuracy (difference not significant)
- Larger receptive field than VGG16 and EfficientNet
- Better training stability through Layer Normalization
- Moderate model size (28.6M parameters, 112 MB)

### 4. Interpretability
Grad-CAM analysis reveals physically meaningful attention patterns:
- Low-frequency focus for magnitude classification
- Phase relationship attention for azimuth estimation

### 5. Practical Implications
Results support the potential of modern CNN architectures for:
- Earthquake early warning systems
- Automated geomagnetic signal analysis
- Real-time precursor monitoring

---

## Graphical Abstract Elements

```
┌────────────────────────────────────────────────────────────────┐
│                                                                │
│   ULF Geomagnetic Signal → Spectrogram → ConvNeXt → Prediction │
│                                                                │
│   ┌─────────┐     ┌─────────┐     ┌─────────┐     ┌─────────┐ │
│   │  H,D,Z  │ →   │  STFT   │ →   │ConvNeXt │ →   │Magnitude│ │
│   │ Signal  │     │224×224  │     │  Tiny   │     │+ Azimuth│ │
│   └─────────┘     └─────────┘     └─────────┘     └─────────┘ │
│                                                                │
│   Key Innovation: Modern CNN with ViT Design Principles        │
│   • Patchify stem (4×4)                                       │
│   • 7×7 depthwise convolutions                                │
│   • Layer Normalization                                       │
│   • GELU activation                                           │
│                                                                │
│   Results: Mag 97.53%±0.96% | Azi 69.30%±5.74% | LOEO 10-fold │
│                                                                │
└────────────────────────────────────────────────────────────────┘
```

---

## Social Media Summary

### Twitter/X (280 chars)
🌍 New research: First application of #ConvNeXt for earthquake precursor detection! Modern CNN architecture achieves 97.53% magnitude accuracy on ULF geomagnetic signals via LOEO cross-validation. #DeepLearning #Seismology #EarthquakePrediction

### LinkedIn Summary
Excited to share our latest research on earthquake precursor detection using ConvNeXt architecture. This modern CNN design, incorporating Vision Transformer principles, achieves 97.53% ± 0.96% magnitude classification accuracy through rigorous LOEO 10-fold cross-validation. Key innovations include patchify stem, 7×7 kernels, and Layer Normalization for improved feature extraction from ULF geomagnetic spectrograms.

---

## Press Release Points

1. **What**: First use of ConvNeXt (modern CNN) for earthquake precursor detection
2. **Why**: Improve early warning systems using advanced AI
3. **How**: Analyze ULF geomagnetic signals before earthquakes
4. **Results**: 97.53% accuracy in predicting earthquake magnitude
5. **Impact**: Potential for real-time earthquake monitoring systems

---

## Keywords for Indexing

**Primary Keywords:**
- ConvNeXt
- Earthquake precursor
- ULF geomagnetic signals
- Deep learning
- Multi-task learning
- LOEO cross-validation

**Secondary Keywords:**
- Spectrogram classification
- Seismology
- Early warning system
- Convolutional neural network
- Transfer learning

**Geographic Keywords:**
- Indonesia
- Southeast Asia
- Pacific Ring of Fire

---

*Highlights finalized with LOEO validation results - 6 February 2026*
