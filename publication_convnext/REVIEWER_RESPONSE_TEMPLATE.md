# Reviewer Response Template

## ConvNeXt for Earthquake Precursor Detection

---

## Response to Reviewer Comments

**Manuscript Title**: Earthquake Precursor Detection using ConvNeXt: A Modern Convolutional Approach for ULF Geomagnetic Signal Classification

**Manuscript ID**: [To be assigned]

**Date**: [Response date]

---

Dear Editor and Reviewers,

We thank the reviewers for their constructive comments and suggestions. We have carefully addressed each point and made the necessary revisions. Below, we provide detailed responses to each comment.

---

## Reviewer 1

### Comment 1.1: [Architecture Justification]
*"Why was ConvNeXt chosen over other modern architectures like Vision Transformers?"*

**Response:**
We chose ConvNeXt for several reasons:

1. **Efficiency**: ConvNeXt maintains the computational efficiency of CNNs while incorporating ViT design principles
2. **Inductive bias**: Convolutional operations provide useful inductive biases for spectrogram analysis
3. **Training stability**: Layer Normalization and GELU activation provide stable training
4. **Interpretability**: Grad-CAM visualizations are more straightforward with convolutional architectures

We have added a discussion of this choice in Section 2.3 of the revised manuscript.

### Comment 1.2: [Sample Size Concerns]
*"The sample sizes for rare classes (Large, Moderate) are very small. How reliable are the reported metrics?"*

**Response:**
We acknowledge this limitation and have added explicit discussion in Section 4.3:

> "While the high F1 scores indicate strong pattern recognition capabilities within the tested domain, we acknowledge that the limited sample sizes for rare magnitude classes (Large: n=28, Moderate: n=20) suggest these specific metrics should be interpreted with caution."

We have also added bootstrap confidence intervals for rare class metrics in the Supplementary Materials.

### Comment 1.3: [Cross-Validation Details]
*"Please provide more details on the LOEO and LOSO validation procedures."*

**Response:**
We have expanded Section 2.6 to include:
- Detailed description of event grouping for LOEO
- Station selection criteria for LOSO
- Statistical analysis of fold-wise results
- Comparison with random split validation

---

## Reviewer 2

### Comment 2.1: [Comparison with Previous Work]
*"How does ConvNeXt compare with the VGG16 and EfficientNet models from your previous work?"*

**Response:**
We have added a comprehensive comparison in Section 3.4 and Table 3:

| Model | Mag Acc | Azi Acc | Params | Inference |
|-------|---------|---------|--------|-----------|
| VGG16 | 98.68% | 54.93% | 138M | 45 ms |
| EfficientNet-B0 | 97.53% | 69.51% | 5.3M | 18 ms |
| **ConvNeXt-Tiny** | **97.53%** | **69.30%** | **28.6M** | **~30 ms** |

### Comment 2.2: [Grad-CAM Analysis]
*"The Grad-CAM visualizations should be discussed in more detail."*

**Response:**
We have expanded Section 3.5 to include:
- Detailed analysis of attention patterns for each magnitude class
- Comparison of attention patterns between ConvNeXt and EfficientNet
- Physical interpretation of the focused frequency regions
- Additional Grad-CAM examples in Supplementary Materials

### Comment 2.3: [Real-World Applicability]
*"How would this model perform in a real-time monitoring scenario?"*

**Response:**
We have added Section 4.4 discussing practical deployment considerations:
- Inference time analysis (CPU vs GPU)
- Memory requirements
- Integration with existing monitoring systems
- Confidence thresholds for operational use

---

## Reviewer 3

### Comment 3.1: [Normal Class Selection]
*"The 100% accuracy for Normal class detection seems suspicious. How were Normal samples selected?"*

**Response:**
We appreciate this important observation. We have added explicit disclosure in Section 4.3:

> "Normal samples were exclusively selected from geomagnetically quiet days (Kp index < 2). This creates favorable testing conditions that may not reflect operational deployment scenarios. Future validation should include moderate geomagnetic activity days (Kp 2-4) to provide more realistic performance estimates."

We have also added a recommendation for future work to include "noisy" Normal samples.

### Comment 3.2: [Data Leakage Verification]
*"How was data leakage prevented between training and test sets?"*

**Response:**
We have added Section S8.2 in Supplementary Materials detailing our data leakage prevention:

1. **Event-level split**: Data was split at the earthquake event level
2. **LOEO validation**: Confirms no temporal leakage
3. **LOSO validation**: Confirms no spatial leakage
4. **Performance consistency**: LOEO/LOSO results are consistent with random split

### Comment 3.3: [Reproducibility]
*"Please provide more details for reproducibility."*

**Response:**
We have added Section S6 in Supplementary Materials with:
- Complete environment specifications
- Random seed settings
- Hyperparameter configurations
- Link to code repository
- Instructions for reproducing results

---

## Summary of Changes

1. **Section 2.3**: Added justification for ConvNeXt architecture choice
2. **Section 2.6**: Expanded cross-validation methodology
3. **Section 3.4**: Added comprehensive model comparison table
4. **Section 3.5**: Expanded Grad-CAM analysis
5. **Section 4.3**: Added limitations discussion
6. **Section 4.4**: Added real-world applicability discussion
7. **Supplementary Materials**: Added S8 (Overfitting Analysis) and S9 (Limitations)

---

We believe these revisions address all reviewer concerns and significantly improve the manuscript. We thank the reviewers again for their valuable feedback.

Sincerely,

[Corresponding Author]

---

*Template to be customized based on actual reviewer comments*
