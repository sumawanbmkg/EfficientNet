# Response to Reviewers Template

## Manuscript: Earthquake Precursor Detection using Deep Learning

---

Dear Editor and Reviewers,

We thank the reviewers for their constructive comments and suggestions. We have carefully addressed all concerns and revised the manuscript accordingly. Below we provide point-by-point responses to each comment.

---

## Response to Reviewer 1

### Comment 1.1: [Paste reviewer comment here]

**Response:** [Your response]

**Changes made:** [Describe changes, with page/line numbers]

---

### Comment 1.2: [Paste reviewer comment here]

**Response:** [Your response]

**Changes made:** [Describe changes]

---

## Response to Reviewer 2

### Comment 2.1: [Paste reviewer comment here]

**Response:** [Your response]

**Changes made:** [Describe changes]

---

## Common Anticipated Questions and Prepared Responses

### Q1: Why not use more recent architectures like Vision Transformers?

**Prepared Response:** We focused on VGG16 and EfficientNet-B0 for several reasons: (1) VGG16 remains widely used in geoscience applications due to its interpretability and well-understood behavior; (2) EfficientNet-B0 represents the state-of-the-art in efficient CNN design; (3) our dataset size (1,972 samples) is better suited for CNN architectures than data-hungry transformers; (4) the transfer learning approach with ImageNet pre-training is well-established for these architectures. Future work will explore transformer-based models as dataset size increases.

### Q2: How do you address the class imbalance, especially for Major earthquakes?

**Prepared Response:** We addressed class imbalance through: (1) inverse frequency class weighting during training; (2) stratified sampling to maintain class proportions across train/val/test splits; (3) evaluation using macro-averaged metrics that give equal weight to all classes; (4) per-class analysis to identify potential biases. While the Major class (n=13) has limited samples, the model achieves 92% F1-score, suggesting reasonable generalization despite the imbalance.

### Q3: Can the model generalize to other geographic regions?

**Prepared Response:** This is an important limitation we acknowledge in Section 5.5. The current model was trained exclusively on Indonesian data, and generalization to other tectonic settings requires validation. However, the physical basis of ULF precursors (LAI coupling) is universal, suggesting potential transferability. We recommend fine-tuning on local data when deploying in new regions.

### Q4: What is the false positive rate in operational settings?

**Prepared Response:** On our test set, both models achieve 100% accuracy in distinguishing precursor signals from normal conditions (0% false positive rate for normal class). However, operational false positive rates depend on factors not captured in our controlled study, including solar activity, anthropogenic noise, and equipment malfunctions. We recommend implementing confidence thresholds and multi-station confirmation for operational deployment.

### Q5: Why is azimuth accuracy lower than magnitude accuracy?

**Prepared Response:** The lower azimuth accuracy (54-57% vs. 94-99%) reflects the inherent difficulty of determining earthquake direction from single-station data. Factors include: (1) signal propagation effects from local geology; (2) potential overlap of precursor signatures from different directions; (3) the 8-class azimuth problem is more challenging than 4-class magnitude. Future work will explore multi-station fusion to improve directional accuracy.

### Q6: Is the model overfitting? The training curves show a gap between training and validation loss.

**Prepared Response:** We conducted comprehensive overfitting analysis (see Supplementary Materials S8). While training curves show moderate overfitting tendency (loss gap of 2.6), our rigorous cross-validation demonstrates excellent generalization:

- **LOEO (Leave-One-Event-Out):** 97.53% ± 0.96% magnitude accuracy, confirming temporal generalization to unseen earthquake events
- **LOSO (Leave-One-Station-Out):** 97.57% weighted magnitude accuracy, confirming spatial generalization to unseen geographic stations
- **Performance drop from random split:** Only ~1.4%, indicating no significant data leakage

The low variance in LOEO (std = 0.96%) and consistent performance across all stations in LOSO provide strong evidence that the model has learned generalizable features rather than memorizing training data.

### Q7: How do you ensure there is no data leakage between training and test sets?

**Prepared Response:** We implemented rigorous data splitting protocols:

1. **Event-level split:** All spectrograms from a single earthquake event appear in only one subset (train, validation, or test), preventing temporal leakage
2. **LOEO validation:** Leave-One-Event-Out cross-validation explicitly tests generalization to completely unseen events
3. **LOSO validation:** Leave-One-Station-Out cross-validation verifies the model doesn't memorize station-specific characteristics

Evidence of no leakage: LOEO and LOSO results (97.53% and 97.57%) are nearly identical to random split validation (98.94%), with only ~1.4% drop. If data leakage existed, we would expect significantly larger performance degradation (typically >10-20%).

### Q8: The per-class metrics for rare classes (Large, Major) may not be statistically reliable due to small sample sizes.

**Prepared Response:** We acknowledge this limitation in Supplementary Materials S8.3 and S9.1. With only 28 Large and 20 Moderate samples, per-class metrics have limited statistical power. We address this by:

1. Reporting weighted metrics that reflect overall performance
2. Using inverse frequency class weighting during training
3. Applying SMOTE augmentation for minority classes
4. Clearly stating this limitation in the manuscript
5. Providing confidence intervals for rare class metrics

We explicitly state: "While the high F1 scores indicate strong pattern recognition capabilities within the tested domain, we acknowledge that the limited sample sizes for rare magnitude classes suggest these specific metrics should be interpreted with caution."

Future work will focus on collecting additional samples from larger magnitude events to improve statistical reliability.

### Q9: The 100% normal detection accuracy seems suspicious. Is the model learning artifacts?

**Prepared Response:** This is a valid concern we address in Supplementary Materials S8.6 and S9.3. We provide honest disclosure:

> "The 100% Normal detection accuracy, while technically correct on our test set, should not be interpreted as perfect real-world performance. The strict selection criteria for Normal samples (Kp < 2) created favorable testing conditions that may not reflect operational deployment scenarios."

The high accuracy may be influenced by:
1. Normal samples were selected from geomagnetically quiet days (Kp < 2)
2. Significant difference in H-component mean values (Normal: 40,207 nT vs. Precursor: 38,251 nT)

We recommend future validation with moderate activity days (Kp 2-4) to provide more realistic performance estimates (expected 95-98% accuracy).

### Q10: Can you show examples where the model fails? This would demonstrate the model is not a "black box."

**Prepared Response:** Yes, we provide comprehensive failure analysis in Supplementary Materials S9.2. We document:

1. **Magnitude misclassification patterns**: Adjacent class confusion (e.g., Large→Medium) occurs in ~5% of cases, typically for borderline magnitudes near class boundaries.

2. **Azimuth misclassification patterns**: Adjacent direction confusion (45° angular proximity) accounts for most azimuth errors.

3. **Grad-CAM visualizations**: For each failure case, we show where the model focused attention, demonstrating physically interpretable behavior even in error cases.

4. **Root cause analysis**: We identify noise interference, signal amplitude ambiguity, and wave propagation complexity as primary error sources.

This transparency demonstrates our commitment to scientific rigor and shows the model behaves predictably even when incorrect.

### Q11: What are the main limitations of this study?

**Prepared Response:** We provide comprehensive limitations in Supplementary Materials S9:

1. **Statistical limitations**: Rare classes (Large n=28, Moderate n=20) have limited statistical power
2. **Normal class bias**: Selection from quiet days only may inflate accuracy
3. **Single-station limitation**: Azimuth estimation from single station is inherently challenging
4. **Geographic scope**: Model trained only on Indonesian data
5. **Confidence calibration**: Low-confidence predictions (<70%) show overconfidence

We believe acknowledging these limitations strengthens rather than weakens the paper, demonstrating scientific maturity and providing clear directions for future work.

---

## Summary of Major Revisions

1. Added comprehensive Limitations and Failure Analysis section (S9)
2. Included honest disclosure about 100% Normal detection accuracy
3. Provided Grad-CAM visualizations for misclassified samples
4. Added statistical power analysis for rare classes
5. Documented root causes of misclassification patterns

---

We believe these revisions have significantly strengthened the manuscript and addressed all reviewer concerns. We hope the revised version is now suitable for publication in [Journal Name].

Sincerely,
[Authors]
