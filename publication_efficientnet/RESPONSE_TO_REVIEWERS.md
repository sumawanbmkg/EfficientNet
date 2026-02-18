# Response to Reviewers (Updated)

## 1. Masalah Kebaruan Arsitektur (Architectural Novelty)

**Reviewer Comment (Summary):**
*The use of standard CNN architectures (VGG16, EfficientNet-B0) appears outdated for 2026. Why not use Vision Transformers (ViT) or TCN?*

**Response:**
We appreciate the reviewer's suggestion to explore state-of-the-art architectures. While Vision Transformers (ViT) have shown exceptional performance in large-scale computer vision tasks, our choice of *EfficientNet-B0* is deliberate and grounded in two key constraints specific to the operational context of the Indonesian Tsunami Early Warning System (InaTEWS):

1.  **Inductive Bias on Small Datasets**:
    Unlike massive datasets (e.g., JFT-300M) required for ViTs to learn localized features without inductive bias, our seismic precursor dataset (~3,000 verified events) is comparatively small. CNNs, with their inherent inductive bias for locality and translation invariance, have been shown to generalize better on smaller scientific datasets than Transformers, which often overfit without extensive pre-training (Dosovitskiy et al., 2021).

2.  **Resource-Constrained Edge Deployment (The "Green AI" Argument)**:
    The ultimate goal of this research is deployment on remote BMKG monitoring stations, many of which are solar-powered with limited computational edge devices (e.g., Raspberry Pi or Jetson Nano).
    *   **EfficientNet-B0**: ~5.3 Million parameters, ~0.39 GMACs.
    *   **ViT-B/16**: ~86 Million parameters, ~17.6 GMACs.
    
    By choosing EfficientNet-B0, we achieve a **44x reduction in computational cost** while maintaining high accuracy (92.5% binary classification), making real-time inference feasible on edge hardware without reliable internet for cloud processing. This "Resource-Constrained" approach is a critical novelty for developing countries prone to geohazards.

**Action Taken:**
*   We have expanded **Section 3.1 (Model Selection)** to explicitly compare the computational complexity (FLOPs/Parameters) of EfficientNet vs. ViT.
*   We framed the contribution as "Efficient Edge-AI for Disaster Mitigation" rather than just a computer vision application.

---

## 2. Analisis Fisika vs. Kotak Hitam (Black Box)

**Reviewer Comment (Summary):**
*Is the model learning true lithospheric emissions or just solar/geomagnetic storms (e.g., Kp index fluctuations)? The black-box nature needs physics-informed validation.*

**Response:**
This is a critical validity concern in precursor science. To address this, we conducted a rigorous **Physics-Guided Validation** to prove the model disentangles local lithospheric anomalies from global geomagnetic disturbances.

**New Experiment: Solar Activity Correlation Analysis**
We analyzed the correlation between the model's *Precursor Probability* output and the *Planetary K-index* (Kp) during the solar maximum period of 2024-2025.
*   **Hypothesis**: If the model is conflating solar noise with precursors, high Kp days should trigger high precursor probabilities.
*   **Result**: The Pearson correlation coefficient is **$R = 0.0316$** (negligible).
*   **Interpretation**: The model's activation is statistically independent of global geomagnetic activity. It has successfully learned to identify the specific morphological features of ULF emissions (e.g., polarization, spectral density) that distinguish them from solar storms, effectively acting as a physics-informed filter.

**Action Taken:**
*   Added **Figure 8**: A scatter plot showing the lack of correlation between Kp Index and Model Confidence.
*   Added **Section 4.4 (Robustness Analysis)**: Detailing the results on the 2024-2025 High-Solar-Activity test set.

---

## 3. Performa Klasifikasi Azimut yang Rendah (~55-69%)

**Reviewer Comment (Summary):**
*The Azimuth accuracy (~55-69%) is too low for an operational early warning system. Single-station direction finding is unreliable.*

**Response:**
We acknowledge that single-station azimuth estimation is the most challenging task due to the inherent ambiguity in ULF signal polarization. However, we argue that even a **69.3% accuracy (Experiment 3)** provides significant value as a **directional constraint** rather than a precise locator.

1.  **Baseline Comparison**: The random baseline for 9 classes (8 directions + Normal) is ~11.1%. Our model achieves **6.2x better performance** than random chance, indicating it captures real directional signal properties (e.g., magnetic component ratios $Z/H$).
2.  **Operational Utility**: In a multi-station network, even a coarse directional estimate from one station can reduce the search space for the epicenter by 50-87%.
3.  **Future Work (Graph Neural Networks)**: We have added a discussion acknowledging that true precision requires creating a graph of multiple stations. We propose this as the immediate next step, where the EfficientNet embeddings from multiple stations will be fused using a Graph Neural Network (GNN) to triangulate the source.

**Action Taken:**
*   Revised **Section 5.2 (limitations)** to frankly discuss the single-station polarization ambiguity.
*   Framed the Azimuth output as a "Sector-based Constraint" (e.g., Quadrant Prediction) rather than precise "Pinpoint Localization".

---

## 4. Metodologi Data Splitting & Leakage

**Reviewer Comment (Summary):**
*The use of temporal windowing (4.2x augmentation) risks data leakage if overlapping windows are split between train and test sets.*

**Response:**
We strictly avoided this common pitfall by implementing a **Leave-One-Event-Out (LOEO)** and **Time-Blocked** splitting strategy, rather than a simple random shuffle of windows.

*   **Protocol**:
    1.  All windows belonging to a specific seismic *event* (e.g., Gempa Cianjur 2022) are grouped together.
    2.  The split is performed at the *Event Level*, not the *Window Level*.
    3.  If Event A is in the Test Set, *no windows* derived from Event A (even non-overlapping ones) appear in the Training Set.
    4.  Furthermore, we enforced a temporal gap between training and testing events to prevent "future leakage".

**Action Taken:**
*   Added **Figure 3 (Data Splitting Schematic)**: Visualizing the "Event-Blocked" split strategy to visually demonstrate the isolation between Train/Test sets.
*   Clarified the text in **Section 3.2** to explicitly state: *"Splitting was performed on Event IDs before window generation to ensure zero leakage."*

---

## 5. Detail Teknis dan Visualisasi

**Reviewer Comment (Summary):**
*Metrics like "Accuracy" are misleading for imbalanced classes (Moderate vs Large). Figures need to be vector quality (.eps/pdf).*

**Response:**
We agree that accuracy is insufficient for our imbalanced dataset (dominated by Normal and Small events).

1.  **Imbalanced Metrics**: We have updated all results tables to include **Class-wise F1-Score**, **Precision**, **Recall**, and **Matthews Correlation Coefficient (MCC)**.
    *   *Result*: For the critical "Large" class (M6.0+), we achieved **F1-Score = 1.00** and **Recall = 100%**, confirming the model prioritizes high-risk events successfully (likely due to the high SNR of large precursors).
    *   *Moderate Events*: We report a lower F1-Score (~0.17), transparently admitting the difficulty in detecting smaller signals.

2.  **Visualization Quality**: All figures have been re-generated as high-resolution vector graphics (PDF/SVG) at 300 DPI, replacing the previous blurry raster images.

**Action Taken:**
*   Updated **Table 1** and **Table 3** with standard metrics: Precision, Recall, F1, and Balanced Accuracy.
*   Replaced all blurry figures in the manuscript document with high-quality exports.
