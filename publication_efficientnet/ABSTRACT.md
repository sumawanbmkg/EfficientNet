# Abstract: Enhancing Seismic Precursor Reliability via Hierarchical EfficientNet and Spectral Homogenization

**Keywords**: Earthquake Prediction, Deep Learning, EfficientNet, Z/H Ratio, Signal Homogenization, Hierarchical Classification.

### Background
Short-term earthquake forecasting remains a significant challenge due to the low signal-to-noise ratio in geomagnetic precursors and the temporal bias introduced by solar activity cycles. Previous attempts using standard deep learning architectures have often struggled with high false-positive rates and inconsistent performance across different earthquake magnitudes.

### Objectives
This study introduces a novel approach using a **Hierarchical EfficientNet (Phase 2.1)** architecture designed specifically to detect ULF (Ultra-Low Frequency) geomagnetic anomalies while minimizing the impact of solar cycle fluctuations. The primary goal was to maximize the recall for catastrophic seismic events (M6.0+) while maintaining overall system precision.

### Methodology
We utilized a homogenized dataset of **2,340 samples** derived from Z/H spectral ratios recorded across a multi-station network in Indonesia. The dataset was balanced using a combination of historical (2018) and modern (2024-2025) data to neutralize solar flux bias. The proposed architecture employs a two-stage hierarchical head:
1. **Detection Stage**: Binary classification between Normal (Quiet) and Precursor signals.
2. **Estimation Stage**: Magnitude-aware classification into Moderate (M4.5-4.9), Medium (M5.0-5.9), and Large (M6.0+) events.

### Results
The model, built on an EfficientNet-B0 backbone, achieved a **binary accuracy of 89.0%** in Phase 2.1. In **Experiment 3**, which focused on a modernized high-flux dataset (**2,265 samples**, 2024-2025 focus), the system demonstrated exceptional sensitivity by achieving a **100.0% Recall** and **100.0% Precision** for the Large Magnitude class. Although the True Negative Rate for solar quiet periods adjusted to **86.0%** under high-flux conditions, the categorical locking for disaster-level events remained superior to existing benchmarks.

### Conclusion
The results demonstrate that hierarchical deep learning, when combined with careful spectral homogenization, can significantly improve the reliability of geomagnetic precursor detection. This model provides a robust foundation for next-generation automated seismic alert systems, particularly for high-magnitude event mitigation.
