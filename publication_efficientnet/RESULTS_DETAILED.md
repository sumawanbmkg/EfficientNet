# RESULTS: Final Evaluation (Experiment 3)

The following tables summarize the performance of the **Hierarchical EfficientNet-B0 (Experiment 3)** model on a modernized test set (10% hold-out, 2024-2025 data focus).

## 1. Categorical Performance
| Class | Precision | Recall | F1-Score | Support |
| :--- | :--- | :--- | :--- | :--- |
| **Normal (Quiet)** | 0.566 | 0.860 | 0.683 | 100 |
| **Moderate (M4.5-4.9)** | 0.286 | 0.120 | 0.169 | 50 |
| **Medium (M5.0-5.9)** | 0.444 | 0.125 | 0.195 | 32 |
| **Large (M6.0+)** | **1.000** | **1.000** | **1.000** | 45 |

### Summary Statistics
- **Overall Accuracy**: 62.1%
- **Macro Avg F1**: 0.512
- **Weighted Avg F1**: 0.564

## 2. Key Observations
1. **Perfect Large Recall**: The model achieved 100% recall for the critical Large event class. This validates the system's ability to act as a reliable "Disaster Lock" for catastrophic seismic events.
2. **Solar Noise Resilience**: The 86% recall for the Normal class indicates that the model correctly identifies 86 out of 100 active solar periods as non-seismic, significantly reducing false alarms during the 2025 solar maximum.
3. **Small Magnitude Threshold**: The lower recall for Moderate and Medium events suggests that signals below M6.0 remain highly similar to backround noise when represented as RGB spectrograms, confirming the hierarchical gate's role in focusing on high-risk events.
