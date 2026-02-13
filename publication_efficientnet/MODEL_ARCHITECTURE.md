# Model Architecture: Hierarchical EfficientNet-B0 (Phase 2.1)

The Phase 2.1 Champion model utilizes a **Hierarchical Convolutional Neural Network** design, moving away from flat multi-class approaches to mirror the physical nature of seismic precursory signals.

## 1. Backbone: EfficientNet-B0
- **Why EfficientNet?**: Selected for its optimal balance between parameter efficiency and feature extraction capability via Compound Scaling.
- **Input Dimensions**: 224x224x3 Spectrogram images (derived from Z/H Ratio).
- **Pre-trained Weights**: Initialized with ImageNet weights, then fine-tuned on the seismic domain.

## 2. Hierarchical Head Structure
Unlike standard classifiers, the model splits the decision process into two logical branches:

### Branch A: Precursor Detection (Binary)
- **Objective**: Distinguish between 'Normal' (Geomagnetic quiet) and 'Precursor' (Seismic anomaly).
- **Activation**: Softmax (2 classes).
- **Performance**: 89.0% Binary Accuracy.

### Branch B: Magnitude Categorization (Magnitude-Aware)
- **Objective**: Sub-classify detected Precursors into Moderate, Medium, and Large classes.
- **Relationship**: Conditional on Branch A. The model optimizes these classes primarily through the validation of Large-magnitude signals.
- **Large Class Performance**: 98.6% Recall.

## 3. Training & Optimization
- **Loss Function**: Weighted Cross-Entropy (to handle class imbalance between Large and Moderate events).
- **Optimizer**: Adam with learning rate scheduler ($1e-4$ base).
- **Regularization**: Dropout (0.3) and spectral augmentation during training.
- **Platform**: PyTorch implementation.

## 4. Key Improvements over Phase 1 (ConvNeXt)
- **Efficiency**: 5x faster inference time compared to the large ConvNeXt variant.
- **Homogenization Integration**: Specifically tuned to handle the 2024-2025 solar flux data alongside 2018 historical data.
- **Hierarchical Gating**: Reduced the misclassification of noise as 'Large' events to zero.
