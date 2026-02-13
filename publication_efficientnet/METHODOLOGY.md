# Methodology: Dataset Homogenization and Signal Processing

The success of the Phase 2.1 model is rooted in the rigorous preparation of the geomagnetic dataset, specifically addressing the temporal and environmental biases present in seismic monitoring.

## 1. Data Acquisition
- **Sources**: Remote sensing from a multi-station network (Stations: GTO, MLB, SCN, YOG, etc.).
- **Signal**: 24-hour vertical (Z) and horizontal (H) geomagnetic field intensity.
- **Derived Metric**: Spectral Ratio ($Z/H$), calculated via Short-Time Fourier Transform (STFT).

## 2. Dataset Homogenization (Phase 2.1 Strategy)
One of the core novelties is the **Solar Cycle Flux Homogenization**.
- **The Problem**: Precursors in 2018 (Solar Minimum) look visually different from precursors in 2024 (Solar Maximum) due to higher baseline magnetic activity.
- **The Solution**: 
    - Integration of a balanced 2,340-sample master dataset.
    - Explicit addition of **500 samples of 'Modern Normal' (2024)** data to teach the model to ignore high-flux solar noise.
    - Uniform spectrogram generation parameters across all epochs.

## 3. Signal Processing Pipeline
1. **Binary Extraction**: SSH protocols to fetch raw magnetic records from the observatory servers.
2. **Spectral Generation**: 
    - Frequency Range: ULF (0.01 - 0.1 Hz).
    - Windowing: Hanning window with 50% overlap.
    - Visualization: Jet-colormap Jet Intensity Spectrograms (224x224).
3. **Ground Truth Correlation**: Catalog matching with Indonesian seismic records (M4.5 - M7.0+).

## 4. Evaluation Framework
- **Primary Metric**: Recall for Large Magnitudes (High-risk events).
- **Secondary Metric**: Binary True Negative Rate (System stability).
- **Validation Strategy**: 80/10/10 Split with station-stratified sampling to ensure spatial generalizability.
