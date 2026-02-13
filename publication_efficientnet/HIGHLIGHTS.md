# Highlights: Hierarchical EfficientNet for Earthquake Precursor Detection
**Model Version: Phase 2.1 (Champion State)**
**Publication Target: Geoscience Frontiers / IEEE Transactions on Geoscience and Remote Sensing**

1. **Unprecedented Large Event Recall**: The hierarchical model achieves a **98.65% recall rate** for large-magnitude earthquakes (M6.0+), significantly outperforming standard CNN architectures.
2. **Zero False Positives in High-Magnitude Detection**: Achieved **100.0% precision** for Large events, critical for preventing "cry wolf" scenarios in early warning systems.
3. **Optimized Hierarchical Architecture**: Implemented a two-stage decision process using EfficientNet-B0 backbone, separating binary classification (precursor vs. noise) from magnitude estimation.
4. **Spectral Homogenization for Solar Bias**: Successfully mitigated "Solar Cycle Flux Bias" through a curated 2,340-sample dataset, integrating modern (2024-2025) and historical (2018) Z/H spectral ratios.
5. **Real-time Deployment Readiness**: The model demonstrates inference times under 100ms per station-hour, enabling seamless integration into automated seismic monitoring pipelines.
6. **Robust Negative Validation**: Maintained a **96.9% True Negative Rate (Recall Normal)**, ensuring high reliability during seismically quiet periods.
