# Figures List

## List of Figures for ConvNeXt Publication

**Last Updated**: 6 February 2026  
**Status**: ✅ All Figures Generated

---

## Generated Figures (publication_convnext/figures/)

### Figure 1: LOEO Per-Fold Accuracy
- **File**: `fig1_loeo_per_fold_accuracy.png`
- **Description**: Bar chart showing magnitude, azimuth, and combined accuracy for each of 10 LOEO folds
- **Data**: Magnitude 97.53% ± 0.96%, Azimuth 69.30% ± 5.74%
- **Status**: ✅ Generated

### Figure 2: Model Comparison
- **File**: `fig2_model_comparison.png`
- **Description**: Comparison of VGG16, EfficientNet-B0, and ConvNeXt-Tiny
- **Panels**: (a) Magnitude Acc, (b) Azimuth Acc, (c) Parameters, (d) Model Size
- **Status**: ✅ Generated

### Figure 3: Architecture Diagram
- **File**: `fig3_architecture_diagram.png`
- **Description**: ConvNeXt-Tiny multi-task architecture diagram
- **Components**: Stem, 4 Stages, GAP, Magnitude Head, Azimuth Head
- **Status**: ✅ Generated

### Figure 4: LOEO Box Plot
- **File**: `fig4_loeo_boxplot.png`
- **Description**: Box plot showing distribution of accuracy across 10 folds
- **Metrics**: Magnitude, Azimuth, Combined accuracy distributions
- **Status**: ✅ Generated

### Figure 5: ConvNeXt vs EfficientNet
- **File**: `fig5_convnext_vs_efficientnet.png`
- **Description**: Direct comparison with error bars
- **Data**: Both models achieve 97.53% magnitude accuracy
- **Status**: ✅ Generated

### Figure 6: Per-Fold Heatmap
- **File**: `fig6_fold_heatmap.png`
- **Description**: Heatmap visualization of per-fold performance
- **Metrics**: Magnitude, Azimuth, Combined for all 10 folds
- **Status**: ✅ Generated

### Figure 7: Sample Distribution
- **File**: `fig7_sample_distribution.png`
- **Description**: Training and test sample counts per fold
- **Purpose**: Shows LOEO data split balance
- **Status**: ✅ Generated

### Figure 8: Summary Table
- **File**: `fig8_summary_table.png`
- **Description**: Summary statistics table as image
- **Content**: Mean, Std, Min, Max, Best Fold for each metric
- **Status**: ✅ Generated

---

## Word Documents Generated

| Document | File | Status |
|----------|------|--------|
| Manuscript Draft | `ConvNeXt_Manuscript_Draft.docx` | ✅ Generated |
| Supplementary Materials | `ConvNeXt_Supplementary_Materials.docx` | ✅ Generated |

---

## Figure Specifications

### Resolution
- **All figures**: 300 DPI (publication quality)
- **Format**: PNG

### Color Scheme
- **Magnitude**: Green (#2ecc71)
- **Azimuth**: Blue (#3498db)
- **Combined**: Purple (#9b59b6)
- **VGG16**: Blue (#3498db)
- **EfficientNet**: Green (#2ecc71)
- **ConvNeXt**: Red (#e74c3c)

---

## Quick Commands

```bash
# Regenerate all figures
python generate_convnext_publication_figures.py

# Generate Word manuscript
python generate_convnext_manuscript_word.py

# Generate supplementary materials
python generate_convnext_supplementary_word.py
```

---

## Checklist

### Main Figures
- [x] Figure 1: LOEO per-fold accuracy
- [x] Figure 2: Model comparison
- [x] Figure 3: Architecture diagram
- [x] Figure 4: LOEO box plot
- [x] Figure 5: ConvNeXt vs EfficientNet
- [x] Figure 6: Per-fold heatmap
- [x] Figure 7: Sample distribution
- [x] Figure 8: Summary table

### Documents
- [x] Manuscript Word document
- [x] Supplementary Materials Word document

### Pending (Optional)
- [ ] Grad-CAM visualizations (requires trained model weights)
- [ ] t-SNE feature visualization
- [ ] Confusion matrices (requires per-class predictions)

---

*Figure list updated - 6 February 2026*
