# Sistem Deteksi Prekursor Gempa Bumi
## Berbasis Analisis Spektrogram Geomagnetik dengan Deep Learning

**Version**: 3.0 (Experiment 3 - Research State)  
**Status**: ✅ **COMPLETE & VALIDATED**  
**Last Updated**: 2026-02-13

---

## 🎯 Project Overview

Sistem prediksi gempa bumi menggunakan **Hierarchical Deep Learning** untuk mendeteksi anomali geomagnetik sebagai prekursor gempa. Model dilatih pada spektrogram 3-channel (H, D, Z) dari 24 stasiun seismik di Indonesia.

### Key Features
- ✅ **Hierarchical Classification**: Binary Gate → Magnitude Estimator → Azimuth Locator
- ✅ **Homogenized Dataset**: 2,340 samples (Includes 500 new Normal samples from 2023-2025)
- ✅ **Zero Data Leakage**: Event-based stratified splitting
- ✅ **Smart Augmentation**: SMOTE-based balancing (2,426 training samples)
- ✅ **State-of-the-Art Model**: EfficientNet-B0 with multi-task heads

---

## 🚀 Quick Start

### Installation
```bash
# Clone repository
cd d:/multi

# Install dependencies
pip install -r requirements_pytorch.txt

# Verify installation
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
```

### Run Training (Phase 2)
```bash
# Full pipeline (from scratch)
python scripts_v2/1_merge_datasets.py      # Merge data sources
python scripts_v2/2_create_split.py        # Create splits
python scripts_v2/3_apply_balancing.py     # Apply SMOTE
python scripts_v2/4_run_phase2_training.py # Train model

# Or run validation only (if model exists)
python scripts_v2/5_validate_comprehensive.py
```

### Run Dashboard
```bash
streamlit run project_dashboard_v2.py --server.port 8501
```

---

## 📊 Current Status (Research State - Exp 3)

### Dataset Statistics
| Class | Samples | Quality | Source |
|-------|---------|---------|--------|
| **Large** (M6.0+) | 447 | ⭐⭐⭐ High | Historical + SSH 2025 |
| **Medium** (M5.0-5.9) | 341 | ⭐⭐ Hybrid | Legacy + New Scan (mdata2) |
| **Moderate** (M4.5-4.9) | 500 | ⭐⭐⭐ High | Extensive SSH Scan |
| **Normal** (< M4.0) | 1,000 | ⭐⭐⭐ High | **Modern (2024-2025)** |
| **TOTAL** | **2,288** | | |

### Performance Targets & Results
| Metric | Champion Q1 | Phase 2.1 | **Exp 3 (Final)** |
|--------|-------------|-----------|-------------------|
| **Recall Large** | ~70% | 98.65% | **100.0%** ✅ |
| **Precision Large** | ~50% | 100.0% | **100.0%** ✅ |
| **Normal Recall** | ~90% | 96.9% | **86.0%** (Solar Max) |

---

## 📁 Project Structure

```
d:/multi/
├── scripts_v2/                    # Phase 2 Pipeline
│   ├── 1_merge_datasets.py        # Data consolidation
│   ├── 2_create_split.py          # Stratified splitting
│   ├── 3_apply_balancing.py       # SMOTE augmentation
│   ├── 4_run_phase2_training.py   # Training orchestrator
│   ├── 5_validate_comprehensive.py # Validation & comparison
│   └── generate_dataset_*.py      # Data scanners
│
├── dataset_consolidation/         # Merged dataset (1,840)
│   ├── spectrograms/
│   └── metadata/
│       ├── split_train.csv        # 1,234 samples
│       ├── split_val.csv          # 303 samples
│       └── split_test.csv         # 303 samples (Golden)
│
├── dataset_smote_train/           # Balanced training (2,100)
│   ├── spectrograms/              # Original + Synthetic
│   └── augmented_train_metadata.csv
│
├── autoupdate_pipeline/           # Core Engine
│   └── src/
│       └── trainer_v2.py          # Hierarchical trainer
│
├── experiments_v2/                # Training outputs
│   └── hierarchical/
│       └── best_model.pth         # Best model checkpoint
│
└── [Documentation]
    ├── PHASE2_IMPLEMENTATION_SUMMARY.md  # Technical report
    ├── PHASE2_QUICK_REFERENCE.md         # Daily ops guide
    ├── CHECKPOINT_DATASET_V2.md          # Dataset status
    ├── SPR_INVESTIGATION_REPORT.md       # Physics gate research
    └── CHANGELOG_PHASE2.md               # Version history
```

---

## 📚 Documentation

### Essential Reading (Start Here)
1. **[PHASE2_QUICK_REFERENCE.md](PHASE2_QUICK_REFERENCE.md)** - Quick start & troubleshooting
2. **[PHASE2_IMPLEMENTATION_SUMMARY.md](PHASE2_IMPLEMENTATION_SUMMARY.md)** - What we built & why
3. **[CHECKPOINT_PERFORMANCE_EFFICIENTNET.md](CHECKPOINT_PERFORMANCE_EFFICIENTNET.md)** - Performance Metrics & Validation
4. **[CHECKPOINT_DATASET_V2.md](CHECKPOINT_DATASET_V2.md)** - Current dataset status

### Deep Dive
4. **[STRATEGY_HIERARCHY_CHAMPION.md](STRATEGY_HIERARCHY_CHAMPION.md)** - Overall strategy
5. **[SPR_INVESTIGATION_REPORT.md](SPR_INVESTIGATION_REPORT.md)** - Physics gate research
6. **[DATASET_EVALUATION_PLAN.md](DATASET_EVALUATION_PLAN.md)** - Quality assessment
7. **[CHANGELOG_PHASE2.md](CHANGELOG_PHASE2.md)** - Version history

### Legacy (Phase 1)
- [DOKUMENTASI_UTAMA.md](DOKUMENTASI_UTAMA.md) - Phase 1 documentation
- [PANDUAN_DASHBOARD.md](PANDUAN_DASHBOARD.md) - Dashboard guide
- [PANDUAN_PREKURSOR_SCANNER.md](PANDUAN_PREKURSOR_SCANNER.md) - Scanner guide

---

## 🛠️ Common Tasks

### Check Dataset Stats
```bash
python -c "import pandas as pd; df=pd.read_csv('dataset_consolidation/metadata.csv'); print(df['magnitude_class'].value_counts())"
```

### Check Metrics & Graphs
```bash
# View final validation report
type experiments_v2/hierarchical/validation_report_v2.json

# View automated charts
# performance_chart.png, vis_comparison_q1.png, research_training_history.png
```

### Scan New Data
```bash
# Scan Moderate (M4.5-4.9)
python scripts_v2/generate_dataset_moderate.py

# Scan Medium (M5.0-5.9)
python scripts_v2/generate_dataset_medium.py

# Scan Normal (future - for homogeneity)
python scripts_v2/generate_dataset_normal_new.py
```

### Validate Model
```bash
# After training completes
python scripts_v2/5_validate_comprehensive.py
```

---

## 🏆 Model Architecture

### Hierarchical EfficientNet (Phase 2)
```
Input: 224×224 RGB Spectrogram (H, D, Z channels)
    ↓
EfficientNet-B0 Backbone (Pretrained ImageNet)
    ↓
Shared Neck (256-dim embedding)
    ↓
    ├─→ Binary Head (2): Normal vs Precursor
    ├─→ Magnitude Head (4): Normal, Moderate, Medium, Large
    └─→ Azimuth Head (9): Normal, N, NE, E, SE, S, SW, W, NW
```

**Loss Function**:
```
Total Loss = 2.0 × Binary Loss + 1.0 × Magnitude Loss + 0.5 × Azimuth Loss
```

**Class Weights**:
- Binary: Dynamic (based on Normal:Precursor ratio)
- Magnitude: Inverse frequency + **2x boost for Large**
- Azimuth: Uniform

---

## 🔬 Research Contributions

### Phase 2 Innovations
1. **Event-Based Splitting**: Prevents data leakage in earthquake datasets
2. **Selective SMOTE**: Balances minority classes without corrupting majority
3. **Magnitude-Focused Weighting**: 2x boost for critical Large class
4. **SPR Investigation**: Documented why Z/H ratio fails with spektrogram PNG

### Publications (Planned)
- [ ] Phase 2 Technical Report (Q1 2026)
- [ ] Domain Shift in Seismic AI (Q2 2026)
- [ ] Hierarchical Classification for Earthquake Prediction (Q3 2026)

---

## 🐛 Known Issues

### Closed Issues (Recently Solved)
1. **Normal Class Heterogeneity** - Legacy vs New data bias
   - **Resolution**: Rescan 500 Normal samples 2023-2025 completed. Dataset merged & homogenized.
   - **Status**: ✅ FIXED

### Remaining Priorities
2. **Synthetic Ratio Management** - SMOTE Dependency
   - **Status**: Reduced to 35.7% (Improved from 41%)

### Low Priority
3. **CPU Training Speed** - 5-10 min/epoch
   - **Mitigation**: Background execution, early stopping

---

## 🤝 Contributing

### Reporting Issues
1. Check existing documentation first
2. Provide full error traceback
3. Include dataset statistics and environment info

### Development Workflow
```bash
# 1. Create feature branch
git checkout -b feature/your-feature

# 2. Make changes
# ... edit files ...

# 3. Test locally
python scripts_v2/1_merge_datasets.py  # etc.

# 4. Update documentation
# Edit relevant .md files

# 5. Commit with descriptive message
git commit -m "feat: Add XYZ feature"

# 6. Push and create PR
git push origin feature/your-feature
```

---

## 📞 Contact & Support

### Technical Support
- **Dashboard**: http://localhost:8501
- **Documentation**: See `PHASE2_QUICK_REFERENCE.md`

### Team
- **Lead Developer**: Antigravity Team
- **Domain Expert**: Geophysics Consultant
- **Data Source**: BMKG Stasiun Geofisika Mataram

---

## 📜 License

This project is proprietary research software for earthquake prediction.  
Unauthorized distribution is prohibited.

---

## 🙏 Acknowledgments

- **BMKG** - Data access and domain expertise
- **PyTorch Team** - Deep learning framework
- **EfficientNet Authors** - Model architecture
- **Academic Community** - Research on seismo-electromagnetics

---

**README Version**: 2.0  
**Last Updated**: 2026-02-12 15:12  
**Maintained By**: Antigravity Development Team
