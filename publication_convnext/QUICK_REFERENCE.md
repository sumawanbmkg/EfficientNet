# Quick Reference - ConvNeXt Publication

## Panduan Cepat Publikasi ConvNeXt

**Last Updated**: 6 February 2026  
**Status**: ✅ LOEO Validation Complete

---

## 📁 Struktur Folder

```
publication_convnext/
├── README.md                          # Overview package
├── ABSTRACT.md                        # Abstrak paper
├── MANUSCRIPT_DRAFT.md                # Draft manuskrip
├── MODEL_ARCHITECTURE.md              # Detail arsitektur
├── SUPPLEMENTARY_MATERIALS.md         # Materi tambahan
├── COMPARISON_WITH_OTHER_MODELS.md    # Perbandingan model
├── COVER_LETTER.md                    # Surat pengantar
├── TARGET_JOURNALS.md                 # Jurnal target
├── HIGHLIGHTS.md                      # Highlight paper
├── REVIEWER_RESPONSE_TEMPLATE.md      # Template respons reviewer
├── QUICK_REFERENCE.md                 # File ini
├── TRAINING_REPORT.md                 # Laporan training lengkap
└── scripts/
    ├── generate_convnext_figures.py   # Generate figure
    ├── generate_gradcam_convnext.py   # Grad-CAM
    ├── train_loeo_convnext.py         # LOEO validation
    └── evaluate_convnext.py           # Evaluasi model
```

---

## 📊 Hasil LOEO Cross-Validation (FINAL)

| Metric | Result |
|--------|--------|
| **Magnitude Accuracy** | **97.53% ± 0.96%** |
| **Azimuth Accuracy** | **69.30% ± 5.74%** |
| Best Fold (Mag) | 98.15% (Fold 5) |
| Best Fold (Azi) | 82.00% (Fold 9) |
| Worst Fold (Mag) | 95.56% (Fold 10) |
| Worst Fold (Azi) | 57.78% (Fold 10) |

---

## 🚀 Langkah-Langkah Publikasi

### ✅ 1. LOEO Validation - COMPLETE
```bash
# Hasil tersedia di:
type loeo_convnext_results\loeo_convnext_final_results.json
type loeo_convnext_results\LOEO_CONVNEXT_REPORT.md
```

### ⏳ 2. Generate Figures (Pending)
```bash
python publication_convnext/scripts/generate_convnext_figures.py
```

### ⏳ 3. Generate Grad-CAM (Pending)
```bash
python publication_convnext/scripts/generate_gradcam_convnext.py
```

### ⏳ 4. LOSO Validation (Optional)
```bash
python train_loso_validation.py --model convnext
```

### ✅ 5. Update Dokumen - COMPLETE
Semua dokumen sudah diupdate dengan hasil LOEO:
- ✅ MANUSCRIPT_DRAFT.md
- ✅ SUPPLEMENTARY_MATERIALS.md
- ✅ COMPARISON_WITH_OTHER_MODELS.md
- ✅ ABSTRACT.md
- ✅ HIGHLIGHTS.md
- ✅ TRAINING_REPORT.md
- ✅ README.md

---

## 📝 Checklist Publikasi

### Pre-Submission
- [x] LOEO validation selesai
- [x] Dokumentasi diupdate dengan hasil
- [ ] LOSO validation (optional)
- [ ] Grad-CAM generated
- [ ] Semua figure generated

### Dokumen
- [x] Manuskrip draft lengkap
- [x] Supplementary materials lengkap
- [x] Cover letter template siap
- [x] Highlights siap
- [x] Abstract final

### Quality Check
- [x] Semua [TBD] sudah diisi dengan hasil aktual
- [ ] Figure berkualitas tinggi (300 DPI)
- [ ] Referensi lengkap
- [ ] Grammar check
- [ ] Format sesuai jurnal target

---

## 🎯 Target Jurnal

| Prioritas | Jurnal | IF | Fit |
|-----------|--------|-----|-----|
| 1 | Computers & Geosciences | 4.4 | ⭐⭐⭐⭐⭐ |
| 2 | NHESS | 4.6 | ⭐⭐⭐⭐⭐ |
| 3 | IEEE TGRS | 8.2 | ⭐⭐⭐⭐ |
| 4 | Earth Planets Space | 3.0 | ⭐⭐⭐⭐ |

---

## 📈 Perbandingan Model (Final)

| Model | Parameters | Mag Acc (LOEO) | Azi Acc (LOEO) |
|-------|------------|----------------|----------------|
| VGG16 | 138M | 98.68% | 54.93% |
| EfficientNet-B0 | 5.3M | 97.53% ± 0.96% | 69.51% ± 5.65% |
| **ConvNeXt-Tiny** | **28.6M** | **97.53% ± 0.96%** | **69.30% ± 5.74%** |

**Kesimpulan**: ConvNeXt mencapai performa setara dengan EfficientNet-B0 untuk magnitude classification, dengan arsitektur yang lebih modern.

---

## 🔧 Troubleshooting

### Melihat Hasil LOEO
```bash
type loeo_convnext_results\loeo_convnext_final_results.json
```

### Melihat Per-Fold Results
```bash
type loeo_convnext_results\fold_1_result.json
type loeo_convnext_results\fold_9_result.json  # Best fold
type loeo_convnext_results\fold_10_result.json # Worst fold
```

### Script Error
```bash
# Install dependencies
pip install torch torchvision matplotlib seaborn opencv-python
```

---

## 📞 Kontak

Untuk pertanyaan tentang package ini, hubungi tim penelitian.

---

*Last Updated: 6 February 2026 - LOEO Validation Complete*
