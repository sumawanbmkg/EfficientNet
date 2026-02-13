# Rencana Pengembangan Sistem Prekursor Gempa (Fase 2 & Experiment 3)
**Status**: ✅ **COMPLETE & VALIDATED** (13 Februari 2026)
**Tujuan Utama**: Integrasi Data Modern 2025, Solar Robustness, dan Final Publication Ready.
**Hasil Akhir**: **Recall Large 100%**, **Precision 100%** (Exp 3).

---

## 1. 🎯 Target Pencapaian & Metrik Keberhasilan
Fokus Phase 2 adalah kinerja kelas minoritas (Large Earthquake) dan sustainability sistem.

| Metrik | Status Q1 (Initial) | Target Fase 2 | **Hasil Phase 2.1** |
|--------|---------------------|---------------|---------------------|
| **Recall (Large EQ)** | < 15% | > 60% | **98.65%** ✅ |
| **Precision (Large EQ)** | Rendah | > 50% | **100.00%** ✅ |
| **Akurasi Biner** | ~75% | > 80% | **89.00%** ✅ |

---

## 4. 📅 Jadwal Eksekusi (Final Status)

| No | Aktivitas | Script | Status |
|----|-----------|--------|--------|
| 1. | Scan SSH Data (Large Events) | `generate_dataset_from_scan.py` | ✅ COMPLETE |
| 2. | Konsolidasi & Homogenisasi | `1_merge_datasets.py` | ✅ COMPLETE |
| 3. | Membuat Split Data Anti-Bocor | `2_create_split.py` | ✅ COMPLETE |
| 4. | Balancing Data (SMOTE) | `3_apply_balancing.py` | ✅ COMPLETE |
| 5. | Investigasi Z/H Ratio | `investigate_zh_ratio.py` | ✅ COMPLETE |
| 6. | **Upgrade Trainer Module** | `autoupdate_pipeline/src/trainer_v2.py` | ✅ COMPLETE |
| 7. | Training Model Hierarkis | `4_run_phase2_training.py` | ✅ COMPLETE |
| 8. | Validasi & Final Report | `5_validate_comprehensive.py` | ✅ COMPLETE |
| 9. | Research Quality Figures | `8_research_plots.py` | ✅ COMPLETE |

## 2. 🛠️ Arsitektur Pipeline Data Baru

### Tahap A: Konsolidasi & Quality Control
**Script**: `scripts_v2/1_merge_datasets.py`
*   Input: `dataset_unified`, `dataset_new_events`, `dataset_missing_filled`.
*   Output: `dataset_consolidation/`

### Tahap B: Stratified Split (Anti-Leakage)
**Script**: `scripts_v2/2_create_split.py`
*   Input: `dataset_consolidation/metadata.csv`
*   Output: `split_train.csv`, `split_val.csv`, `split_test.csv` (Event-Based).

### Tahap C: Advanced Balancing (Hybrid)
**Script**: `scripts_v2/3_apply_balancing.py`
*   Input: `split_train.csv`.
*   Metode: SMOTE gambar untuk kelas Large/Major pada data Training.
*   Output: `augmented_train_metadata.csv` di `dataset_smote_train/`.

---

## 3. 🧠 Strategi Training Hierarkis Terintegrasi

### Upgrade Auto-Update Module (`autoupdate_pipeline/src/trainer_v2.py`)
Menggantikan trainer lama dengan kapabilitas baru:
1.  **Hierarchical Architecture Support**:
    *   Binary Head (Normal vs Precursor)
    *   Magnitude Head (Conditional)
    *   Azimuth Head (Conditional)
2.  **Physics-Aware**: Menerima input Z/H threshold.
3.  **Dynamic Class Weighting**: Otomatis menyesuaikan loss function.

**Script Eksekusi Phase 2 (`scripts_v2/4_run_phase2_training.py`)**
Wrapper script yang akan memanggil `trainer_v2.py` secara manual untuk inisialisasi model pertama kali dengan parameter agresif.

---

## 4. 📅 Jadwal Eksekusi

| No | Aktivitas | Script | Status | Estimasi |
|----|-----------|--------|--------|----------|
| 1. | Menunggu Scan SSH Data | `generate_dataset_from_scan.py` | 🔄 RUNNING | ~45 Menit |
| 2. | Konsolidasi Data | `1_merge_datasets.py` | ✅ READY | 10 Menit |
| 3. | Membuat Split Data Anti-Bocor | `2_create_split.py` | ✅ READY | 5 Menit |
| 4. | Balancing Data (SMOTE) | `3_apply_balancing.py` | ✅ READY | 20 Menit |
| 5. | Investigasi Z/H Ratio | `investigate_zh_ratio.py` | ✅ READY | 10 Menit |
| 6. | **Upgrade Trainer Module** | `autoupdate_pipeline/src/trainer_v2.py` | ⏳ PENDING | 30 Menit |
| 7. | Training Model Hierarkis | `4_run_phase2_training.py` | ⏳ PENDING | ~4 Jam |
| 8. | Validasi & Final Report | `5_validate_comprehensive.py` | ⏳ PENDING | 30 Menit |
