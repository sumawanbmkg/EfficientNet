# Investigasi Progress Flowchart Disertasi

**Tanggal Investigasi**: 11 Februari 2026

---

## Flowchart Disertasi (dari Proposal)

```
╔══════════════════════════════════════════════════════════════════════════════╗
║                              TAHUN PERTAMA                                    ║
╠══════════════════════════════════════════════════════════════════════════════╣
║                                                                               ║
║  ┌─────────┐                           ┌─────────┐                           ║
║  │  Start  │                           │    a    │                           ║
║  └────┬────┘                           └────┬────┘                           ║
║       │                                     │                                ║
║       ▼                                     ▼                                ║
║  ┌─────────────────────┐              ┌─────────────────────┐               ║
║  │ 1. Pengumpulan data │              │ 5. Pembuatan        │               ║
║  │    geomagnetik      │              │    Generator Data   │               ║
║  └─────────┬───────────┘              │    Sintetis         │               ║
║            │                          └─────────┬───────────┘               ║
║            ▼                                    │                            ║
║  ┌─────────────────────┐                        ▼                            ║
║  │ 2. Pre-processing   │              ┌─────────────────────┐               ║
║  │    data geomagnetik │              │ 6. Pengembangan     │               ║
║  └─────────┬───────────┘              │    model deteksi    │               ║
║            │                          │    CNN dengan fitur │               ║
║            ▼                          │    self-updating    │               ║
║  ┌─────────────────────┐              └─────────┬───────────┘               ║
║  │ 3. Ekstraksi fitur  │                        │                            ║
║  └─────────┬───────────┘                        ▼                            ║
║            │                          ┌─────────────────────┐               ║
║            ▼                          │ Akurasi > 95%?      │               ║
║  ┌─────────────────────┐              └─────────┬───────────┘               ║
║  │ 4. Pembuatan model  │                        │                            ║
║  │    deteksi CNN      │                   Ya   │                            ║
║  └─────────┬───────────┘                        ▼                            ║
║            │                                ┌───────┐                        ║
║            ▼                                │   b   │ → Tahun 2              ║
║  ┌─────────────────────┐                    └───────┘                        ║
║  │ Akurasi > 80%?      │                                                     ║
║  └─────────┬───────────┘                                                     ║
║            │                                                                 ║
║       Ya   │                                                                 ║
║            ▼                                                                 ║
║        ┌───────┐                                                             ║
║        │   a   │                                                             ║
║        └───────┘                                                             ║
║                                                                               ║
╚══════════════════════════════════════════════════════════════════════════════╝

╔══════════════════════════════════════════════════════════════════════════════╗
║                              TAHUN KEDUA                                      ║
╠══════════════════════════════════════════════════════════════════════════════╣
║                                                                               ║
║  ┌─────────┐                           ┌─────────┐                           ║
║  │    b    │                           │    c    │                           ║
║  └────┬────┘                           └────┬────┘                           ║
║       │                                     │                                ║
║       ▼                                     ▼                                ║
║  ┌─────────────────────┐              ┌─────────────────────┐               ║
║  │ 7. Pengumpulan data │              │ 11. Pembuatan       │               ║
║  │    pendukung        │              │     Generator Data  │               ║
║  └─────────┬───────────┘              │     Sintetis        │               ║
║            │                          └─────────┬───────────┘               ║
║            ▼                                    │                            ║
║  ┌─────────────────────┐                        ▼                            ║
║  │ 8. Pre-processing   │              ┌─────────────────────┐               ║
║  │    data pendukung   │              │ 12. Pengembangan    │               ║
║  └─────────┬───────────┘              │     model prediksi  │               ║
║            │                          │     dengan fitur    │               ║
║            ▼                          │     online learning │               ║
║  ┌─────────────────────┐              └─────────┬───────────┘               ║
║  │ 9. Integrasi data   │                        │                            ║
║  │    multi-parameter  │                        ▼                            ║
║  └─────────┬───────────┘              ┌─────────────────────┐               ║
║            │                          │ Akurasi > 95%?      │               ║
║            ▼                          └─────────┬───────────┘               ║
║  ┌─────────────────────┐                        │                            ║
║  │ 10. Pengembangan    │                   Ya   │                            ║
║  │     model prediksi  │                        ▼                            ║
║  │     parameter gempa │                  ┌──────────┐                       ║
║  └─────────┬───────────┘                  │  Finish  │                       ║
║            │                              └──────────┘                       ║
║            ▼                                                                 ║
║  ┌─────────────────────┐                                                     ║
║  │ Akurasi > 85%?      │                                                     ║
║  └─────────┬───────────┘                                                     ║
║            │                                                                 ║
║       Ya   │                                                                 ║
║            ▼                                                                 ║
║        ┌───────┐                                                             ║
║        │   c   │                                                             ║
║        └───────┘                                                             ║
║                                                                               ║
╚══════════════════════════════════════════════════════════════════════════════╝
```

---

## INVESTIGASI DETAIL PER POIN

### ══════════════════════════════════════════════════════════════
### TAHUN PERTAMA
### ══════════════════════════════════════════════════════════════

---

### 1. Pengumpulan Data Geomagnetik

| Aspek | Status | Detail |
|-------|--------|--------|
| **Status** | ✅ **SELESAI** | |
| **Target Proposal** | Data geomagnetik (H, D, Z) minimal 5 tahun dari BMKG | |
| **Realisasi** | Data 2018-2025 (7 tahun) dari 25 stasiun | |

**Bukti Implementasi:**
- `earthquake_catalog_2018_2025_merged.csv` - Katalog gempa 2018-2025
- `scan_earthquake_precursors.py` - Script untuk fetch data via SSH
- `geomagnetic_dataset_generator_ssh.py` - Generator dataset dari server BMKG
- `mdata2/` dan `missing/` - Folder data lokal

**Statistik:**
- 25 stasiun magnetometer aktif
- 105+ gempa M≥6.0 dalam katalog
- 2000+ spektrogram dihasilkan
- Periode: 2018-2025 (melebihi target 5 tahun)

**Progress**: 100%

---

### 2. Pre-processing Data Geomagnetik

| Aspek | Status | Detail |
|-------|--------|--------|
| **Status** | ✅ **SELESAI** | |
| **Target Proposal** | Pembersihan, normalisasi, pelabelan | |
| **Realisasi** | Filtering PC3, Z/H ratio, quality control | |

**Bukti Implementasi:**
- `geomagnetic_dataset_generator_ssh_v2.py` - Preprocessing pipeline
- Filtering bandpass PC3 (10-45 mHz)
- Perhitungan Z/H ratio per jam
- Quality control dan outlier removal
- Normalization min-max

**Teknik yang Digunakan:**
```python
# Bandpass filter PC3
lowcut = 0.01  # 10 mHz
highcut = 0.045  # 45 mHz
# Z/H ratio calculation per hour
zh_ratio = np.sqrt(psd_z / psd_h)
```

**Progress**: 100%

---

### 3. Ekstraksi Fitur

| Aspek | Status | Detail |
|-------|--------|--------|
| **Status** | ✅ **SELESAI** | |
| **Target Proposal** | Wavelet Scattering Transform (WST) dan analisis rasio Z/H | |
| **Realisasi** | STFT Spectrogram + Z/H ratio analysis | |

**Bukti Implementasi:**
- `geomagnetic_dataset_generator_ssh_v2.py` - STFT spectrogram generation
- `dataset_unified/` - Dataset spektrogram 224×224
- `visualize_augmentation.py` - Visualisasi spektrogram

**Catatan:**
- Proposal menyebut WST, implementasi menggunakan STFT
- Keduanya valid untuk ekstraksi fitur time-frequency
- STFT lebih umum digunakan dan sudah terbukti efektif (97.47% accuracy)

**Output:**
- Spektrogram 224×224×3 (RGB)
- Format PNG
- 2000+ samples

**Progress**: 100%

---

### 4. Pembuatan Model Deteksi CNN

| Aspek | Status | Detail |
|-------|--------|--------|
| **Status** | ✅ **SELESAI** | |
| **Target Proposal** | Model CNN dengan akurasi > 80% | |
| **Realisasi** | EfficientNet-B0 dengan akurasi 97.47% | |

**Bukti Implementasi:**
- `train_with_fixed_split.py` - Training script
- `train_fixed_split_pytorch.py` - PyTorch training
- `experiments_fixed/*/best_model.pth` - Model tersimpan
- `evaluate_fixed_model.py` - Evaluasi model

**Model yang Diuji:**
| Model | Accuracy | Status |
|-------|----------|--------|
| VGG16 | 92.3% | Legacy |
| EfficientNet-B0 | **97.47%** | **Production** |
| ConvNeXt-Tiny | 95.2% | Backup |
| Xception | 91.8% | Tested |

**Validasi:**
- LOEO (Leave-One-Event-Out) validation
- Mean accuracy: 93.2%
- Grad-CAM untuk interpretabilitas

**Progress**: 100% ✅ **MELEBIHI TARGET (97.47% > 80%)**

---

### 5. Pembuatan Generator Data Sintetis

| Aspek | Status | Detail |
|-------|--------|--------|
| **Status** | 🟡 **SEBAGIAN SELESAI** | |
| **Target Proposal** | Generator data sintetis untuk augmentasi | |
| **Realisasi** | SMOTE implemented, perlu enhancement | |

**Bukti Implementasi:**
- `generate_smote_dataset.py` - SMOTE implementation
- `train_with_smote.py` - Training dengan SMOTE
- `generate_augmented_dataset.py` - Data augmentation
- `dataset_augmented/` - Dataset augmented

**Teknik yang Diimplementasikan:**
1. ✅ SMOTE (Synthetic Minority Over-sampling Technique)
2. ✅ Image augmentation (rotation, flip, brightness)
3. ✅ Focal Loss untuk class imbalance

**Masalah yang Masih Ada:**
- Class imbalance: Large class hanya 1.4%
- Perlu lebih banyak data untuk kelas minoritas
- SSH scan sedang berjalan untuk menambah data

**Progress**: 70%

---

### 6. Pengembangan Model CNN dengan Fitur Self-Updating

| Aspek | Status | Detail |
|-------|--------|--------|
| **Status** | ✅ **SELESAI** | |
| **Target Proposal** | Model CNN dengan self-updating, akurasi > 95% | |
| **Realisasi** | Auto-update pipeline dengan champion-challenger | |

**Bukti Implementasi:**
- `autoupdate_pipeline/` - Complete pipeline folder
  - `src/data_ingestion.py` - Data ingestion
  - `src/trainer.py` - Model trainer
  - `src/evaluator.py` - Model evaluator
  - `src/model_comparator.py` - Champion-challenger comparison
  - `src/deployer.py` - Model deployment
- `autoupdate_pipeline/scripts/run_pipeline.py` - Pipeline runner
- `autoupdate_pipeline/scripts/add_new_event.py` - Add new events

**Fitur Self-Updating:**
```
┌─────────────────────────────────────────────────────┐
│                 Auto-Update Pipeline                 │
├─────────────────────────────────────────────────────┤
│  Data Ingestion → Trainer → Evaluator               │
│                              ↓                      │
│  Deployer ← Comparator ← Challenger Model           │
│      ↓                                              │
│  Production Model                                   │
└─────────────────────────────────────────────────────┘
```

**Akurasi Tercapai**: 97.47% ✅ **MELEBIHI TARGET (> 95%)**

**Progress**: 100%

---

### ══════════════════════════════════════════════════════════════
### TAHUN KEDUA (Belum Dimulai)
### ══════════════════════════════════════════════════════════════

---

### 7. Pengumpulan Data Pendukung

| Aspek | Status | Detail |
|-------|--------|--------|
| **Status** | ⏳ **BELUM DIMULAI** | |
| **Target Proposal** | Data seismik, ionosferik, geoatmosferik | |
| **Realisasi** | - | |

**Rencana:**
- Data ionosfer (TEC) dari LAPAN
- Data GPS/GNSS dari BIG
- Data seismik dari BMKG

**Progress**: 0%

---

### 8. Pre-processing Data Pendukung

| Aspek | Status | Detail |
|-------|--------|--------|
| **Status** | ⏳ **BELUM DIMULAI** | |

**Progress**: 0%

---

### 9. Integrasi Data Multi-Parameter

| Aspek | Status | Detail |
|-------|--------|--------|
| **Status** | ⏳ **BELUM DIMULAI** | |
| **Target Proposal** | Fusi data level fitur dengan PCA | |

**Progress**: 0%

---

### 10. Pengembangan Model Prediksi Parameter Gempa

| Aspek | Status | Detail |
|-------|--------|--------|
| **Status** | 🟡 **SEBAGIAN** | |
| **Target Proposal** | Estimasi magnitudo, lokasi, waktu dengan akurasi > 85% | |
| **Realisasi** | Sudah ada prediksi azimuth dan magnitude | |

**Yang Sudah Ada:**
- ✅ Prediksi Azimuth (8 kelas): 96.8% accuracy
- ✅ Prediksi Magnitude (5 kelas): 94.4% accuracy
- ⏳ Prediksi Lokasi (jarak): Belum
- ⏳ Prediksi Time Window: Belum

**Progress**: 40% (sebagian sudah di Tahun 1)

---

### 11. Pembuatan Generator Data Sintetis (untuk data pendukung)

| Aspek | Status | Detail |
|-------|--------|--------|
| **Status** | ⏳ **BELUM DIMULAI** | |

**Progress**: 0%

---

### 12. Pengembangan Model Prediksi dengan Fitur Online Learning

| Aspek | Status | Detail |
|-------|--------|--------|
| **Status** | ⏳ **BELUM DIMULAI** | |
| **Target Proposal** | Online learning untuk adaptasi real-time | |

**Catatan:**
- Self-updating pipeline sudah ada (batch update)
- Online learning (incremental) belum diimplementasi
- Perlu EWC (Elastic Weight Consolidation) untuk mencegah catastrophic forgetting

**Progress**: 0%

---

## RINGKASAN PROGRESS

### Tahun Pertama

| No | Poin | Status | Progress | Catatan |
|----|------|--------|----------|---------|
| 1 | Pengumpulan data geomagnetik | ✅ Selesai | 100% | 7 tahun data, 25 stasiun |
| 2 | Pre-processing data | ✅ Selesai | 100% | PC3 filtering, Z/H ratio |
| 3 | Ekstraksi fitur | ✅ Selesai | 100% | STFT spectrogram |
| 4 | Model deteksi CNN | ✅ Selesai | 100% | 97.47% > 80% target |
| 5 | Generator data sintetis | 🟡 Sebagian | 70% | SMOTE done, perlu enhancement |
| 6 | Model self-updating | ✅ Selesai | 100% | 97.47% > 95% target |

**Total Progress Tahun 1: ~95%**

### Tahun Kedua

| No | Poin | Status | Progress | Catatan |
|----|------|--------|----------|---------|
| 7 | Pengumpulan data pendukung | ⏳ Belum | 0% | Perlu koordinasi LAPAN, BIG |
| 8 | Pre-processing data pendukung | ⏳ Belum | 0% | - |
| 9 | Integrasi multi-parameter | ⏳ Belum | 0% | - |
| 10 | Model prediksi parameter | 🟡 Sebagian | 40% | Azimuth & Mag sudah ada |
| 11 | Generator data sintetis (pendukung) | ⏳ Belum | 0% | - |
| 12 | Model online learning | ⏳ Belum | 0% | - |

**Total Progress Tahun 2: ~7%**

---

## VISUALISASI PROGRESS

```
TAHUN PERTAMA                                    TAHUN KEDUA
═══════════════════════════════════════════════════════════════════════════

[1] Pengumpulan Data     ████████████████████ 100% ✅
[2] Pre-processing       ████████████████████ 100% ✅
[3] Ekstraksi Fitur      ████████████████████ 100% ✅
[4] Model CNN            ████████████████████ 100% ✅ (97.47%)
[5] Generator Sintetis   ██████████████░░░░░░  70% 🟡
[6] Self-Updating        ████████████████████ 100% ✅ (97.47%)

[7] Data Pendukung       ░░░░░░░░░░░░░░░░░░░░   0% ⏳
[8] Pre-proc Pendukung   ░░░░░░░░░░░░░░░░░░░░   0% ⏳
[9] Integrasi Multi-Param░░░░░░░░░░░░░░░░░░░░   0% ⏳
[10] Model Prediksi      ████████░░░░░░░░░░░░  40% 🟡
[11] Generator (Pendukung)░░░░░░░░░░░░░░░░░░░   0% ⏳
[12] Online Learning     ░░░░░░░░░░░░░░░░░░░░   0% ⏳

═══════════════════════════════════════════════════════════════════════════
OVERALL PROGRESS: ████████████░░░░░░░░ ~51% (Tahun 1: 95%, Tahun 2: 7%)
═══════════════════════════════════════════════════════════════════════════
```

---

## PRODUK YANG SUDAH DIHASILKAN

### Software & Scripts
1. ✅ `prekursor_scanner_production.py` - Scanner produksi
2. ✅ `project_dashboard_v2.py` - Dashboard Streamlit
3. ✅ `autoupdate_pipeline/` - Auto-update pipeline lengkap
4. ✅ `scan_earthquake_precursors.py` - SSH scanner
5. ✅ `generate_dataset_from_scan.py` - Dataset generator

### Model
1. ✅ EfficientNet-B0 (97.47%) - Production
2. ✅ ConvNeXt-Tiny (95.2%) - Backup

### Dataset
1. ✅ `dataset_unified/` - 2000+ spektrogram
2. ✅ `earthquake_catalog_2018_2025_merged.csv` - Katalog gempa

### Publikasi
1. 🟡 IEEE TGRS manuscript - Draft ready

---

## REKOMENDASI LANGKAH SELANJUTNYA

### Prioritas Tinggi (Menyelesaikan Tahun 1)
1. **Selesaikan SSH scan** - Menambah data untuk kelas minoritas
2. **Apply SMOTE** - Balancing dataset
3. **Retrain model** - Dengan data yang lebih lengkap
4. **Submit paper** - IEEE TGRS

### Prioritas Menengah (Persiapan Tahun 2)
1. **Koordinasi LAPAN** - Data ionosfer TEC
2. **Koordinasi BIG** - Data GPS/GNSS
3. **Desain arsitektur** - Hybrid CNN-RNN

### Prioritas Rendah
1. **Implementasi WST** - Sesuai proposal (opsional, STFT sudah efektif)
2. **Online learning** - EWC implementation
