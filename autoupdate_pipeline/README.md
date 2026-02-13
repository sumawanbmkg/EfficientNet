# 🔄 Earthquake Model Auto-Update Pipeline

## Overview

Pipeline otomatis untuk memperbarui model prediksi gempa bumi dengan data baru yang tervalidasi. Sistem ini mengimplementasikan **Champion-Challenger Pattern** dimana model baru harus "mengalahkan" model lama sebelum di-deploy ke production.

## 🖥️ Dashboard Web (Recommended!)

Pipeline ini terintegrasi dengan dashboard web untuk kemudahan operasi:

```bash
# Jalankan dashboard
streamlit run project_dashboard_v2.py

# Buka di browser: http://localhost:8501
# Pilih menu: 🔄 Auto-Update Pipeline
```

**Fitur Dashboard:**
- 📊 **Status Overview** - Lihat status pipeline secara visual
- 📈 **Progress Bar** - Lihat progress threshold dengan visual
- 📋 **Event Management** - Tambah/validasi events via form interaktif
- 🚀 **Run Pipeline** - Generate command untuk menjalankan pipeline
- 📜 **History** - Lihat log dan rollback model

## Fitur Utama

- ✅ **Dashboard Web**: Interface visual untuk operasi pipeline
- ✅ **Data Ingestion**: Validasi dan integrasi data gempa baru
- ✅ **Trigger System**: Otomatis atau manual trigger untuk retraining
- ✅ **Champion-Challenger**: Perbandingan model baru vs model lama
- ✅ **Benchmark Testing**: Fixed test set untuk evaluasi fair
- ✅ **Rollback Mechanism**: Kemampuan kembali ke model sebelumnya
- ✅ **Audit Trail**: Log lengkap semua keputusan dan metrik

## Struktur Folder

```
autoupdate_pipeline/
├── README.md                    # Dokumentasi utama
├── config/
│   ├── pipeline_config.yaml     # Konfigurasi pipeline
│   └── model_registry.json      # Registry model (champion/challenger)
├── src/
│   ├── __init__.py
│   ├── data_ingestion.py        # Modul ingestion data baru
│   ├── data_validator.py        # Validasi data gempa
│   ├── trainer.py               # Training model baru
│   ├── evaluator.py             # Evaluasi dan benchmark
│   ├── model_comparator.py      # Perbandingan champion vs challenger
│   ├── deployer.py              # Deployment model baru
│   └── utils.py                 # Utility functions
├── scripts/
│   ├── run_pipeline.py          # Main pipeline runner
│   ├── add_new_event.py         # Tambah event gempa baru
│   ├── check_status.py          # Cek status pipeline
│   ├── rollback_model.py        # Rollback ke model sebelumnya
│   └── setup_champion.py        # Setup champion model awal
├── data/
│   ├── pending/                 # Data baru menunggu validasi
│   ├── validated/               # Data tervalidasi
│   └── benchmark/               # Fixed benchmark test set
├── models/
│   ├── champion/                # Model production saat ini
│   ├── challenger/              # Model kandidat
│   └── archive/                 # Model-model lama
├── logs/
│   └── pipeline_history.json    # History semua pipeline runs
├── tests/
│   └── test_pipeline.py         # Unit tests
└── docs/
    ├── ARCHITECTURE.md          # Arsitektur sistem
    ├── USER_GUIDE.md            # Panduan pengguna
    └── API_REFERENCE.md         # Referensi API
```

## Quick Start

### Via Dashboard (Recommended)
```bash
# Jalankan dashboard
streamlit run project_dashboard_v2.py

# Buka http://localhost:8501
# Pilih menu "🔄 Auto-Update Pipeline"
# Lihat progress bar untuk status threshold
# Tambah event via form interaktif
```

### Via Command Line
```bash
# 0. Setup champion model (pertama kali)
python scripts/setup_champion.py

# 1. Cek status pipeline
python scripts/check_status.py

# 2. Tambah event gempa baru (dengan nilai numerik)
python scripts/add_new_event.py add -d 2026-02-10 -s GTO -m 6.2 -a 45

# 3. Lihat panduan klasifikasi
python scripts/add_new_event.py guide

# 4. Lihat semua events
python scripts/add_new_event.py list

# 5. Validasi pending events
python scripts/add_new_event.py validate

# 6. Jalankan pipeline (jika threshold terpenuhi)
python scripts/run_pipeline.py

# 7. Rollback jika diperlukan
python scripts/rollback_model.py --list
```

## Installation

```bash
pip install torch torchvision pandas numpy scikit-learn pyyaml pillow
```

## Perintah Lengkap

### Menambah Event (Input Numerik)
```bash
python scripts/add_new_event.py add -d YYYY-MM-DD -s STASIUN -m MAGNITUDE -a AZIMUTH

# Parameter:
#   -d, --date      : Tanggal event (YYYY-MM-DD)
#   -s, --station   : Kode stasiun (GTO, SCN, MLB, dll)
#   -m, --magnitude : Nilai magnitudo (0-10), contoh: 5.7, 6.2
#   -a, --azimuth   : Azimuth dalam derajat (0-360), contoh: 45, 180, 270

# Contoh:
python scripts/add_new_event.py add -d 2026-02-15 -s GTO -m 6.2 -a 45    # Large, NE
python scripts/add_new_event.py add -d 2026-02-16 -s TRT -m 5.3 -a 225   # Medium, SW
python scripts/add_new_event.py add -d 2026-02-17 -s SCN -m 4.5 -a 90    # Moderate, E
python scripts/add_new_event.py add -d 2026-02-18 -s MLB -m 0 -a 0       # Normal (no earthquake)
```

### Klasifikasi Otomatis

Sistem akan otomatis mengkonversi nilai numerik ke kelas:

**Magnitude:**
| Nilai | Kelas |
|-------|-------|
| M ≥ 6.0 | Large |
| 5.0 ≤ M < 6.0 | Medium |
| 4.0 ≤ M < 5.0 | Moderate |
| M < 4.0 | Normal |

**Azimuth:**
| Derajat | Arah |
|---------|------|
| 337.5° - 22.5° | N (North) |
| 22.5° - 67.5° | NE (Northeast) |
| 67.5° - 112.5° | E (East) |
| 112.5° - 157.5° | SE (Southeast) |
| 157.5° - 202.5° | S (South) |
| 202.5° - 247.5° | SW (Southwest) |
| 247.5° - 292.5° | W (West) |
| 292.5° - 337.5° | NW (Northwest) |

### Mengelola Events
```bash
# Lihat semua events
python scripts/add_new_event.py list

# Lihat pending saja
python scripts/add_new_event.py list --type pending

# Validasi pending → validated
python scripts/add_new_event.py validate

# Hapus event tertentu
python scripts/add_new_event.py delete --id GTO_20260215

# Hapus semua pending
python scripts/add_new_event.py clear --confirm
```

### Parameter Valid

| Parameter | Nilai |
|-----------|-------|
| Station | SBG, SCN, KPY, LWA, LPS, SRG, SKB, CLP, YOG, TRT, LUT, ALR, SMI, SRO, TNT, TND, GTO, LWK, PLU, TRD, JYP, AMB, GSI, MLB |
| Magnitude | Nilai numerik 0-10 (otomatis dikonversi ke Large/Medium/Moderate/Normal) |
| Azimuth | Nilai derajat 0-360 (otomatis dikonversi ke N/NE/E/SE/S/SW/W/NW) |

## Workflow

```
┌─────────────────────────────────────────────────────────────┐
│                    AUTO-UPDATE WORKFLOW                      │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  [New Event] → [Validate] → [Add to Dataset]                │
│                     │                                        │
│                     ▼                                        │
│            [Check Trigger]                                   │
│                     │                                        │
│         ┌──────────┴──────────┐                             │
│         │                     │                              │
│    [Not Ready]           [Ready]                            │
│         │                     │                              │
│      [Wait]            [Train New Model]                    │
│                              │                               │
│                              ▼                               │
│                    [Evaluate on Benchmark]                   │
│                              │                               │
│                    [Compare with Champion]                   │
│                              │                               │
│              ┌───────────────┴───────────────┐              │
│              │                               │               │
│        [Challenger Wins]              [Champion Wins]        │
│              │                               │               │
│        [Deploy New]                    [Keep Current]        │
│        [Archive Old]                   [Log Results]         │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

## Kriteria Keputusan

Model baru akan di-deploy jika memenuhi kriteria:

| Metric | Weight | Condition |
|--------|--------|-----------|
| Magnitude Accuracy | 40% | ≥ champion |
| Azimuth Accuracy | 20% | ≥ champion - 2% |
| LOEO Validation | 30% | ≥ champion - 1% |
| False Positive Rate | 10% | ≤ champion |

**Composite Score** = Σ(weight × normalized_metric)

## Referensi

- [Self-evolving AI for Earthquake Prediction (PMC11415515)](https://pmc.ncbi.nlm.nih.gov/articles/PMC11415515/)
- [MLOps Continuous Training Best Practices](https://cloud.google.com/architecture/mlops-continuous-delivery-and-automation-pipelines-in-machine-learning)
- [Champion-Challenger Pattern](https://docs.microsoft.com/en-us/azure/machine-learning/concept-model-management-and-deployment)

## File Structure

```
autoupdate_pipeline/
├── README.md                    # Dokumentasi utama (ini)
├── requirements.txt             # Dependencies
├── config/
│   ├── pipeline_config.yaml     # Konfigurasi pipeline
│   └── model_registry.json      # Registry model
├── src/
│   ├── __init__.py
│   ├── data_ingestion.py        # Modul ingestion data
│   ├── data_validator.py        # Validasi data gempa
│   ├── trainer.py               # Training model
│   ├── evaluator.py             # Evaluasi model
│   ├── model_comparator.py      # Champion vs Challenger
│   ├── deployer.py              # Deployment model
│   └── utils.py                 # Utility functions
├── scripts/
│   ├── run_pipeline.py          # Main pipeline runner
│   ├── add_new_event.py         # Tambah event baru
│   ├── check_status.py          # Cek status pipeline
│   ├── rollback_model.py        # Rollback model
│   └── setup_champion.py        # Setup champion awal
├── data/
│   ├── pending/                 # Data menunggu validasi
│   ├── validated/               # Data tervalidasi
│   └── benchmark/               # Fixed benchmark test
├── models/
│   ├── champion/                # Model production
│   ├── challenger/              # Model kandidat
│   └── archive/                 # Model lama
├── logs/
│   └── pipeline_history.json    # History pipeline
├── tests/
│   └── test_pipeline.py         # Unit tests
└── docs/
    ├── ARCHITECTURE.md          # Arsitektur sistem
    ├── USER_GUIDE.md            # Panduan pengguna
    └── API_REFERENCE.md         # Referensi API
```

## Author

Earthquake Prediction Research Team  
Version: 1.0.0  
Date: February 2026
