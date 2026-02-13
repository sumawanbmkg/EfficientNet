# 📖 Panduan Operasional Auto-Update Pipeline

## Daftar Isi
1. [Gambaran Umum](#1-gambaran-umum)
2. [Persiapan Awal](#2-persiapan-awal)
3. [Operasi Harian](#3-operasi-harian)
4. [Menambah Event Baru](#4-menambah-event-baru)
5. [Menjalankan Pipeline](#5-menjalankan-pipeline)
6. [Monitoring dan Troubleshooting](#6-monitoring-dan-troubleshooting)
7. [Rollback Model](#7-rollback-model)
8. [Penjadwalan Otomatis](#8-penjadwalan-otomatis)
9. [Dashboard Web](#9-dashboard-web-baru)
10. [Model Management (BARU!)](#10-model-management-baru)

---

## 1. Gambaran Umum

### Apa itu Auto-Update Pipeline?

Pipeline ini mengotomatisasi proses update model deteksi prekursor gempa bumi dengan pola **Champion-Challenger**:

```
┌─────────────────────────────────────────────────────────────────┐
│                    ALUR PIPELINE                                 │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  [Event Baru] → [Validasi] → [Training] → [Evaluasi] → [Deploy] │
│       ↓            ↓            ↓            ↓            ↓     │
│   add_event    validate    train_model   compare     deploy     │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### Komponen Utama

| Komponen | Fungsi |
|----------|--------|
| **Champion** | Model production saat ini |
| **Challenger** | Model baru yang dilatih |
| **Evaluator** | Mengevaluasi performa model |
| **Comparator** | Membandingkan Champion vs Challenger |
| **Deployer** | Deploy model baru jika menang |
| **Dashboard** | Interface web untuk operasi pipeline |

### Dua Cara Operasi

Pipeline dapat dioperasikan dengan dua cara:

| Metode | Kelebihan | Kapan Digunakan |
|--------|-----------|-----------------|
| **Command Line** | Cepat, scriptable | Automasi, server tanpa GUI |
| **Dashboard Web** | Visual, mudah dipahami | Operasi harian, monitoring |

---

## 2. Persiapan Awal

### 2.1 Cek Status Pipeline

```bash
cd autoupdate_pipeline
python scripts/check_status.py
```

Output akan menampilkan:
- Status model Champion saat ini
- Jumlah event pending/validated
- Status trigger pipeline
- History pipeline

### 2.2 Setup Champion Model (Jika Belum Ada)

```bash
python scripts/setup_champion.py
```

Script ini akan:
1. Copy model production ke folder champion
2. Membuat class mappings
3. Update registry

### 2.3 Verifikasi Konfigurasi

File konfigurasi: `config/pipeline_config.yaml`

Parameter penting:
```yaml
triggers:
  min_new_events: 5        # Minimum event untuk trigger
  max_days_between_training: 90  # Max hari tanpa training

evaluation:
  weights:
    magnitude_accuracy: 0.35
    azimuth_accuracy: 0.15
    macro_f1: 0.20
    mcc: 0.15
    loeo_stability: 0.10
    false_positive_rate: 0.05
```

---

## 3. Operasi Harian

### 3.1 Cek Status Harian

```bash
python scripts/daily_check.py
```

### 3.2 Workflow Harian

```
┌─────────────────────────────────────────────────────────────────┐
│                    WORKFLOW HARIAN                               │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  1. Cek status: python scripts/check_status.py                  │
│                                                                  │
│  2. Jika ada event baru:                                        │
│     python scripts/add_new_event.py add -d DATE -s STA -m M -a A│
│                                                                  │
│  3. Validasi pending events:                                    │
│     python scripts/add_new_event.py validate                    │
│                                                                  │
│  4. Jika trigger ready:                                         │
│     python scripts/run_pipeline.py                              │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

---

## 4. Menambah Event Baru

### 4.1 Format Input

```bash
python scripts/add_new_event.py add \
    --date 2026-02-15 \
    --station GTO \
    --magnitude 6.2 \
    --azimuth 45
```

Parameter:
- `--date, -d`: Tanggal event (YYYY-MM-DD)
- `--station, -s`: Kode stasiun (GTO, SCN, MLB, dll)
- `--magnitude, -m`: Magnitudo gempa (0-10)
- `--azimuth, -a`: Azimuth dalam derajat (0-360)

### 4.2 Klasifikasi Otomatis

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
| 337.5° - 22.5° | N |
| 22.5° - 67.5° | NE |
| 67.5° - 112.5° | E |
| 112.5° - 157.5° | SE |
| 157.5° - 202.5° | S |
| 202.5° - 247.5° | SW |
| 247.5° - 292.5° | W |
| 292.5° - 337.5° | NW |

### 4.3 Contoh Penambahan Event

```bash
# Gempa besar M6.2 arah NE
python scripts/add_new_event.py add -d 2026-02-15 -s GTO -m 6.2 -a 45

# Gempa sedang M5.5 arah S
python scripts/add_new_event.py add -d 2026-02-16 -s SCN -m 5.5 -a 180

# Normal (tidak ada gempa)
python scripts/add_new_event.py add -d 2026-02-17 -s TRT -m 0 -a 0
```

### 4.4 Validasi Event

```bash
# Validasi semua pending events
python scripts/add_new_event.py validate

# Lihat daftar events
python scripts/add_new_event.py list
```

### 4.5 Menghapus Event

```bash
# Hapus pending event
python scripts/add_new_event.py delete --id GTO_20260215

# Hapus validated event
python scripts/add_new_event.py delete --id GTO_20260215 --validated
```

---

## 5. Menjalankan Pipeline

### 5.1 Mode Normal

```bash
# Jalankan jika kondisi trigger terpenuhi
python scripts/run_pipeline.py
```

### 5.2 Mode Force

```bash
# Paksa jalankan meskipun kondisi belum terpenuhi
python scripts/run_pipeline.py --force
```

### 5.3 Mode Quick Test

```bash
# Test cepat (~3-4 menit) dengan dataset kecil
python scripts/run_pipeline.py --force --quick-test
```

### 5.4 Auto-Deploy

```bash
# Otomatis deploy jika challenger menang
python scripts/run_pipeline.py --force --auto-deploy
```

### 5.5 Tahapan Pipeline

| Stage | Deskripsi | Waktu Normal | Waktu Quick Test |
|-------|-----------|--------------|------------------|
| 1. Trigger Check | Cek kondisi trigger | < 1 detik | < 1 detik |
| 2. Training | Train challenger model | ~30-60 menit | ~2 menit |
| 3. Evaluation | Evaluasi challenger | ~5 menit | ~30 detik |
| 4. Comparison | Bandingkan dengan champion | ~2 menit | ~1 menit |
| 5. Deployment | Deploy jika challenger menang | < 1 detik | < 1 detik |

---

## 6. Monitoring dan Troubleshooting

### 6.1 Log Files

```
autoupdate_pipeline/
├── logs/
│   ├── pipeline.log           # Log utama pipeline
│   └── pipeline_history.json  # History semua event
```

### 6.2 Cek Log

```bash
# Windows
type logs\pipeline.log

# Atau buka dengan editor
notepad logs\pipeline.log
```

### 6.3 Common Issues

**Issue: "Spectrogram not found"**
- Pastikan path spectrogram benar
- Cek apakah file ada di `dataset_unified/spectrograms/`

**Issue: "Validation failed"**
- Cek format tanggal (YYYY-MM-DD)
- Cek kode stasiun valid
- Cek range magnitude (0-10) dan azimuth (0-360)

**Issue: "Training failed"**
- Cek GPU/CUDA tersedia
- Cek memory cukup
- Coba dengan `--quick-test` untuk debug

### 6.4 Reset Pipeline

```bash
# Clear semua pending events
python scripts/add_new_event.py clear --confirm

# Clear validated events
python scripts/add_new_event.py clear --validated --confirm
```

---

## 7. Rollback Model

### 7.1 Lihat Model Tersedia

```bash
python scripts/rollback_model.py --list
```

### 7.2 Rollback ke Versi Sebelumnya

```bash
# Rollback ke versi terakhir yang di-archive
python scripts/rollback_model.py

# Rollback ke versi spesifik
python scripts/rollback_model.py --version 1.0.0
```

---

## 8. Penjadwalan Otomatis

### 8.1 Windows Task Scheduler

1. Buka **Task Scheduler**
2. Klik **Create Basic Task**
3. Nama: "Earthquake Model Auto-Update"
4. Trigger: **Daily** atau **Weekly**
5. Action: **Start a program**
6. Program: `path\to\autoupdate_pipeline\scripts\run_scheduled.bat`
7. Start in: `path\to\autoupdate_pipeline`

### 8.2 Linux Cron Job

```bash
# Edit crontab
crontab -e

# Tambahkan (jalankan setiap hari jam 2 pagi)
0 2 * * * cd /path/to/autoupdate_pipeline && python scripts/daily_check.py --run --auto-deploy >> logs/cron.log 2>&1
```

### 8.3 Script Terjadwal

```bash
# Windows
scripts\run_scheduled.bat

# Linux/Mac
python scripts/daily_check.py --run --auto-deploy
```

---

## Referensi Cepat

### Perintah Utama

```bash
# Status
python scripts/check_status.py

# Tambah event
python scripts/add_new_event.py add -d DATE -s STATION -m MAG -a AZI

# Validasi
python scripts/add_new_event.py validate

# Jalankan pipeline
python scripts/run_pipeline.py --force --quick-test

# Rollback
python scripts/rollback_model.py --list
```

### Stasiun Valid

```
SBG, SCN, KPY, LWA, LPS, SRG, SKB, CLP, YOG, TRT, 
LUT, ALR, SMI, SRO, TNT, TND, GTO, LWK, PLU, TRD, 
JYP, AMB, GSI, MLB
```

### Kriteria Keputusan

Model baru (Challenger) di-promote jika:
1. ✅ Composite score lebih tinggi
2. ✅ Improvement ≥ 0.5%
3. ✅ Statistik signifikan (p < 0.05)
4. ✅ Tidak ada regresi berbahaya

---

## 9. Dashboard Web (BARU!)

### 9.1 Menjalankan Dashboard

```bash
# Dari folder utama project
streamlit run project_dashboard_v2.py
```

Dashboard akan terbuka di browser: `http://localhost:8501`

### 9.2 Mengakses Menu Auto-Update Pipeline

1. Buka dashboard di browser
2. Di sidebar kiri, pilih menu **"🔄 Auto-Update Pipeline"**

### 9.3 Fitur Dashboard

#### 📊 Status Overview
Menampilkan ringkasan status pipeline:
- Champion Model (versi dan akurasi)
- Pending Events (jumlah event menunggu validasi)
- Validated Events (jumlah event siap training)
- Pipeline Status (READY/WAITING)

#### 📈 Progress Menuju Update Model
**Fitur utama untuk melihat status threshold:**

```
┌─────────────────────────────────────────────────────────────────┐
│  Progress Bar: [████████████░░░░░░░░] 60%                       │
│  Validated Events: 3 / 5 (threshold)                            │
│                                                                  │
│  ┌─────────────┐ ┌─────────────┐ ┌─────────────┐                │
│  │ ❌ Event    │ │ ⏳ Time     │ │ ⏳ Keputusan│                │
│  │ Threshold   │ │ Threshold   │ │             │                │
│  │ 3/5 events  │ │ 1/90 hari   │ │ BELUM SIAP  │                │
│  │ BELUM       │ │ BELUM       │ │             │                │
│  └─────────────┘ └─────────────┘ └─────────────┘                │
└─────────────────────────────────────────────────────────────────┘
```

- **Progress Bar**: Menunjukkan persentase threshold tercapai
- **Event Threshold**: Jumlah event saat ini vs minimum yang diperlukan
- **Time Threshold**: Hari sejak update terakhir vs maximum hari
- **Keputusan**: Apakah pipeline siap dijalankan

#### 📋 Event Management
Tab untuk mengelola events:

| Sub-menu | Fungsi |
|----------|--------|
| **➕ Add New Event** | Form untuk menambah event baru dengan input numerik |
| **📋 View Events** | Melihat daftar pending dan validated events |
| **✅ Validate Events** | Tombol untuk memvalidasi semua pending events |

**Menambah Event via Dashboard:**
1. Pilih tab "📋 Event Management"
2. Pilih "➕ Add New Event"
3. Isi form:
   - Tanggal Event
   - Stasiun (dropdown)
   - Magnitude (0-10)
   - Azimuth (0-360°)
4. Lihat preview klasifikasi otomatis
5. Klik "➕ Tambah Event"

#### 🚀 Run Pipeline
Tab untuk menjalankan pipeline:
- Melihat status trigger
- Opsi: Force Run, Quick Test, Auto Deploy
- Generate command untuk terminal

#### 📈 Evaluation Results
Tab untuk melihat hasil evaluasi:
- Perbandingan Champion vs Challenger
- Chart perbandingan metrik
- Formula evaluasi dengan referensi

#### 📜 Pipeline History
Tab untuk melihat history:
- Log event pipeline
- Daftar archived models
- Fungsi rollback

#### ⚙️ Configuration
Tab untuk melihat konfigurasi:
- Trigger settings
- Evaluation weights
- Quick commands
- Classification reference

### 9.4 Workflow via Dashboard

```
┌─────────────────────────────────────────────────────────────────┐
│                 WORKFLOW VIA DASHBOARD                           │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  1. Buka Dashboard → Menu "🔄 Auto-Update Pipeline"             │
│                                                                  │
│  2. Cek "📈 Progress Menuju Update Model"                       │
│     - Lihat progress bar                                        │
│     - Cek status threshold                                      │
│                                                                  │
│  3. Jika belum tercapai:                                        │
│     - Pilih "📋 Event Management" → "➕ Add New Event"          │
│     - Isi form dan submit                                       │
│     - Pilih "✅ Validate Events" untuk validasi                 │
│                                                                  │
│  4. Jika threshold tercapai (progress bar 100%):                │
│     - Pilih "🚀 Run Pipeline"                                   │
│     - Centang opsi yang diinginkan                              │
│     - Jalankan command di terminal                              │
│                                                                  │
│  5. Lihat hasil di "📈 Evaluation Results"                      │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### 9.5 Perbandingan: Dashboard vs Command Line

| Operasi | Dashboard | Command Line |
|---------|-----------|--------------|
| Cek status | Langsung terlihat di Status Overview | `python scripts/check_status.py` |
| Cek threshold | Progress bar visual | Output text |
| Tambah event | Form interaktif | `python scripts/add_new_event.py add ...` |
| Validasi | Tombol "Validate Events" | `python scripts/add_new_event.py validate` |
| Run pipeline | Generate command | `python scripts/run_pipeline.py` |
| Lihat history | Tab "Pipeline History" | `type logs/pipeline_history.json` |

### 9.6 Tips Penggunaan Dashboard

1. **Refresh otomatis**: Dashboard akan refresh saat ada perubahan
2. **Rerun**: Klik tombol "Rerun" di kanan atas untuk refresh manual
3. **Multiple tabs**: Bisa buka beberapa tab browser untuk monitoring
4. **Mobile friendly**: Dashboard bisa diakses dari HP/tablet

---

## 10. Model Management (BARU!)

### 10.1 Gambaran Umum

Model Management memungkinkan Anda untuk:
- Melihat semua versi model (Champion dan Archive)
- Membandingkan performa antar versi
- Melakukan rollback ke versi sebelumnya
- Melihat metadata dan history setiap model

### 10.2 Mengakses Model Management

**Via Dashboard:**
1. Buka dashboard: `streamlit run project_dashboard_v2.py`
2. Pilih menu **"🔄 Auto-Update Pipeline"**
3. Pilih tab **"📦 Model Management"**

**Via Command Line:**
```bash
# List semua model
python -c "from autoupdate_pipeline.src.deployer import ModelDeployer; d=ModelDeployer(); print(d.get_all_versions())"

# Lihat champion
python -c "from autoupdate_pipeline.src.deployer import ModelDeployer; d=ModelDeployer(); print(d.get_current_champion())"

# Lihat archived
python -c "from autoupdate_pipeline.src.deployer import ModelDeployer; d=ModelDeployer(); print(d.list_archived_models())"
```

### 10.3 Struktur Penyimpanan Model

```
autoupdate_pipeline/models/
├── champion/                    # Model aktif
│   ├── best_model.pth
│   ├── class_mappings.json
│   └── metadata.json           # Info model
│
├── challenger/                  # Model kandidat (sementara)
│   └── ...
│
└── archive/                     # Model lama
    ├── convnext_v1.0.0/
    │   ├── best_model.pth
    │   ├── class_mappings.json
    │   └── metadata.json
    └── convnext_v1.0.1/
        └── ...
```

### 10.4 Prinsip Manajemen Model

| Prinsip | Penjelasan |
|---------|------------|
| **Model lama TIDAK dihapus** | Selalu di-archive untuk rollback |
| **Champion = model terbaik** | Digunakan untuk operasional |
| **Semua versi bisa diakses** | Untuk analisis dan perbandingan |
| **Rollback mudah** | Satu klik di dashboard |

### 10.5 Rollback Model

**Via Dashboard:**
1. Buka tab "📦 Model Management"
2. Scroll ke section "🔄 Rollback to Previous Version"
3. Pilih versi dari dropdown
4. Klik "🔄 Execute Rollback"

**Via Command Line:**
```bash
# Lihat model tersedia
python scripts/rollback_model.py --list

# Rollback ke versi spesifik
python scripts/rollback_model.py --version 1.0.0

# Rollback ke versi terakhir
python scripts/rollback_model.py
```

### 10.6 Perbandingan Model

Dashboard menampilkan chart perbandingan:
- Magnitude Accuracy per versi
- Azimuth Accuracy per versi
- Composite Score per versi

**Via Python API:**
```python
from autoupdate_pipeline.src.deployer import ModelDeployer

deployer = ModelDeployer()
comparison = deployer.compare_models("1.0.0", "1.0.1")
print(comparison['differences'])
```

---

## Referensi Cepat

### Perintah Utama

```bash
# Status
python scripts/check_status.py

# Tambah event
python scripts/add_new_event.py add -d DATE -s STATION -m MAG -a AZI

# Validasi
python scripts/add_new_event.py validate

# Jalankan pipeline
python scripts/run_pipeline.py --force --quick-test

# Rollback
python scripts/rollback_model.py --list

# Dashboard (BARU!)
streamlit run project_dashboard_v2.py
```

### Stasiun Valid

```
SBG, SCN, KPY, LWA, LPS, SRG, SKB, CLP, YOG, TRT, 
LUT, ALR, SMI, SRO, TNT, TND, GTO, LWK, PLU, TRD, 
JYP, AMB, GSI, MLB
```

### Kriteria Keputusan

Model baru (Challenger) di-promote jika:
1. ✅ Composite score lebih tinggi
2. ✅ Improvement ≥ 0.5%
3. ✅ Statistik signifikan (p < 0.05)
4. ✅ Tidak ada regresi berbahaya

---

*Dokumentasi ini dibuat untuk operasional Auto-Update Pipeline.*
*Terakhir diperbarui: 10 Februari 2026*
