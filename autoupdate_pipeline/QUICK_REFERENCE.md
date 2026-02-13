# 🚀 Quick Reference - Auto-Update Pipeline

## 🖥️ Dashboard Web (Recommended!)

```bash
# Jalankan dashboard
streamlit run project_dashboard_v2.py

# Buka di browser: http://localhost:8501
# Pilih menu: 🔄 Auto-Update Pipeline
```

**Fitur Dashboard:**
- 📊 Status Overview - Lihat status pipeline
- 📈 Progress Bar - Visual threshold progress
- 📋 Event Management - Tambah/validasi events via form
- 🚀 Run Pipeline - Generate command
- 📦 Model Management - Kelola versi model (BARU!)
- 📜 History - Lihat log dan rollback

---

## 📦 Model Management (BARU!)

```bash
# Via Dashboard
# Menu: 🔄 Auto-Update Pipeline → 📦 Model Management

# Via Command Line
# List semua versi
python -c "from autoupdate_pipeline.src.deployer import ModelDeployer; d=ModelDeployer(); print(d.get_all_versions())"

# Rollback ke versi sebelumnya
python scripts/rollback_model.py --list
python scripts/rollback_model.py --version 1.0.0
```

**Fitur Model Management:**
- 🏆 Lihat Champion Model aktif
- 📦 Lihat semua Archived Models
- 📊 Chart perbandingan performa antar versi
- 🔄 Rollback ke versi sebelumnya (satu klik!)

---

## Perintah Utama (Command Line)

```bash
# Cek status
python scripts/check_status.py

# Lihat panduan klasifikasi
python scripts/add_new_event.py guide

# Tambah event (dengan nilai numerik)
python scripts/add_new_event.py add -d 2026-02-15 -s GTO -m 6.2 -a 45

# Lihat events
python scripts/add_new_event.py list

# Validasi pending
python scripts/add_new_event.py validate

# Hapus event
python scripts/add_new_event.py delete --id GTO_20260215

# Jalankan pipeline
python scripts/run_pipeline.py

# Quick test mode (untuk testing cepat ~2 menit)
python scripts/run_pipeline.py --force --quick-test

# Auto-deploy jika challenger menang
python scripts/run_pipeline.py --force --quick-test --auto-deploy

# Rollback
python scripts/rollback_model.py --list
```

## 📊 Evaluation Metrics & Decision Criteria

### Composite Score Calculation

Model baru (Challenger) harus membuktikan diri lebih baik dari model lama (Champion) berdasarkan **weighted composite score**:

| Metric | Weight | Deskripsi |
|--------|--------|-----------|
| Magnitude Accuracy | 35% | Task utama: deteksi magnitudo gempa |
| Azimuth Accuracy | 15% | Prediksi arah gempa |
| Macro F1-Score | 20% | Handle class imbalance |
| MCC (Matthews) | 15% | Robust untuk imbalanced data |
| LOEO Stability | 10% | Generalisasi spasial-temporal |
| False Positive Rate | 5% | Hindari false alarm |

### Decision Rules

```
┌─────────────────────────────────────────────────────────────┐
│                    DECISION FLOW                             │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  1. Challenger Score > Champion Score?                       │
│     └── NO  → REJECT (Keep Champion)                        │
│     └── YES → Continue                                       │
│                                                              │
│  2. Improvement >= 0.5% (min_improvement)?                   │
│     └── NO  → REJECT (Insufficient improvement)             │
│     └── YES → Continue                                       │
│                                                              │
│  3. Statistical Test Significant (p < 0.05)?                 │
│     └── NO  → REJECT (Not statistically significant)        │
│     └── YES → Continue                                       │
│                                                              │
│  4. No Harmful Regressions?                                  │
│     └── NO  → REJECT (Critical metrics degraded)            │
│     └── YES → ACCEPT (Promote Challenger!)                  │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

### Regression Tolerances (No-Harm Principle)

| Metric | Toleransi | Alasan |
|--------|-----------|--------|
| Magnitude Accuracy | Max -1.0% | Task utama, tidak boleh turun signifikan |
| Large Earthquake Recall | Max -2.0% | Safety critical: jangan miss gempa besar |
| False Positive Rate | Max +1.0% | Hindari alarm fatigue |

### Statistical Testing

- **McNemar's Test**: Membandingkan dua classifier pada data yang sama
- **Bootstrap Test**: Confidence interval tanpa asumsi distribusi normal
- **Significance Level**: p < 0.05 (default)

Referensi:
- [arXiv:2506.17442](https://arxiv.org/abs/2506.17442) - Medical AI Monitoring
- [arXiv:2512.18390](https://arxiv.org/abs/2512.18390) - Model Switching Decision

## Quick Test Mode

Quick test mode mempercepat testing dengan:
- Dataset dikurangi ke ~150 samples (50 per kelas)
- Training hanya 3 epochs
- Batch size 8
- num_workers 0 (kompatibel Windows)

```bash
# Test cepat tanpa deploy
python scripts/run_pipeline.py --force --quick-test

# Test cepat dengan auto-deploy
python scripts/run_pipeline.py --force --quick-test --auto-deploy
```

## Contoh Penambahan Event

```bash
# Gempa besar (M=6.2) arah NE (45°)
python scripts/add_new_event.py add -d 2026-02-15 -s GTO -m 6.2 -a 45

# Gempa sedang (M=5.3) arah S (180°)
python scripts/add_new_event.py add -d 2026-02-16 -s SCN -m 5.3 -a 180

# Gempa moderat (M=4.5) arah W (270°)
python scripts/add_new_event.py add -d 2026-02-17 -s MLB -m 4.5 -a 270

# Normal (tidak ada gempa)
python scripts/add_new_event.py add -d 2026-02-18 -s TRT -m 0 -a 0
```

## Klasifikasi Otomatis

### Magnitude
| Nilai | Kelas |
|-------|-------|
| M ≥ 6.0 | Large |
| 5.0 ≤ M < 6.0 | Medium |
| 4.0 ≤ M < 5.0 | Moderate |
| M < 4.0 | Normal |

### Azimuth
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

## Stasiun Valid

SBG, SCN, KPY, LWA, LPS, SRG, SKB, CLP, YOG, TRT, LUT, ALR, SMI, SRO, TNT, TND, GTO, LWK, PLU, TRD, JYP, AMB, GSI, MLB

## Workflow

```
┌─────────────────────────────────────────────────────────────────┐
│  VIA DASHBOARD (Recommended):                                    │
│  Dashboard → 🔄 Auto-Update Pipeline → Lihat Progress Bar       │
│  → Tambah Event → Validate → Run Pipeline (jika 100%)           │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│  VIA COMMAND LINE:                                               │
│  add → pending → validate → validated → (5*) → pipeline → deploy│
└─────────────────────────────────────────────────────────────────┘

* Threshold saat ini: 5 events (untuk testing)
  Production: 20 events
```

## Melihat Status Threshold

**Via Dashboard:**
1. Buka menu "🔄 Auto-Update Pipeline"
2. Lihat section "📈 Progress Menuju Update Model"
3. Progress bar menunjukkan persentase
4. Status TERPENUHI/BELUM TERPENUHI untuk setiap kondisi

**Via Command Line:**
```bash
python scripts/check_status.py
```

## Trigger Conditions
- Validated events ≥ threshold (5 untuk testing, 20 untuk production)
- Atau 90 hari sejak training terakhir

## Pipeline Stages

| Stage | Deskripsi | Waktu (Quick Test) |
|-------|-----------|-------------------|
| 1. Trigger Check | Cek kondisi trigger | < 1 detik |
| 2. Training | Train challenger model | ~2 menit |
| 3. Evaluation | Evaluasi challenger | ~30 detik |
| 4. Comparison | Bandingkan dengan champion | ~1 menit |
| 5. Deployment | Deploy jika challenger menang | < 1 detik |

Total waktu quick test: ~3-4 menit
