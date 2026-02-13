# 📦 Skenario Manajemen Model Auto-Update Pipeline

## Daftar Isi
1. [Gambaran Umum](#1-gambaran-umum)
2. [Struktur Penyimpanan Model](#2-struktur-penyimpanan-model)
3. [Skenario Update Model](#3-skenario-update-model)
4. [Prioritas Model untuk Operasional](#4-prioritas-model-untuk-operasional)
5. [Analisis dengan Model Lama](#5-analisis-dengan-model-lama)
6. [Rollback dan Recovery](#6-rollback-dan-recovery)
7. [Integrasi dengan Dashboard](#7-integrasi-dengan-dashboard)

---

## 1. Gambaran Umum

### Prinsip Utama

```
┌─────────────────────────────────────────────────────────────────┐
│                    PRINSIP MANAJEMEN MODEL                       │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  1. MODEL LAMA TIDAK PERNAH DIHAPUS                             │
│     → Selalu di-archive untuk analisis dan rollback             │
│                                                                  │
│  2. MODEL TERBAIK = CHAMPION                                     │
│     → Digunakan untuk operasional (scanner, prediksi)           │
│                                                                  │
│  3. SEMUA MODEL BISA DIAKSES                                    │
│     → Via dashboard untuk perbandingan dan analisis             │
│                                                                  │
│  4. VERSIONING SEMANTIK                                          │
│     → v1.0.0 → v1.0.1 → v1.0.2 → ...                           │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### Lifecycle Model

```
┌──────────┐    ┌────────────┐    ┌──────────┐    ┌─────────┐
│ Training │ → │ Challenger │ → │ Champion │ → │ Archive │
└──────────┘    └────────────┘    └──────────┘    └─────────┘
     │               │                 │              │
     │               │                 │              │
   Baru          Kandidat         Operasional     Tersimpan
   dilatih       menunggu         (aktif)         (backup)
                 evaluasi
```

---

## 2. Struktur Penyimpanan Model

### Struktur Folder

```
autoupdate_pipeline/
├── models/
│   ├── champion/                    # Model aktif untuk operasional
│   │   ├── best_model.pth          # Weights model
│   │   ├── class_mappings.json     # Mapping kelas
│   │   ├── training_config.json    # Konfigurasi training
│   │   └── metadata.json           # Info model (versi, tanggal, metrik)
│   │
│   ├── challenger/                  # Model kandidat (sementara)
│   │   ├── best_model.pth
│   │   ├── class_mappings.json
│   │   └── training_history.csv
│   │
│   └── archive/                     # Semua model lama
│       ├── convnext_v1.0.0/        # Versi pertama
│       │   ├── best_model.pth
│       │   ├── class_mappings.json
│       │   ├── metadata.json
│       │   └── evaluation_results.json
│       │
│       ├── convnext_v1.0.1/        # Versi kedua
│       │   └── ...
│       │
│       ├── convnext_v1.0.2/        # Versi ketiga
│       │   └── ...
│       │
│       └── convnext_v1.0.1_pre_rollback/  # Backup sebelum rollback
│           └── ...
│
├── config/
│   └── model_registry.json          # Registry semua model
│
└── production/                      # Symlink ke champion
    └── models/
        └── earthquake_model.pth → ../autoupdate_pipeline/models/champion/best_model.pth
```

### Model Registry (model_registry.json)

```json
{
  "registry_version": "2.0.0",
  "last_updated": "2026-02-10T15:00:00",
  
  "champion": {
    "model_id": "convnext_v1.0.2",
    "version": "1.0.2",
    "architecture": "convnext_tiny",
    "path": "models/champion/best_model.pth",
    "deployed_at": "2026-02-10T15:00:00",
    "metrics": {
      "magnitude_accuracy": 98.50,
      "azimuth_accuracy": 72.30,
      "composite_score": 0.9350
    },
    "training_data": {
      "total_samples": 2100,
      "events_included": ["GTO_20260210", "SCN_20260211", ...]
    },
    "status": "active"
  },
  
  "challenger": null,
  
  "archive": [
    {
      "model_id": "convnext_v1.0.0",
      "version": "1.0.0",
      "archived_at": "2026-02-09T13:00:00",
      "path": "models/archive/convnext_v1.0.0",
      "metrics": {
        "magnitude_accuracy": 98.36,
        "azimuth_accuracy": 50.66,
        "composite_score": 0.8234
      },
      "training_data": {
        "total_samples": 1972,
        "events_included": []
      },
      "reason": "Replaced by v1.0.1 (better azimuth)"
    },
    {
      "model_id": "convnext_v1.0.1",
      "version": "1.0.1",
      "archived_at": "2026-02-10T15:00:00",
      "path": "models/archive/convnext_v1.0.1",
      "metrics": {
        "magnitude_accuracy": 97.63,
        "azimuth_accuracy": 71.28,
        "composite_score": 0.9260
      },
      "training_data": {
        "total_samples": 1972,
        "events_included": []
      },
      "reason": "Replaced by v1.0.2 (new events added)"
    }
  ],
  
  "all_versions": [
    {"version": "1.0.0", "status": "archived", "path": "models/archive/convnext_v1.0.0"},
    {"version": "1.0.1", "status": "archived", "path": "models/archive/convnext_v1.0.1"},
    {"version": "1.0.2", "status": "champion", "path": "models/champion"}
  ]
}
```

---

## 3. Skenario Update Model

### Skenario 1: Challenger Menang → Promosi ke Champion

```
┌─────────────────────────────────────────────────────────────────┐
│  SEBELUM UPDATE                                                  │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  Champion: v1.0.1 (Mag: 97.63%, Azi: 71.28%)                    │
│  Challenger: v1.0.2 (Mag: 98.50%, Azi: 72.30%) ← MENANG        │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│  PROSES UPDATE                                                   │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  1. Archive champion lama (v1.0.1)                              │
│     models/champion/ → models/archive/convnext_v1.0.1/          │
│                                                                  │
│  2. Promosi challenger ke champion                               │
│     models/challenger/ → models/champion/                        │
│                                                                  │
│  3. Update registry                                              │
│     champion = v1.0.2                                            │
│     archive += v1.0.1                                            │
│                                                                  │
│  4. Clear challenger folder                                      │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│  SESUDAH UPDATE                                                  │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  Champion: v1.0.2 (Mag: 98.50%, Azi: 72.30%) ← AKTIF           │
│  Archive:                                                        │
│    - v1.0.0 (tersimpan, bisa diakses)                           │
│    - v1.0.1 (tersimpan, bisa diakses)                           │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### Skenario 2: Challenger Kalah → Tetap Champion Lama

```
┌─────────────────────────────────────────────────────────────────┐
│  HASIL EVALUASI                                                  │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  Champion: v1.0.1 (Mag: 97.63%, Score: 0.926) ← TETAP          │
│  Challenger: v1.0.2 (Mag: 96.50%, Score: 0.910) ← KALAH        │
│                                                                  │
│  Keputusan: REJECT CHALLENGER                                    │
│  Alasan: Composite score lebih rendah                            │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│  AKSI                                                            │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  1. Champion tetap v1.0.1 (tidak berubah)                       │
│                                                                  │
│  2. Challenger di-archive sebagai "rejected"                     │
│     models/challenger/ → models/archive/convnext_v1.0.2_rejected│
│                                                                  │
│  3. Log alasan penolakan                                         │
│                                                                  │
│  4. Validated events dikembalikan ke pool                        │
│     (bisa digunakan untuk training berikutnya)                   │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### Skenario 3: Rollback ke Versi Lama

```
┌─────────────────────────────────────────────────────────────────┐
│  SITUASI: Model v1.0.2 bermasalah di production                 │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  Champion saat ini: v1.0.2 (ada masalah)                        │
│  Target rollback: v1.0.1 (stabil)                               │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│  PROSES ROLLBACK                                                 │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  1. Backup champion bermasalah                                   │
│     models/champion/ → models/archive/convnext_v1.0.2_pre_rollback│
│                                                                  │
│  2. Restore dari archive                                         │
│     models/archive/convnext_v1.0.1/ → models/champion/          │
│                                                                  │
│  3. Update registry                                              │
│     champion = v1.0.1                                            │
│     rollback_from = v1.0.2                                       │
│                                                                  │
│  4. Log rollback event                                           │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

---

## 4. Prioritas Model untuk Operasional

### Hierarki Prioritas

```
┌─────────────────────────────────────────────────────────────────┐
│                    PRIORITAS MODEL                               │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  PRIORITAS 1: CHAMPION (Default untuk semua operasi)            │
│  ├── Scanner production                                          │
│  ├── Dashboard predictions                                       │
│  └── API endpoints                                               │
│                                                                  │
│  PRIORITAS 2: ARCHIVED MODELS (Untuk analisis/perbandingan)     │
│  ├── Perbandingan performa antar versi                          │
│  ├── Analisis regresi                                            │
│  └── Debugging                                                   │
│                                                                  │
│  PRIORITAS 3: CHALLENGER (Hanya saat evaluasi)                  │
│  └── Evaluasi sebelum promosi                                    │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### Konfigurasi di Dashboard

```python
# Di project_dashboard_v2.py

AVAILABLE_MODELS = {
    # Champion (default, prioritas tertinggi)
    "🏆 Champion v1.0.2 (98.50% Mag)": {
        "name": "Champion",
        "path": "autoupdate_pipeline/models/champion/best_model.pth",
        "priority": 1,
        "status": "active"
    },
    
    # Archived models (untuk analisis)
    "📦 Archive v1.0.1 (97.63% Mag)": {
        "name": "Archive v1.0.1",
        "path": "autoupdate_pipeline/models/archive/convnext_v1.0.1/best_model.pth",
        "priority": 2,
        "status": "archived"
    },
    "📦 Archive v1.0.0 (98.36% Mag)": {
        "name": "Archive v1.0.0",
        "path": "autoupdate_pipeline/models/archive/convnext_v1.0.0/best_model.pth",
        "priority": 2,
        "status": "archived"
    }
}
```

---

## 5. Analisis dengan Model Lama

### Use Cases

| Use Case | Model yang Digunakan | Tujuan |
|----------|---------------------|--------|
| Prediksi operasional | Champion | Hasil terbaik |
| Perbandingan performa | Champion + Archives | Lihat improvement |
| Debug false positive | Semua versi | Cari penyebab |
| Validasi event baru | Champion + Archive terbaik | Cross-check |
| Penelitian | Semua versi | Analisis evolusi model |

### Fitur Dashboard untuk Analisis

```
┌─────────────────────────────────────────────────────────────────┐
│  DASHBOARD: Model Comparison                                     │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐ │
│  │ Select Model 1  │  │ Select Model 2  │  │    Compare      │ │
│  │ [Champion v1.0.2]│  │ [Archive v1.0.1]│  │    [Button]     │ │
│  └─────────────────┘  └─────────────────┘  └─────────────────┘ │
│                                                                  │
│  ┌─────────────────────────────────────────────────────────────┐│
│  │  COMPARISON RESULTS                                          ││
│  │                                                               ││
│  │  Metric          │ v1.0.2    │ v1.0.1    │ Diff             ││
│  │  ─────────────────────────────────────────────────────────  ││
│  │  Magnitude Acc   │ 98.50%    │ 97.63%    │ +0.87% ✅        ││
│  │  Azimuth Acc     │ 72.30%    │ 71.28%    │ +1.02% ✅        ││
│  │  Composite Score │ 0.9350    │ 0.9260    │ +0.0090 ✅       ││
│  │                                                               ││
│  └─────────────────────────────────────────────────────────────┘│
│                                                                  │
│  ┌─────────────────────────────────────────────────────────────┐│
│  │  TEST WITH SPECIFIC EVENT                                    ││
│  │                                                               ││
│  │  Event: [SCN_20180117]  [Run Both Models]                   ││
│  │                                                               ││
│  │  v1.0.2: Large (95.2%)  │  v1.0.1: Large (93.8%)            ││
│  │                                                               ││
│  └─────────────────────────────────────────────────────────────┘│
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### API untuk Analisis

```python
# Contoh penggunaan untuk analisis

from autoupdate_pipeline.src.deployer import ModelDeployer
from autoupdate_pipeline.src.utils import load_registry

# List semua model tersedia
deployer = ModelDeployer()
archived = deployer.list_archived_models()
champion = deployer.get_current_champion()

print("Champion:", champion['model_id'])
print("Archived models:")
for model in archived:
    print(f"  - {model['model_id']}: {model['metrics']}")

# Load model spesifik untuk analisis
def load_model_for_analysis(version: str):
    registry = load_registry()
    
    if version == "champion":
        return registry['champion']['path']
    
    for archived in registry['archive']:
        if version in archived['model_id']:
            return archived['path']
    
    return None
```

---

## 6. Rollback dan Recovery

### Kapan Rollback Diperlukan

| Situasi | Aksi | Prioritas |
|---------|------|-----------|
| False positive meningkat drastis | Rollback segera | CRITICAL |
| Akurasi turun di production | Rollback + investigasi | HIGH |
| Bug di model baru | Rollback + fix | HIGH |
| User request | Rollback manual | MEDIUM |

### Prosedur Rollback

```bash
# 1. Lihat model tersedia
python scripts/rollback_model.py --list

# Output:
# Available models for rollback:
#   1. convnext_v1.0.1 (archived: 2026-02-10)
#      Mag: 97.63%, Azi: 71.28%
#   2. convnext_v1.0.0 (archived: 2026-02-09)
#      Mag: 98.36%, Azi: 50.66%

# 2. Rollback ke versi spesifik
python scripts/rollback_model.py --version 1.0.1

# 3. Atau rollback ke versi terakhir
python scripts/rollback_model.py
```

### Recovery dari Kegagalan

```
┌─────────────────────────────────────────────────────────────────┐
│  RECOVERY SCENARIOS                                              │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  Skenario 1: Deployment gagal di tengah jalan                   │
│  ─────────────────────────────────────────────────────────────  │
│  → Champion lama sudah di-archive                                │
│  → Challenger belum selesai di-copy                              │
│  Solusi: Rollback otomatis ke archive terakhir                   │
│                                                                  │
│  Skenario 2: Model corrupt setelah deployment                   │
│  ─────────────────────────────────────────────────────────────  │
│  → Champion baru tidak bisa di-load                              │
│  Solusi: Rollback manual + investigasi                           │
│                                                                  │
│  Skenario 3: Registry corrupt                                    │
│  ─────────────────────────────────────────────────────────────  │
│  → model_registry.json rusak                                     │
│  Solusi: Restore dari backup atau rebuild dari folder structure  │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

---

## 7. Integrasi dengan Dashboard

### Model Selection di Dashboard

Dashboard akan menampilkan semua model dengan prioritas:

```python
# Pseudocode untuk dashboard model selection

def get_available_models():
    registry = load_registry()
    models = []
    
    # 1. Champion (prioritas 1)
    champion = registry['champion']
    models.append({
        "display": f"🏆 Champion v{champion['version']} ({champion['metrics']['magnitude_accuracy']:.1f}% Mag)",
        "path": champion['path'],
        "priority": 1,
        "is_default": True
    })
    
    # 2. Archived models (prioritas 2)
    for archived in registry['archive']:
        models.append({
            "display": f"📦 Archive v{archived['model_id'].split('_v')[-1]} ({archived['metrics'].get('magnitude_accuracy', 0):.1f}% Mag)",
            "path": archived['path'],
            "priority": 2,
            "is_default": False
        })
    
    # Sort by priority
    return sorted(models, key=lambda x: x['priority'])
```

### Visualisasi di Dashboard

```
┌─────────────────────────────────────────────────────────────────┐
│  🔄 Auto-Update Pipeline > Model Management                     │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌─────────────────────────────────────────────────────────────┐│
│  │  🏆 CHAMPION MODEL (Active)                                  ││
│  │  ─────────────────────────────────────────────────────────  ││
│  │  Version: v1.0.2                                             ││
│  │  Deployed: 2026-02-10 15:00                                  ││
│  │  Magnitude: 98.50%  │  Azimuth: 72.30%                      ││
│  │  Composite Score: 0.9350                                     ││
│  │  Training Data: 2100 samples                                 ││
│  │                                                               ││
│  │  [Use for Scanner] [View Details]                            ││
│  └─────────────────────────────────────────────────────────────┘│
│                                                                  │
│  ┌─────────────────────────────────────────────────────────────┐│
│  │  📦 ARCHIVED MODELS                                          ││
│  │  ─────────────────────────────────────────────────────────  ││
│  │                                                               ││
│  │  Version │ Archived    │ Mag Acc │ Azi Acc │ Actions        ││
│  │  ────────────────────────────────────────────────────────── ││
│  │  v1.0.1  │ 2026-02-10  │ 97.63%  │ 71.28%  │ [Use] [Rollback]│
│  │  v1.0.0  │ 2026-02-09  │ 98.36%  │ 50.66%  │ [Use] [Rollback]│
│  │                                                               ││
│  └─────────────────────────────────────────────────────────────┘│
│                                                                  │
│  ┌─────────────────────────────────────────────────────────────┐│
│  │  📊 VERSION COMPARISON                                       ││
│  │  ─────────────────────────────────────────────────────────  ││
│  │                                                               ││
│  │  [Chart: Accuracy over versions]                             ││
│  │                                                               ││
│  │  v1.0.0 ████████████████████░░░░ 98.36%                     ││
│  │  v1.0.1 ███████████████████░░░░░ 97.63%                     ││
│  │  v1.0.2 ████████████████████░░░░ 98.50% ← Current           ││
│  │                                                               ││
│  └─────────────────────────────────────────────────────────────┘│
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

---

## Ringkasan

### Poin Penting

1. **Model lama TIDAK PERNAH dihapus** - selalu di-archive
2. **Champion = model terbaik** untuk operasional
3. **Semua versi bisa diakses** untuk analisis
4. **Rollback mudah** jika ada masalah
5. **Dashboard terintegrasi** untuk manajemen visual

### File Penting

| File | Fungsi |
|------|--------|
| `models/champion/` | Model aktif untuk operasional |
| `models/archive/` | Semua model lama |
| `config/model_registry.json` | Registry semua model |
| `src/deployer.py` | Logic deployment dan rollback |

---

## 8. Status Implementasi

### ✅ IMPLEMENTASI SELESAI (10 Februari 2026)

| Komponen | Status | Keterangan |
|----------|--------|------------|
| Struktur folder `models/` | ✅ Done | champion/, challenger/, archive/ |
| Model Registry v2.0 | ✅ Done | Dengan `all_versions` dan `reason` |
| Metadata.json | ✅ Done | Untuk champion dan archive |
| Dashboard Model Management | ✅ Done | Tab baru di Auto-Update Pipeline |
| Rollback via Dashboard | ✅ Done | Button rollback terintegrasi |
| Version Comparison Chart | ✅ Done | Bar chart perbandingan metrik |
| Deployer API | ✅ Done | Fungsi baru untuk manajemen model |

### Cara Mengakses

1. **Via Dashboard**:
   ```bash
   streamlit run project_dashboard_v2.py
   ```
   Navigasi ke: `🔄 Auto-Update Pipeline` → `📦 Model Management`

2. **Via CLI**:
   ```bash
   # List archived models
   python -c "from autoupdate_pipeline.src.deployer import ModelDeployer; d=ModelDeployer(); print(d.list_archived_models())"
   
   # Rollback
   python autoupdate_pipeline/scripts/rollback_model.py --version 1.0.0
   ```

3. **Via Python API**:
   ```python
   from autoupdate_pipeline.src.deployer import ModelDeployer
   
   deployer = ModelDeployer()
   
   # Get champion
   champion = deployer.get_current_champion()
   
   # Get all versions
   versions = deployer.get_all_versions()
   
   # Compare models
   comparison = deployer.compare_models("1.0.0", "1.0.1")
   
   # Rollback
   result = deployer.rollback("1.0.0")
   ```

---

*Dokumentasi Skenario Manajemen Model*
*Terakhir diperbarui: 10 Februari 2026*
