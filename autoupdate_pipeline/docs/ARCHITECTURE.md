# 🏗️ Architecture Documentation

## System Overview

Pipeline auto-update ini menggunakan arsitektur modular dengan pola **Champion-Challenger** untuk memastikan model baru hanya di-deploy jika performanya lebih baik dari model saat ini.

## Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        AUTO-UPDATE PIPELINE                              │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐               │
│  │   External   │    │   Data       │    │   Model      │               │
│  │   Sources    │───▶│   Ingestion  │───▶│   Registry   │               │
│  │              │    │              │    │              │               │
│  └──────────────┘    └──────────────┘    └──────────────┘               │
│         │                   │                   │                        │
│         │                   ▼                   │                        │
│         │           ┌──────────────┐            │                        │
│         │           │   Data       │            │                        │
│         └──────────▶│   Validator  │            │                        │
│                     │              │            │                        │
│                     └──────────────┘            │                        │
│                            │                    │                        │
│                            ▼                    │                        │
│                     ┌──────────────┐            │                        │
│                     │   Trigger    │◀───────────┘                        │
│                     │   System     │                                     │
│                     └──────────────┘                                     │
│                            │                                             │
│              ┌─────────────┴─────────────┐                              │
│              │                           │                               │
│              ▼                           ▼                               │
│       [Not Ready]                  [Ready]                              │
│              │                           │                               │
│              ▼                           ▼                               │
│         [Wait]                   ┌──────────────┐                       │
│                                  │   Trainer    │                       │
│                                  │  (Challenger)│                       │
│                                  └──────────────┘                       │
│                                         │                                │
│                                         ▼                                │
│                                  ┌──────────────┐                       │
│                                  │   Evaluator  │                       │
│                                  │              │                       │
│                                  └──────────────┘                       │
│                                         │                                │
│                                         ▼                                │
│  ┌──────────────┐               ┌──────────────┐                       │
│  │   Champion   │◀─────────────▶│  Comparator  │                       │
│  │   Model      │               │              │                       │
│  └──────────────┘               └──────────────┘                       │
│                                         │                                │
│                          ┌──────────────┴──────────────┐               │
│                          │                             │                │
│                          ▼                             ▼                │
│                   [Challenger Wins]            [Champion Wins]          │
│                          │                             │                │
│                          ▼                             ▼                │
│                   ┌──────────────┐             [Keep Current]           │
│                   │   Deployer   │             [Log Results]            │
│                   │              │                                      │
│                   └──────────────┘                                      │
│                          │                                              │
│                          ▼                                              │
│                   ┌──────────────┐                                      │
│                   │   Archive    │                                      │
│                   │   Manager    │                                      │
│                   └──────────────┘                                      │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

## Component Details

### 1. Data Ingestion (`src/data_ingestion.py`)

Bertanggung jawab untuk:
- Menerima data gempa baru dari berbagai sumber
- Menyimpan data ke pending queue
- Memindahkan data tervalidasi ke dataset

```python
class DataIngestion:
    def add_pending_event(event_data) -> dict
    def validate_pending_events() -> dict
    def get_pending_count() -> int
```

### 2. Data Validator (`src/data_validator.py`)

Memvalidasi data gempa baru:
- Format tanggal
- Kode stasiun valid
- Kelas magnitude valid
- Kelas azimuth valid
- Keberadaan spectrogram

```python
class DataValidator:
    def validate_event(event_data) -> dict
    def validate_spectrogram(path) -> bool
    def check_duplicate(event_data) -> bool
```

### 3. Trigger System

Menentukan kapan pipeline harus dijalankan:

| Trigger | Condition | Default |
|---------|-----------|---------|
| Min Events | Jumlah event tervalidasi ≥ threshold | 20 events |
| Max Days | Hari sejak training terakhir ≥ threshold | 90 days |
| Performance Drop | Akurasi turun ≥ threshold | 2% |

### 4. Model Trainer (`src/trainer.py`)

Melatih model challenger baru:
- Menggabungkan dataset lama + event baru
- Menggunakan arsitektur ConvNeXt-Tiny
- Data augmentation (MixUp, CutMix)
- Early stopping

```python
class ModelTrainer:
    def prepare_dataset(include_new_events) -> DataFrame
    def train_model() -> dict
```

### 5. Model Evaluator (`src/evaluator.py`)

Mengevaluasi model pada benchmark test set:
- Accuracy (Magnitude & Azimuth)
- F1-Score
- Precision & Recall
- MCC (Matthews Correlation Coefficient)
- Confidence Intervals

```python
class ModelEvaluator:
    def evaluate_champion() -> dict
    def evaluate_challenger() -> dict
    def compute_metrics(predictions, labels) -> dict
```

### 6. Model Comparator (`src/model_comparator.py`)

Membandingkan champion vs challenger:

```
Composite Score = Σ(weight × normalized_metric)

Weights:
- Magnitude Accuracy: 40%
- Azimuth Accuracy: 20%
- LOEO Validation: 30%
- False Positive Rate: 10%
```

Decision Rules:
1. Challenger harus memiliki composite score ≥ champion
2. Tidak boleh ada degradasi signifikan di metric manapun
3. Statistical significance test (optional)

### 7. Model Deployer (`src/deployer.py`)

Menangani deployment model baru:
- Backup champion lama ke archive
- Copy challenger ke champion directory
- Update model registry
- Cleanup old archives

```python
class ModelDeployer:
    def deploy_challenger(comparison_results) -> dict
    def rollback(version) -> dict
    def list_archived_models() -> list
```

## Data Flow

```
1. New Event → Pending Queue
2. Pending Queue → Validation → Validated Events
3. Validated Events (≥20) → Trigger Pipeline
4. Pipeline → Train Challenger
5. Challenger → Evaluate on Benchmark
6. Challenger vs Champion → Compare
7. If Challenger Wins → Deploy
8. Old Champion → Archive
```

## File Structure

```
autoupdate_pipeline/
├── config/
│   ├── pipeline_config.yaml    # Main configuration
│   └── model_registry.json     # Model tracking
├── src/
│   ├── data_ingestion.py       # Data input handling
│   ├── data_validator.py       # Data validation
│   ├── trainer.py              # Model training
│   ├── evaluator.py            # Model evaluation
│   ├── model_comparator.py     # Champion vs Challenger
│   ├── deployer.py             # Model deployment
│   └── utils.py                # Utility functions
├── scripts/
│   ├── run_pipeline.py         # Main runner
│   ├── add_new_event.py        # Add events
│   ├── check_status.py         # Status check
│   └── rollback_model.py       # Rollback
├── data/
│   ├── pending/                # Pending events
│   ├── validated/              # Validated events
│   └── benchmark/              # Fixed test set
├── models/
│   ├── champion/               # Current production
│   ├── challenger/             # Candidate model
│   └── archive/                # Old models
└── logs/
    └── pipeline_history.json   # Audit trail
```

## Security Considerations

1. **Model Integrity**: Checksum verification untuk model files
2. **Rollback**: Selalu backup sebelum deploy
3. **Audit Trail**: Log semua keputusan dan metrik
4. **Access Control**: Approval required untuk deployment (configurable)

## Scalability

Pipeline ini dirancang untuk:
- Batch processing (bukan real-time)
- Single model architecture (ConvNeXt)
- Moderate dataset size (< 10,000 samples)

Untuk scale yang lebih besar, pertimbangkan:
- Distributed training (multi-GPU)
- Model versioning dengan DVC
- Orchestration dengan Airflow/Kubeflow
