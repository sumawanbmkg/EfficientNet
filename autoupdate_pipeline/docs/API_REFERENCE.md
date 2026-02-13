# 📚 API Reference

## Module Overview

| Module | Description |
|--------|-------------|
| `data_validator` | Validasi data event gempa |
| `data_ingestion` | Ingestion dan manajemen data |
| `trainer` | Training model challenger |
| `evaluator` | Evaluasi model pada benchmark |
| `model_comparator` | Perbandingan champion vs challenger |
| `deployer` | Deployment dan rollback model |
| `utils` | Utility functions |

---

## DataValidator

```python
from src.data_validator import DataValidator

validator = DataValidator(config)
```

### Methods

#### `validate_event(event_data: dict) -> dict`

Validasi data event gempa.

**Parameters:**
- `event_data`: Dictionary dengan keys: date, station, magnitude, azimuth

**Returns:**
```python
{
    "is_valid": bool,
    "errors": list,
    "warnings": list
}
```

#### `validate_spectrogram(path: str) -> bool`

Validasi file spectrogram.

---

## DataIngestion

```python
from src.data_ingestion import DataIngestion

ingestion = DataIngestion(config)
```

### Methods

| Method | Description |
|--------|-------------|
| `add_pending_event(event_data)` | Tambah event ke pending queue |
| `validate_pending_events()` | Validasi semua pending → validated |
| `get_pending_events()` | Get list pending events |
| `get_pending_count()` | Get jumlah pending events |
| `move_to_validated(event_ids)` | Pindahkan events ke validated |
| `clear_pending()` | Hapus semua pending events |

---

## ModelTrainer

```python
from src.trainer import ModelTrainer

trainer = ModelTrainer(config)
```

### Methods

#### `train_model(include_new_events: bool = True) -> dict`

Train model challenger baru.

**Parameters:**
- `include_new_events`: Include validated events baru

**Returns:**
```python
{
    "success": bool,
    "model_id": str,
    "epochs_trained": int,
    "training_time_hours": float,
    "best_val_mag_acc": float,
    "test_results": dict,
    "history": list
}
```

#### `prepare_dataset(include_new_events: bool) -> DataFrame`

Prepare combined dataset.

---

## ModelEvaluator

```python
from src.evaluator import ModelEvaluator

evaluator = ModelEvaluator(config)
```

### Methods

#### `evaluate_champion() -> dict`

Evaluate current champion model.

**Returns:**
```python
{
    "success": bool,
    "results": {
        "magnitude_accuracy": float,
        "azimuth_accuracy": float,
        "magnitude_f1": float,
        "azimuth_f1": float
    }
}
```

#### `evaluate_challenger() -> dict`

Evaluate challenger model.

---

## ModelComparator

```python
from src.model_comparator import ModelComparator

comparator = ModelComparator(config)
```

### Methods

#### `compare() -> dict`

Compare champion vs challenger.

**Returns:**
```python
{
    "success": bool,
    "champion": {
        "results": dict,
        "composite_score": float
    },
    "challenger": {
        "results": dict,
        "composite_score": float
    },
    "decision": {
        "promote_challenger": bool,
        "reason": str,
        "improvement": float
    }
}
```

---

## ModelDeployer

```python
from src.deployer import ModelDeployer

deployer = ModelDeployer(config)
```

### Methods

#### `deploy_challenger(comparison_results: dict, force: bool = False) -> dict`

Deploy challenger sebagai champion baru.

**Parameters:**
- `comparison_results`: Results dari ModelComparator
- `force`: Skip approval requirement

**Returns:**
```python
{
    "success": bool,
    "message": str,
    "old_champion": dict,
    "new_champion": dict,
    "archived_to": str
}
```

#### `rollback(version: str = None) -> dict`

Rollback ke versi sebelumnya.

**Parameters:**
- `version`: Version to rollback to (latest if None)

**Returns:**
```python
{
    "success": bool,
    "message": str,
    "rolled_back_to": str
}
```

#### `list_archived_models() -> list`

List semua archived models.

#### `get_current_champion() -> dict`

Get info champion saat ini.

---

## Utility Functions

```python
from src.utils import *
```

### Functions

| Function | Description |
|----------|-------------|
| `load_config(path)` | Load YAML config |
| `load_registry()` | Load model registry |
| `save_registry(data)` | Save model registry |
| `generate_model_id(arch)` | Generate unique model ID |
| `archive_model(src, dst, id)` | Archive model to directory |
| `cleanup_old_archives(path, max)` | Remove old archives |
| `increment_version(ver)` | Increment version string |
| `log_pipeline_event(event, data)` | Log event to history |

---

## Configuration Schema

```yaml
pipeline:
  name: str
  version: str

triggers:
  min_new_events: int
  max_days_between_training: int
  auto_trigger_enabled: bool

training:
  architecture: str  # convnext, efficientnet
  batch_size: int
  learning_rate: float
  epochs: int

comparison:
  weights:
    magnitude_accuracy: float
    azimuth_accuracy: float
    loeo_validation: float
    false_positive_rate: float

deployment:
  auto_deploy: bool
  require_approval: bool
  backup_champion: bool
```

---

## Error Codes

| Code | Description |
|------|-------------|
| `E001` | Invalid event data |
| `E002` | Validation failed |
| `E003` | Training failed |
| `E004` | Evaluation failed |
| `E005` | Deployment failed |
| `E006` | Rollback failed |
