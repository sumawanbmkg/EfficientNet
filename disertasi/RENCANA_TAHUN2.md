# Rencana Tahun Kedua

**Periode**: 2026-2027

---

## 1. Ringkasan Rencana

| Tahap | Target | Timeline |
|-------|--------|----------|
| Pengumpulan Data Pendukung | Multi-parameter data | Bulan 1-3 |
| Pre-processing Data Pendukung | Clean & integrate | Bulan 3-5 |
| Integrasi Multi-Parameter | Feature fusion | Bulan 5-6 |
| Model Prediksi Parameter Gempa | Akurasi > 85% | Bulan 6-8 |
| Generator Data Sintetis | Augmentasi | Bulan 8-9 |
| Model Online Learning | Akurasi > 95% | Bulan 9-11 |
| Finalisasi & Publikasi | 2 papers | Bulan 11-12 |

---

## 2. Data Pendukung yang Akan Dikumpulkan

### 2.1 Kandidat Data Pendukung

| No | Jenis Data | Sumber | Parameter |
|----|------------|--------|-----------|
| 1 | Data Seismik | BMKG | Seismisitas, b-value |
| 2 | Data Ionosfer | LAPAN | TEC, foF2 |
| 3 | Data GPS/GNSS | BIG | Deformasi crustal |
| 4 | Data Radon | Lokal | Konsentrasi radon |
| 5 | Data Temperatur | BMKG | Anomali suhu |

### 2.2 Prioritas Data

**Prioritas Tinggi:**
1. **Data Ionosfer (TEC)** - Korelasi kuat dengan aktivitas seismik
2. **Data GPS** - Deformasi crustal sebagai indikator stress

**Prioritas Sedang:**
3. **Data Seismik** - Pola seismisitas sebelum gempa besar
4. **Data Radon** - Emisi gas dari rekahan

**Prioritas Rendah:**
5. **Data Temperatur** - Anomali termal

### 2.3 Sumber Data

| Data | Institusi | Akses |
|------|-----------|-------|
| Ionosfer TEC | LAPAN | Kerjasama |
| GPS/GNSS | BIG | Publik/Kerjasama |
| Seismik | BMKG | Internal |
| Radon | Universitas | Kolaborasi |

---

## 3. Metodologi Integrasi Multi-Parameter

### 3.1 Arsitektur Multi-Modal

```
┌─────────────────────────────────────────────────────────────┐
│                    Multi-Parameter Model                     │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐    │
│  │Geomag    │  │Ionosfer  │  │  GPS     │  │ Seismik  │    │
│  │Spectro   │  │  TEC     │  │Deformasi │  │ b-value  │    │
│  └────┬─────┘  └────┬─────┘  └────┬─────┘  └────┬─────┘    │
│       │             │             │             │           │
│       ▼             ▼             ▼             ▼           │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐    │
│  │Encoder 1 │  │Encoder 2 │  │Encoder 3 │  │Encoder 4 │    │
│  │(CNN)     │  │(LSTM)    │  │(MLP)     │  │(MLP)     │    │
│  └────┬─────┘  └────┬─────┘  └────┬─────┘  └────┬─────┘    │
│       │             │             │             │           │
│       └──────┬──────┴──────┬──────┴──────┬──────┘           │
│              │             │             │                   │
│              ▼             ▼             ▼                   │
│         ┌─────────────────────────────────┐                 │
│         │      Feature Fusion Layer       │                 │
│         │   (Attention / Concatenation)   │                 │
│         └───────────────┬─────────────────┘                 │
│                         │                                    │
│                         ▼                                    │
│         ┌─────────────────────────────────┐                 │
│         │      Prediction Heads           │                 │
│         ├─────────┬─────────┬─────────────┤                 │
│         │Magnitude│ Location│  Time Window│                 │
│         └─────────┴─────────┴─────────────┘                 │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

### 3.2 Teknik Fusion

| Teknik | Deskripsi | Kelebihan |
|--------|-----------|-----------|
| Early Fusion | Gabung fitur di awal | Simple |
| Late Fusion | Gabung prediksi di akhir | Modular |
| Attention Fusion | Weighted combination | Adaptive |
| Cross-Modal Attention | Inter-modal learning | Best performance |

### 3.3 Rencana Implementasi

1. **Fase 1**: Implementasi encoder per modalitas
2. **Fase 2**: Eksperimen berbagai teknik fusion
3. **Fase 3**: Optimasi dengan attention mechanism
4. **Fase 4**: Fine-tuning dan evaluasi

---

## 4. Model Prediksi Parameter Gempa

### 4.1 Parameter yang Diprediksi

| Parameter | Tipe | Target Akurasi |
|-----------|------|----------------|
| Magnitude | Regression/Classification | MAE < 0.5 |
| Lokasi (Azimuth) | Classification (8 kelas) | > 85% |
| Jarak | Regression | MAE < 50 km |
| Time Window | Classification | > 80% |

### 4.2 Definisi Time Window

| Kelas | Rentang |
|-------|---------|
| Imminent | 1-3 hari |
| Short-term | 4-7 hari |
| Medium-term | 8-14 hari |
| Long-term | 15-30 hari |

### 4.3 Loss Function

```python
# Multi-task loss dengan weighted combination
total_loss = (
    w1 * magnitude_loss +      # MSE atau Focal Loss
    w2 * location_loss +       # Cross-Entropy
    w3 * distance_loss +       # MSE
    w4 * time_window_loss      # Cross-Entropy
)
```

---

## 5. Online Learning Implementation

### 5.1 Konsep Online Learning

```
┌─────────────────────────────────────────────────────────────┐
│                    Online Learning System                    │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  Real-time Data Stream                                       │
│       │                                                      │
│       ▼                                                      │
│  ┌──────────┐                                               │
│  │  Buffer  │ ← Accumulate new samples                      │
│  └────┬─────┘                                               │
│       │                                                      │
│       ▼ (When buffer full or trigger)                       │
│  ┌──────────────────────────────────────┐                   │
│  │        Incremental Update            │                   │
│  │  ┌─────────────────────────────────┐ │                   │
│  │  │ 1. Load current model weights   │ │                   │
│  │  │ 2. Fine-tune on new data        │ │                   │
│  │  │ 3. Apply EWC regularization     │ │                   │
│  │  │ 4. Validate on holdout set      │ │                   │
│  │  └─────────────────────────────────┘ │                   │
│  └────────────────┬─────────────────────┘                   │
│                   │                                          │
│                   ▼                                          │
│  ┌──────────────────────────────────────┐                   │
│  │         Validation Gate              │                   │
│  │  - Performance check                 │                   │
│  │  - Catastrophic forgetting check     │                   │
│  │  - Drift detection                   │                   │
│  └────────────────┬─────────────────────┘                   │
│                   │                                          │
│          ┌───────┴───────┐                                  │
│          ▼               ▼                                  │
│     [Pass]          [Fail]                                  │
│          │               │                                  │
│          ▼               ▼                                  │
│     Deploy          Rollback                                │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

### 5.2 Teknik Anti-Forgetting

| Teknik | Deskripsi |
|--------|-----------|
| EWC (Elastic Weight Consolidation) | Regularisasi untuk menjaga weight penting |
| Experience Replay | Simpan dan replay sampel lama |
| Progressive Networks | Tambah kapasitas untuk task baru |
| PackNet | Pruning dan re-use weight |

### 5.3 Implementasi EWC

```python
class EWCLoss:
    def __init__(self, model, dataset, lambda_ewc=1000):
        self.lambda_ewc = lambda_ewc
        self.fisher = self._compute_fisher(model, dataset)
        self.optimal_params = {n: p.clone() for n, p in model.named_parameters()}
    
    def _compute_fisher(self, model, dataset):
        """Compute Fisher Information Matrix"""
        fisher = {}
        for n, p in model.named_parameters():
            fisher[n] = torch.zeros_like(p)
        
        model.eval()
        for x, y in dataset:
            model.zero_grad()
            output = model(x)
            loss = F.cross_entropy(output, y)
            loss.backward()
            
            for n, p in model.named_parameters():
                fisher[n] += p.grad ** 2
        
        for n in fisher:
            fisher[n] /= len(dataset)
        
        return fisher
    
    def penalty(self, model):
        """EWC penalty term"""
        loss = 0
        for n, p in model.named_parameters():
            loss += (self.fisher[n] * (p - self.optimal_params[n]) ** 2).sum()
        return self.lambda_ewc * loss
```

---

## 6. Timeline Detail

### Bulan 1-3: Pengumpulan Data Pendukung

| Minggu | Kegiatan |
|--------|----------|
| 1-2 | Koordinasi dengan LAPAN untuk data ionosfer |
| 3-4 | Koordinasi dengan BIG untuk data GPS |
| 5-6 | Download dan preprocessing data ionosfer |
| 7-8 | Download dan preprocessing data GPS |
| 9-10 | Quality control dan alignment temporal |
| 11-12 | Dokumentasi dataset |

### Bulan 4-6: Integrasi Multi-Parameter

| Minggu | Kegiatan |
|--------|----------|
| 1-2 | Desain arsitektur multi-modal |
| 3-4 | Implementasi encoder per modalitas |
| 5-6 | Implementasi fusion layer |
| 7-8 | Training dan evaluasi awal |

### Bulan 7-9: Model Prediksi Parameter

| Minggu | Kegiatan |
|--------|----------|
| 1-2 | Implementasi prediction heads |
| 3-4 | Training multi-task model |
| 5-6 | Hyperparameter tuning |
| 7-8 | SMOTE untuk data pendukung |
| 9-10 | Evaluasi dan iterasi |

### Bulan 10-12: Online Learning & Finalisasi

| Minggu | Kegiatan |
|--------|----------|
| 1-2 | Implementasi EWC |
| 3-4 | Implementasi experience replay |
| 5-6 | Testing online learning |
| 7-8 | Finalisasi paper 2 |
| 9-10 | Finalisasi paper 3 |
| 11-12 | Dokumentasi dan wrap-up |

---

## 7. Target Publikasi Tahun 2

### Paper 2

| Item | Detail |
|------|--------|
| Judul | Multi-Parameter Integration for Earthquake Precursor Detection using Deep Learning |
| Target | Nature Scientific Reports (Q1) |
| Fokus | Integrasi data geomagnetik + ionosfer + GPS |
| Timeline | Bulan 8-10 |

### Paper 3

| Item | Detail |
|------|--------|
| Judul | Online Learning Framework for Continuous Earthquake Precursor Monitoring |
| Target | JGR Solid Earth (Q1) |
| Fokus | Online learning dan real-time adaptation |
| Timeline | Bulan 10-12 |

---

## 8. Risiko dan Mitigasi

| Risiko | Probabilitas | Dampak | Mitigasi |
|--------|--------------|--------|----------|
| Data pendukung tidak tersedia | Medium | High | Alternatif: fokus pada 2 parameter |
| Akurasi tidak tercapai | Low | High | Iterasi arsitektur, tambah data |
| Catastrophic forgetting | Medium | Medium | EWC + Experience Replay |
| Keterlambatan timeline | Medium | Medium | Buffer time di setiap fase |

---

## 9. Kebutuhan Resources

### 9.1 Hardware

| Item | Spesifikasi | Status |
|------|-------------|--------|
| GPU | NVIDIA A100 atau setara | Perlu akses |
| Storage | 2 TB untuk multi-parameter data | Perlu |
| RAM | 64 GB | Perlu upgrade |

### 9.2 Kolaborasi

| Institusi | Data | Status |
|-----------|------|--------|
| LAPAN | Ionosfer TEC | Perlu koordinasi |
| BIG | GPS/GNSS | Perlu koordinasi |
| BMKG | Seismik | Sudah ada akses |

---

## 10. Metrik Keberhasilan Tahun 2

| Target | Nilai Target | Metrik |
|--------|--------------|--------|
| Akurasi Model Multi-Parameter | > 85% | Overall accuracy |
| Akurasi Online Learning | > 95% | Setelah adaptation |
| Publikasi | 2 papers | Submitted/Accepted |
| Forgetting Rate | < 5% | Performance drop on old data |
