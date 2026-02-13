# 📊 Strategi Evaluation Metrics untuk Model Update Decision

## Ringkasan Eksekutif

Dokumen ini menjelaskan strategi **evaluation metrics** dan **kriteria keputusan** untuk menentukan apakah model baru (Challenger) layak menggantikan model lama (Champion). Strategi ini didasarkan pada penelitian terkini di bidang MLOps dan Medical AI Monitoring.

---

## 1. Latar Belakang: Mengapa Evaluation Metrics Penting?

### 1.1 Masalah Model Drift

Berdasarkan penelitian terbaru ([Guan et al., 2025](https://arxiv.org/html/2506.17442v2)):

> "91% of machine learning models experience degradation over time, and 75% of businesses observed AI performance declines without proper monitoring."

Model AI mengalami **performance degradation** karena:
- **Covariate Shift**: Distribusi input berubah (misal: data dari stasiun baru)
- **Label Shift**: Proporsi kelas berubah (misal: lebih banyak gempa besar)
- **Concept Shift**: Hubungan input-output berubah (misal: pola prekursor berbeda)

### 1.2 Champion-Challenger Pattern

Menurut [ModelOp](https://www.modelop.com/ai-governance/glossary/champion-challenger-testing):

> "The system tracks key metrics—accuracy, latency, fairness, and business impact—and applies statistical tests to see if any challenger delivers clear, sustained gains. If a challenger outperforms the champion, it replaces the champion in production."

**Prinsip utama:**
- Model baru HARUS membuktikan diri lebih baik sebelum di-deploy
- Perbandingan harus **fair** menggunakan benchmark test set yang sama
- Keputusan harus **data-driven**, bukan berdasarkan asumsi

---

## 2. Framework Evaluation Metrics

### 2.1 Multi-Dimensional Evaluation

Berdasarkan [arXiv:2512.18390](https://arxiv.org/abs/2512.18390) tentang "When Do New Data Sources Justify Switching Machine Learning Models?":

```
┌─────────────────────────────────────────────────────────────┐
│                 EVALUATION DIMENSIONS                        │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  1. DISCRIMINATION METRICS (Kemampuan Membedakan Kelas)     │
│     ├── Accuracy (Overall)                                   │
│     ├── Per-Class Accuracy                                   │
│     ├── F1-Score (Macro/Weighted)                           │
│     └── Matthews Correlation Coefficient (MCC)               │
│                                                              │
│  2. CALIBRATION METRICS (Keandalan Probabilitas)            │
│     ├── Expected Calibration Error (ECE)                    │
│     ├── Brier Score                                          │
│     └── Reliability Diagram                                  │
│                                                              │
│  3. ROBUSTNESS METRICS (Ketahanan terhadap Variasi)         │
│     ├── Cross-Validation Stability (Std Dev)                │
│     ├── Out-of-Distribution Performance                      │
│     └── Sensitivity Analysis                                 │
│                                                              │
│  4. OPERATIONAL METRICS (Kinerja Operasional)               │
│     ├── Inference Time                                       │
│     ├── False Positive Rate (FPR)                           │
│     └── False Negative Rate (FNR)                           │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

### 2.2 Metrics untuk Earthquake Precursor Detection

Untuk kasus spesifik deteksi prekursor gempa, metrics yang relevan:

| Metric | Deskripsi | Bobot | Alasan |
|--------|-----------|-------|--------|
| **Magnitude Accuracy** | Akurasi prediksi kelas magnitudo | 35% | Task utama: mendeteksi potensi gempa besar |
| **Azimuth Accuracy** | Akurasi prediksi arah gempa | 15% | Penting untuk early warning |
| **Macro F1-Score** | Rata-rata F1 semua kelas | 20% | Menangani class imbalance |
| **MCC (Matthews)** | Korelasi prediksi vs aktual | 15% | Robust untuk imbalanced data |
| **LOEO Stability** | Std dev akurasi antar fold | 10% | Generalisasi spasial-temporal |
| **False Positive Rate** | Tingkat false alarm | 5% | Menghindari alarm palsu |

---

## 3. Kriteria Keputusan Update Model

### 3.1 Decision Framework

Berdasarkan penelitian [Davis et al., 2019](https://academic.oup.com/jamia/article/26/12/1448/5566685) tentang model updating:

```
┌─────────────────────────────────────────────────────────────┐
│                    DECISION TREE                             │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  [Challenger Trained] ──► [Evaluate on Benchmark]           │
│                                   │                          │
│                                   ▼                          │
│                    ┌──────────────────────────┐              │
│                    │ Calculate Composite Score │              │
│                    └──────────────────────────┘              │
│                                   │                          │
│              ┌────────────────────┼────────────────────┐     │
│              │                    │                    │     │
│              ▼                    ▼                    ▼     │
│     [Score < Champion]   [Score ≈ Champion]   [Score > Champion]
│              │                    │                    │     │
│              ▼                    ▼                    ▼     │
│         [REJECT]          [Statistical Test]      [Check    │
│                                   │              Regressions]│
│                                   │                    │     │
│                           ┌───────┴───────┐           │     │
│                           │               │           │     │
│                           ▼               ▼           ▼     │
│                    [Not Significant] [Significant] [No Regression]
│                           │               │           │     │
│                           ▼               ▼           ▼     │
│                       [REJECT]        [ACCEPT]    [ACCEPT]  │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

### 3.2 Composite Score Calculation

**Formula:**

```python
Composite_Score = Σ(weight_i × normalized_metric_i)

# Normalisasi ke skala 0-1:
normalized_metric = (metric - min_expected) / (max_expected - min_expected)
```

**Contoh Perhitungan:**

| Metric | Champion | Challenger | Weight | Champion Score | Challenger Score |
|--------|----------|------------|--------|----------------|------------------|
| Mag Acc | 97.53% | 98.10% | 0.35 | 0.9753 × 0.35 = 0.341 | 0.9810 × 0.35 = 0.343 |
| Azi Acc | 69.30% | 71.50% | 0.15 | 0.6930 × 0.15 = 0.104 | 0.7150 × 0.15 = 0.107 |
| Macro F1 | 0.72 | 0.74 | 0.20 | 0.72 × 0.20 = 0.144 | 0.74 × 0.20 = 0.148 |
| MCC | 0.68 | 0.70 | 0.15 | 0.68 × 0.15 = 0.102 | 0.70 × 0.15 = 0.105 |
| LOEO Std | 2.5% | 2.0% | 0.10 | (1-0.025) × 0.10 = 0.098 | (1-0.020) × 0.10 = 0.098 |
| FPR | 3.0% | 2.5% | 0.05 | (1-0.030) × 0.05 = 0.049 | (1-0.025) × 0.05 = 0.049 |
| **TOTAL** | | | **1.00** | **0.838** | **0.850** |

**Keputusan:** Challenger Score (0.850) > Champion Score (0.838) → **Lanjut ke Statistical Test**

### 3.3 Statistical Significance Testing

Berdasarkan [Rauba et al., 2024](https://arxiv.org/abs/2401.14093) tentang self-healing ML:

**Metode yang direkomendasikan:**

1. **McNemar's Test** - Untuk membandingkan dua classifier pada data yang sama
2. **Paired t-test** - Untuk membandingkan performa pada multiple folds
3. **Bootstrap Confidence Interval** - Untuk estimasi uncertainty

```python
# McNemar's Test
from scipy.stats import mcnemar

# Contingency table:
#                    Challenger Correct  Challenger Wrong
# Champion Correct        a                   b
# Champion Wrong          c                   d

# H0: b = c (tidak ada perbedaan signifikan)
# H1: b ≠ c (ada perbedaan signifikan)

result = mcnemar([[a, b], [c, d]], exact=True)
p_value = result.pvalue

# Jika p_value < 0.05, perbedaan signifikan
```

### 3.4 Regression Check (No-Harm Principle)

Berdasarkan [Subasri et al., 2025](https://jamanetwork.com/journals/jamanetworkopen/fullarticle/2831234):

> "Detecting and remediating harmful data shifts for the responsible deployment of clinical AI models"

**Critical Metrics yang TIDAK BOLEH Turun:**

| Metric | Toleransi | Alasan |
|--------|-----------|--------|
| Magnitude Accuracy | -1.0% | Task utama, tidak boleh degradasi |
| Large Earthquake Recall | -2.0% | Keselamatan: jangan miss gempa besar |
| False Positive Rate | +1.0% | Menghindari alarm fatigue |

```python
def check_regressions(champion_metrics, challenger_metrics):
    """
    Check for harmful regressions in critical metrics.
    Returns: (passed: bool, regressions: list)
    """
    regressions = []
    
    # Magnitude accuracy: max 1% drop allowed
    mag_diff = challenger_metrics['magnitude_accuracy'] - champion_metrics['magnitude_accuracy']
    if mag_diff < -1.0:
        regressions.append(f"Magnitude Accuracy: {mag_diff:+.2f}%")
    
    # Large earthquake recall: max 2% drop allowed
    large_recall_diff = challenger_metrics['large_recall'] - champion_metrics['large_recall']
    if large_recall_diff < -2.0:
        regressions.append(f"Large Earthquake Recall: {large_recall_diff:+.2f}%")
    
    # False positive rate: max 1% increase allowed
    fpr_diff = challenger_metrics['false_positive_rate'] - champion_metrics['false_positive_rate']
    if fpr_diff > 1.0:
        regressions.append(f"False Positive Rate: {fpr_diff:+.2f}%")
    
    return len(regressions) == 0, regressions
```

---

## 4. Implementasi: Enhanced Model Comparator

### 4.1 Arsitektur Baru

```
┌─────────────────────────────────────────────────────────────┐
│                 ENHANCED MODEL COMPARATOR                    │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐      │
│  │  Champion   │    │  Challenger │    │  Benchmark  │      │
│  │   Model     │    │    Model    │    │  Test Set   │      │
│  └──────┬──────┘    └──────┬──────┘    └──────┬──────┘      │
│         │                  │                  │              │
│         └──────────────────┼──────────────────┘              │
│                            │                                 │
│                            ▼                                 │
│              ┌─────────────────────────┐                    │
│              │   Multi-Metric Evaluator │                    │
│              │   - Discrimination       │                    │
│              │   - Calibration          │                    │
│              │   - Robustness           │                    │
│              │   - Operational          │                    │
│              └───────────┬─────────────┘                    │
│                          │                                   │
│                          ▼                                   │
│              ┌─────────────────────────┐                    │
│              │  Composite Score Calc   │                    │
│              │  (Weighted Average)     │                    │
│              └───────────┬─────────────┘                    │
│                          │                                   │
│                          ▼                                   │
│              ┌─────────────────────────┐                    │
│              │  Statistical Testing    │                    │
│              │  - McNemar's Test       │                    │
│              │  - Bootstrap CI         │                    │
│              └───────────┬─────────────┘                    │
│                          │                                   │
│                          ▼                                   │
│              ┌─────────────────────────┐                    │
│              │   Regression Check      │                    │
│              │   (No-Harm Principle)   │                    │
│              └───────────┬─────────────┘                    │
│                          │                                   │
│                          ▼                                   │
│              ┌─────────────────────────┐                    │
│              │   DECISION ENGINE       │                    │
│              │   ACCEPT / REJECT       │                    │
│              └─────────────────────────┘                    │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

### 4.2 Decision Rules

```python
class DecisionEngine:
    """
    Decision engine based on research best practices.
    
    References:
    - arXiv:2512.18390 (Model Switching Decision)
    - arXiv:2506.17442 (Medical AI Monitoring)
    - ModelOp Champion-Challenger Pattern
    """
    
    def __init__(self, config):
        self.min_improvement = config.get('min_improvement', 0.005)  # 0.5%
        self.significance_level = config.get('significance_level', 0.05)
        self.strict_mode = config.get('strict_mode', False)
    
    def decide(self, champion_results, challenger_results, 
               statistical_test, regression_check):
        """
        Make final decision on model promotion.
        
        Decision Logic:
        1. Challenger composite score must be higher
        2. Improvement must be statistically significant (p < 0.05)
        3. No harmful regressions in critical metrics
        4. (Optional) Strict mode: ALL metrics must improve
        """
        
        decision = {
            'promote': False,
            'reason': '',
            'confidence': 0.0,
            'details': {}
        }
        
        score_diff = challenger_results['composite_score'] - champion_results['composite_score']
        
        # Rule 1: Score improvement
        if score_diff < self.min_improvement:
            decision['reason'] = f"Insufficient improvement: {score_diff:.4f} < {self.min_improvement}"
            return decision
        
        # Rule 2: Statistical significance
        if statistical_test['p_value'] >= self.significance_level:
            decision['reason'] = f"Not statistically significant: p={statistical_test['p_value']:.4f}"
            return decision
        
        # Rule 3: No regressions
        if not regression_check['passed']:
            decision['reason'] = f"Harmful regressions detected: {regression_check['regressions']}"
            return decision
        
        # Rule 4: Strict mode (optional)
        if self.strict_mode:
            if not all(challenger_results['improvements'].values()):
                decision['reason'] = "Strict mode: Not all metrics improved"
                return decision
        
        # All checks passed!
        decision['promote'] = True
        decision['reason'] = f"Challenger wins with {score_diff:.4f} improvement (p={statistical_test['p_value']:.4f})"
        decision['confidence'] = 1 - statistical_test['p_value']
        
        return decision
```

---

## 5. Konfigurasi yang Direkomendasikan

### 5.1 Default Configuration

```yaml
# autoupdate_pipeline/config/evaluation_config.yaml

evaluation:
  # Metric weights (must sum to 1.0)
  weights:
    magnitude_accuracy: 0.35
    azimuth_accuracy: 0.15
    macro_f1: 0.20
    mcc: 0.15
    loeo_stability: 0.10
    false_positive_rate: 0.05
  
  # Decision thresholds
  decision:
    min_improvement: 0.005        # 0.5% minimum composite score improvement
    significance_level: 0.05      # p-value threshold for statistical test
    strict_mode: false            # Require ALL metrics to improve
  
  # Regression tolerances (negative = allowed drop)
  regression_tolerance:
    magnitude_accuracy: -1.0      # Max 1% drop allowed
    large_recall: -2.0            # Max 2% drop allowed
    false_positive_rate: 1.0      # Max 1% increase allowed
  
  # Statistical testing
  statistical_test:
    method: "mcnemar"             # Options: mcnemar, paired_ttest, bootstrap
    bootstrap_iterations: 1000
    confidence_level: 0.95
```

### 5.2 Conservative Configuration (untuk Production)

```yaml
# Untuk deployment production yang lebih hati-hati

evaluation:
  weights:
    magnitude_accuracy: 0.40      # Lebih fokus pada task utama
    azimuth_accuracy: 0.15
    macro_f1: 0.15
    mcc: 0.15
    loeo_stability: 0.10
    false_positive_rate: 0.05
  
  decision:
    min_improvement: 0.01         # 1% minimum improvement
    significance_level: 0.01      # Lebih ketat: p < 0.01
    strict_mode: true             # Semua metrics harus improve
  
  regression_tolerance:
    magnitude_accuracy: -0.5      # Lebih ketat: max 0.5% drop
    large_recall: -1.0            # Max 1% drop
    false_positive_rate: 0.5      # Max 0.5% increase
```

---

## 6. Referensi Penelitian

### 6.1 MLOps & Model Monitoring

1. **Guan, H., Bates, D., & Zhou, L. (2025)**. "Keeping Medical AI Healthy and Trustworthy: A Review of Detection and Correction Methods for System Degradation." [arXiv:2506.17442](https://arxiv.org/abs/2506.17442)

2. **arXiv:2512.18390 (2025)**. "The Challenger: When Do New Data Sources Justify Switching Machine Learning Models?" [Link](https://arxiv.org/abs/2512.18390)

3. **Rauba, P. et al. (2024)**. "Self-Healing Machine Learning: A Framework for Autonomous Adaptation in Real-World Environments." NeurIPS 2024.

4. **DataRobot (2025)**. "Introducing MLOps Champion/Challenger Models." [Blog](https://www.datarobot.com/blog/introducing-mlops-champion-challenger-models/)

### 6.2 Model Updating & Calibration

5. **Davis, S.E. et al. (2019)**. "A nonparametric updating method to correct clinical prediction model drift." JAMIA.

6. **Subasri, V. et al. (2025)**. "Detecting and remediating harmful data shifts for the responsible deployment of clinical AI models." JAMA Network Open.

7. **Chi, S. et al. (2022)**. "A novel lifelong machine learning-based method to eliminate calibration drift in clinical prediction models." AI in Medicine.

### 6.3 Statistical Testing

8. **McNemar, Q. (1947)**. "Note on the sampling error of the difference between correlated proportions or percentages." Psychometrika.

9. **Dietterich, T.G. (1998)**. "Approximate Statistical Tests for Comparing Supervised Classification Learning Algorithms." Neural Computation.

---

## 7. Kesimpulan

Strategi evaluation metrics yang baik harus:

1. **Multi-dimensional**: Tidak hanya accuracy, tapi juga F1, MCC, calibration, dll.
2. **Weighted**: Bobot sesuai prioritas task (magnitude > azimuth)
3. **Statistically rigorous**: Gunakan statistical test untuk validasi
4. **Safe**: No-harm principle untuk critical metrics
5. **Configurable**: Dapat disesuaikan untuk testing vs production

Dengan framework ini, keputusan update model menjadi **data-driven**, **reproducible**, dan **safe**.

---

*Dokumen ini dibuat berdasarkan penelitian terkini di bidang MLOps dan Medical AI Monitoring.*
*Terakhir diperbarui: Februari 2026*
