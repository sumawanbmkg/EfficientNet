# 📐 Formula Evaluasi Detail untuk Keputusan Update Model

## Daftar Isi
1. [Pendahuluan](#1-pendahuluan)
2. [Komponen Evaluasi](#2-komponen-evaluasi)
3. [Formula Composite Score](#3-formula-composite-score)
4. [Statistical Significance Testing](#4-statistical-significance-testing)
5. [Regression Check (No-Harm Principle)](#5-regression-check-no-harm-principle)
6. [Decision Algorithm](#6-decision-algorithm)
7. [Contoh Perhitungan Lengkap](#7-contoh-perhitungan-lengkap)
8. [Konfigurasi Threshold](#8-konfigurasi-threshold)

---

## 1. Pendahuluan

### 1.1 Tujuan
Dokumen ini menjelaskan **formula matematis** yang digunakan untuk memutuskan apakah model baru (Challenger) layak menggantikan model lama (Champion).

### 1.2 Prinsip Dasar
```
┌─────────────────────────────────────────────────────────────────┐
│                    PRINSIP KEPUTUSAN                             │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  Model baru HANYA di-deploy jika memenuhi SEMUA kriteria:       │
│                                                                  │
│  ✓ Composite Score lebih tinggi                                 │
│  ✓ Peningkatan signifikan secara statistik                      │
│  ✓ Tidak ada regresi pada metrik kritis                         │
│                                                                  │
│  Jika SATU saja tidak terpenuhi → REJECT (Keep Champion)        │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

---

## 2. Komponen Evaluasi

### 2.1 Metrik yang Dievaluasi

| No | Metrik | Simbol | Range | Deskripsi |
|----|--------|--------|-------|-----------|
| 1 | Magnitude Accuracy | `M_acc` | 0-100% | Akurasi prediksi kelas magnitudo |
| 2 | Azimuth Accuracy | `A_acc` | 0-100% | Akurasi prediksi arah gempa |
| 3 | Macro F1-Score | `F1_macro` | 0-100% | Rata-rata F1 semua kelas |
| 4 | Matthews Correlation Coefficient | `MCC` | -1 to +1 | Korelasi prediksi vs aktual |
| 5 | LOEO Standard Deviation | `σ_LOEO` | 0-100% | Variabilitas antar fold (lower=better) |
| 6 | False Positive Rate | `FPR` | 0-100% | Tingkat false alarm (lower=better) |
| 7 | Large Earthquake Recall | `R_large` | 0-100% | Recall untuk gempa besar (M≥6.0) |

### 2.2 Sumber Data Metrik

```python
# Dari hasil evaluasi model pada benchmark test set
champion_metrics = {
    'magnitude_accuracy': 97.53,    # dari confusion matrix
    'azimuth_accuracy': 69.30,      # dari confusion matrix
    'macro_f1': 72.15,              # dihitung dari precision & recall per kelas
    'mcc': 0.68,                    # Matthews Correlation Coefficient
    'loeo_std': 2.5,                # std dev dari 10-fold LOEO
    'false_positive_rate': 3.2,     # FP / (FP + TN)
    'large_recall': 95.0            # TP_large / (TP_large + FN_large)
}
```

---

## 3. Formula Composite Score

### 3.1 Formula Utama

```
                    n
Composite Score = Σ (wᵢ × norm(mᵢ))
                   i=1

Dimana:
- wᵢ = bobot untuk metrik ke-i
- mᵢ = nilai metrik ke-i
- norm() = fungsi normalisasi ke skala 0-1
- Σwᵢ = 1.0 (total bobot = 100%)
```

### 3.2 Bobot Default (Earthquake Precursor Detection)

| Metrik | Bobot (wᵢ) | Justifikasi |
|--------|------------|-------------|
| `M_acc` | 0.35 (35%) | Task utama: deteksi magnitudo gempa |
| `A_acc` | 0.15 (15%) | Penting untuk early warning direction |
| `F1_macro` | 0.20 (20%) | Handle class imbalance |
| `MCC` | 0.15 (15%) | Robust untuk imbalanced data |
| `1 - σ_LOEO` | 0.10 (10%) | Stabilitas generalisasi |
| `1 - FPR` | 0.05 (5%) | Hindari false alarm |
| **Total** | **1.00** | |

### 3.3 Fungsi Normalisasi

```python
def normalize_metric(metric_name, value):
    """
    Normalisasi metrik ke skala 0-1.
    
    Untuk metrik "higher is better" (accuracy, F1, recall):
        norm(m) = m / 100
    
    Untuk metrik "lower is better" (FPR, std dev):
        norm(m) = 1 - (m / 100)
    
    Untuk MCC (range -1 to +1):
        norm(m) = (m + 1) / 2
    """
    
    if metric_name in ['magnitude_accuracy', 'azimuth_accuracy', 
                       'macro_f1', 'large_recall']:
        # Higher is better: 0-100% → 0-1
        return value / 100.0
    
    elif metric_name in ['false_positive_rate', 'loeo_std']:
        # Lower is better: invert
        return 1.0 - (value / 100.0)
    
    elif metric_name == 'mcc':
        # MCC range: -1 to +1 → 0 to 1
        return (value + 1.0) / 2.0
    
    else:
        return value / 100.0
```

### 3.4 Formula Lengkap

```
Composite Score = 0.35 × (M_acc/100) 
                + 0.15 × (A_acc/100)
                + 0.20 × (F1_macro/100)
                + 0.15 × ((MCC+1)/2)
                + 0.10 × (1 - σ_LOEO/100)
                + 0.05 × (1 - FPR/100)
```

---

## 4. Statistical Significance Testing

### 4.1 McNemar's Test

McNemar's test digunakan untuk membandingkan dua classifier pada **data yang sama**.

**Contingency Table:**
```
                        Challenger
                    Correct    Wrong
Champion  Correct     a          b
          Wrong       c          d

a = kedua model benar
b = Champion benar, Challenger salah
c = Champion salah, Challenger benar
d = kedua model salah
```

**Hipotesis:**
- H₀: b = c (tidak ada perbedaan signifikan)
- H₁: b ≠ c (ada perbedaan signifikan)

**Formula:**
```
Untuk n = b + c < 25 (sampel kecil):
    p-value = 2 × Σ(k=0 to min(b,c)) C(n,k) × 0.5^n
    (Exact Binomial Test)

Untuk n = b + c ≥ 25 (sampel besar):
    χ² = (|b - c| - 1)² / (b + c)
    p-value = 1 - CDF_χ²(χ², df=1)
    (Chi-Square Approximation)
```

**Interpretasi:**
- p-value < 0.05 → Perbedaan **signifikan**
- p-value ≥ 0.05 → Perbedaan **tidak signifikan**

### 4.2 Bootstrap Confidence Interval

Digunakan ketika prediksi individual tidak tersedia, hanya hasil per-fold.

**Algoritma:**
```python
def bootstrap_test(champion_fold_results, challenger_fold_results, 
                   n_iterations=1000):
    """
    Bootstrap test untuk membandingkan performa model.
    
    Input:
    - champion_fold_results: [acc_fold1, acc_fold2, ..., acc_fold10]
    - challenger_fold_results: [acc_fold1, acc_fold2, ..., acc_fold10]
    
    Output:
    - p_value: probabilitas perbedaan terjadi secara kebetulan
    - confidence_interval: [lower, upper] untuk perbedaan
    """
    
    n = len(champion_fold_results)
    observed_diff = mean(challenger_fold_results) - mean(champion_fold_results)
    
    # Gabungkan data untuk null hypothesis
    combined = champion_fold_results + challenger_fold_results
    
    bootstrap_diffs = []
    for i in range(n_iterations):
        # Resample dengan replacement
        resampled = random.choices(combined, k=2*n)
        boot_champion = resampled[:n]
        boot_challenger = resampled[n:]
        bootstrap_diffs.append(mean(boot_challenger) - mean(boot_champion))
    
    # Hitung p-value (two-tailed)
    p_value = sum(abs(d) >= abs(observed_diff) for d in bootstrap_diffs) / n_iterations
    
    # Confidence interval (95%)
    ci_lower = percentile(bootstrap_diffs, 2.5)
    ci_upper = percentile(bootstrap_diffs, 97.5)
    
    return p_value, (ci_lower, ci_upper)
```

---

## 5. Regression Check (No-Harm Principle)

### 5.1 Konsep

Berdasarkan penelitian [Subasri et al., 2025](https://jamanetwork.com/journals/jamanetworkopen/fullarticle/2831234):

> "Even if overall performance improves, degradation in critical metrics can cause harm."

### 5.2 Critical Metrics & Tolerances

| Metrik | Toleransi | Formula | Alasan |
|--------|-----------|---------|--------|
| Magnitude Accuracy | -1.0% | `M_acc_new ≥ M_acc_old - 1.0` | Task utama |
| Large Earthquake Recall | -2.0% | `R_large_new ≥ R_large_old - 2.0` | Safety critical |
| False Positive Rate | +1.0% | `FPR_new ≤ FPR_old + 1.0` | Avoid alarm fatigue |

### 5.3 Formula Regression Check

```python
def check_regressions(champion, challenger, tolerances):
    """
    Cek apakah ada regresi yang melebihi toleransi.
    
    Returns:
    - passed: True jika tidak ada regresi berbahaya
    - regressions: list metrik yang mengalami regresi
    """
    
    regressions = []
    
    # Magnitude Accuracy (higher is better)
    diff_mag = challenger['magnitude_accuracy'] - champion['magnitude_accuracy']
    if diff_mag < tolerances['magnitude_accuracy']:  # default: -1.0
        regressions.append(f"Magnitude Accuracy: {diff_mag:+.2f}%")
    
    # Large Earthquake Recall (higher is better)
    diff_recall = challenger['large_recall'] - champion['large_recall']
    if diff_recall < tolerances['large_recall']:  # default: -2.0
        regressions.append(f"Large Recall: {diff_recall:+.2f}%")
    
    # False Positive Rate (lower is better)
    diff_fpr = challenger['false_positive_rate'] - champion['false_positive_rate']
    if diff_fpr > tolerances['false_positive_rate']:  # default: +1.0
        regressions.append(f"FPR: {diff_fpr:+.2f}%")
    
    return len(regressions) == 0, regressions
```

---

## 6. Decision Algorithm

### 6.1 Flowchart Keputusan

```
┌─────────────────────────────────────────────────────────────────┐
│                    DECISION ALGORITHM                            │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  INPUT: champion_metrics, challenger_metrics                     │
│                                                                  │
│  STEP 1: Calculate Composite Scores                              │
│  ─────────────────────────────────────                          │
│  S_champion = Σ(wᵢ × norm(mᵢ_champion))                         │
│  S_challenger = Σ(wᵢ × norm(mᵢ_challenger))                     │
│  ΔS = S_challenger - S_champion                                  │
│                                                                  │
│  STEP 2: Check Score Improvement                                 │
│  ─────────────────────────────────                              │
│  IF ΔS ≤ 0:                                                      │
│      RETURN REJECT("Score not improved")                         │
│                                                                  │
│  IF ΔS < min_improvement (default: 0.005):                       │
│      RETURN REJECT("Improvement below threshold")                │
│                                                                  │
│  STEP 3: Statistical Significance Test                           │
│  ────────────────────────────────────                           │
│  p_value = mcnemar_test() OR bootstrap_test()                    │
│                                                                  │
│  IF p_value ≥ significance_level (default: 0.05):                │
│      RETURN REJECT("Not statistically significant")              │
│                                                                  │
│  STEP 4: Regression Check                                        │
│  ───────────────────────                                        │
│  passed, regressions = check_regressions()                       │
│                                                                  │
│  IF NOT passed:                                                  │
│      RETURN REJECT(f"Regressions: {regressions}")                │
│                                                                  │
│  STEP 5: (Optional) Strict Mode                                  │
│  ─────────────────────────────                                  │
│  IF strict_mode AND NOT all_metrics_improved():                  │
│      RETURN REJECT("Strict mode: not all metrics improved")      │
│                                                                  │
│  RETURN ACCEPT("Challenger wins!")                               │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### 6.2 Pseudocode

```python
def decide_model_update(champion, challenger, config):
    """
    Algoritma keputusan update model.
    
    Args:
        champion: dict dengan metrik model lama
        challenger: dict dengan metrik model baru
        config: konfigurasi threshold
    
    Returns:
        decision: {promote: bool, reason: str, confidence: float}
    """
    
    # === STEP 1: Calculate Composite Scores ===
    S_champion = calculate_composite_score(champion, config.weights)
    S_challenger = calculate_composite_score(challenger, config.weights)
    delta_S = S_challenger - S_champion
    
    # === STEP 2: Check Score Improvement ===
    if delta_S <= 0:
        return {
            'promote': False,
            'reason': f'Score not improved: {S_challenger:.4f} ≤ {S_champion:.4f}',
            'confidence': 0.0
        }
    
    if delta_S < config.min_improvement:  # default: 0.005
        return {
            'promote': False,
            'reason': f'Improvement {delta_S:.4f} < threshold {config.min_improvement}',
            'confidence': 0.0
        }
    
    # === STEP 3: Statistical Significance Test ===
    if config.has_predictions:
        stat_result = mcnemar_test(champion.predictions, 
                                   challenger.predictions, 
                                   true_labels)
    else:
        stat_result = bootstrap_test(champion.fold_results, 
                                     challenger.fold_results)
    
    if stat_result.p_value >= config.significance_level:  # default: 0.05
        return {
            'promote': False,
            'reason': f'Not significant: p={stat_result.p_value:.4f} ≥ {config.significance_level}',
            'confidence': 0.0
        }
    
    # === STEP 4: Regression Check ===
    passed, regressions = check_regressions(champion, challenger, config.tolerances)
    
    if not passed:
        return {
            'promote': False,
            'reason': f'Harmful regressions: {regressions}',
            'confidence': 0.0
        }
    
    # === STEP 5: Strict Mode (Optional) ===
    if config.strict_mode:
        all_improved = check_all_metrics_improved(champion, challenger)
        if not all_improved:
            return {
                'promote': False,
                'reason': 'Strict mode: not all metrics improved',
                'confidence': 0.0
            }
    
    # === ALL CHECKS PASSED ===
    return {
        'promote': True,
        'reason': f'Challenger wins: +{delta_S:.4f} (p={stat_result.p_value:.4f})',
        'confidence': 1.0 - stat_result.p_value
    }
```

---

## 7. Contoh Perhitungan Lengkap

### 7.1 Data Input

**Champion Model (Model Lama):**
```python
champion = {
    'magnitude_accuracy': 97.53,
    'azimuth_accuracy': 69.30,
    'macro_f1': 72.15,
    'mcc': 0.68,
    'loeo_std': 2.5,
    'false_positive_rate': 3.2,
    'large_recall': 95.0,
    'fold_results': [96.8, 97.2, 98.1, 97.5, 96.9, 97.8, 98.0, 97.3, 98.5, 96.2]
}
```

**Challenger Model (Model Baru dengan data tambahan):**
```python
challenger = {
    'magnitude_accuracy': 98.10,
    'azimuth_accuracy': 71.50,
    'macro_f1': 74.80,
    'mcc': 0.72,
    'loeo_std': 2.0,
    'false_positive_rate': 2.8,
    'large_recall': 96.5,
    'fold_results': [97.5, 98.0, 98.5, 98.2, 97.8, 98.3, 98.6, 98.0, 99.0, 97.1]
}
```

### 7.2 Step 1: Calculate Composite Scores

**Champion Score:**
```
S_champion = 0.35 × (97.53/100)           # Magnitude Accuracy
           + 0.15 × (69.30/100)           # Azimuth Accuracy
           + 0.20 × (72.15/100)           # Macro F1
           + 0.15 × ((0.68+1)/2)          # MCC
           + 0.10 × (1 - 2.5/100)         # LOEO Stability
           + 0.05 × (1 - 3.2/100)         # FPR (inverted)

S_champion = 0.35 × 0.9753
           + 0.15 × 0.6930
           + 0.20 × 0.7215
           + 0.15 × 0.8400
           + 0.10 × 0.9750
           + 0.05 × 0.9680

S_champion = 0.3414 + 0.1040 + 0.1443 + 0.1260 + 0.0975 + 0.0484
S_champion = 0.8616
```

**Challenger Score:**
```
S_challenger = 0.35 × (98.10/100)
             + 0.15 × (71.50/100)
             + 0.20 × (74.80/100)
             + 0.15 × ((0.72+1)/2)
             + 0.10 × (1 - 2.0/100)
             + 0.05 × (1 - 2.8/100)

S_challenger = 0.35 × 0.9810
             + 0.15 × 0.7150
             + 0.20 × 0.7480
             + 0.15 × 0.8600
             + 0.10 × 0.9800
             + 0.05 × 0.9720

S_challenger = 0.3434 + 0.1073 + 0.1496 + 0.1290 + 0.0980 + 0.0486
S_challenger = 0.8759
```

**Score Difference:**
```
ΔS = S_challenger - S_champion
ΔS = 0.8759 - 0.8616
ΔS = 0.0143 (atau +1.43%)
```

### 7.3 Step 2: Check Score Improvement

```
Kondisi 1: ΔS > 0?
→ 0.0143 > 0 ✓ PASS

Kondisi 2: ΔS ≥ min_improvement (0.005)?
→ 0.0143 ≥ 0.005 ✓ PASS
```

### 7.4 Step 3: Statistical Significance Test

**Bootstrap Test:**
```python
champion_folds = [96.8, 97.2, 98.1, 97.5, 96.9, 97.8, 98.0, 97.3, 98.5, 96.2]
challenger_folds = [97.5, 98.0, 98.5, 98.2, 97.8, 98.3, 98.6, 98.0, 99.0, 97.1]

mean_champion = 97.43
mean_challenger = 98.10
observed_diff = 98.10 - 97.43 = 0.67

# Setelah 1000 bootstrap iterations:
p_value = 0.023  # (contoh hasil)
```

```
Kondisi: p_value < significance_level (0.05)?
→ 0.023 < 0.05 ✓ PASS (Signifikan!)
```

### 7.5 Step 4: Regression Check

```
Magnitude Accuracy:
  diff = 98.10 - 97.53 = +0.57%
  tolerance = -1.0%
  +0.57% ≥ -1.0% ✓ PASS

Large Earthquake Recall:
  diff = 96.5 - 95.0 = +1.5%
  tolerance = -2.0%
  +1.5% ≥ -2.0% ✓ PASS

False Positive Rate:
  diff = 2.8 - 3.2 = -0.4%
  tolerance = +1.0%
  -0.4% ≤ +1.0% ✓ PASS

Regression Check: ✓ PASSED (No harmful regressions)
```

### 7.6 Final Decision

```
┌─────────────────────────────────────────────────────────────────┐
│                    HASIL EVALUASI                                │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  Champion Score:    0.8616                                       │
│  Challenger Score:  0.8759                                       │
│  Improvement:       +0.0143 (+1.66%)                            │
│                                                                  │
│  Statistical Test:  p = 0.023 (< 0.05) ✓                        │
│  Regression Check:  PASSED ✓                                     │
│                                                                  │
│  ══════════════════════════════════════════════════════════════ │
│  KEPUTUSAN: ✅ PROMOTE CHALLENGER                                │
│  ══════════════════════════════════════════════════════════════ │
│                                                                  │
│  Confidence: 97.7% (1 - p_value)                                │
│  Reason: Challenger wins with +1.43% composite score            │
│          improvement (p=0.023)                                   │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

---

## 8. Konfigurasi Threshold

### 8.1 Default Configuration (Testing)

```yaml
evaluation:
  weights:
    magnitude_accuracy: 0.35
    azimuth_accuracy: 0.15
    macro_f1: 0.20
    mcc: 0.15
    loeo_stability: 0.10
    false_positive_rate: 0.05
  
  decision:
    min_improvement: 0.005      # 0.5%
    significance_level: 0.05    # p < 0.05
    strict_mode: false
  
  regression_tolerance:
    magnitude_accuracy: -1.0    # Max 1% drop
    large_recall: -2.0          # Max 2% drop
    false_positive_rate: 1.0    # Max 1% increase
```

### 8.2 Conservative Configuration (Production)

```yaml
evaluation:
  weights:
    magnitude_accuracy: 0.40    # Lebih fokus pada task utama
    azimuth_accuracy: 0.15
    macro_f1: 0.15
    mcc: 0.15
    loeo_stability: 0.10
    false_positive_rate: 0.05
  
  decision:
    min_improvement: 0.01       # 1% (lebih ketat)
    significance_level: 0.01    # p < 0.01 (lebih ketat)
    strict_mode: true           # Semua metrik harus improve
  
  regression_tolerance:
    magnitude_accuracy: -0.5    # Max 0.5% drop (lebih ketat)
    large_recall: -1.0          # Max 1% drop
    false_positive_rate: 0.5    # Max 0.5% increase
```

### 8.3 Aggressive Configuration (Rapid Iteration)

```yaml
evaluation:
  weights:
    magnitude_accuracy: 0.30
    azimuth_accuracy: 0.20
    macro_f1: 0.20
    mcc: 0.15
    loeo_stability: 0.10
    false_positive_rate: 0.05
  
  decision:
    min_improvement: 0.001      # 0.1% (lebih longgar)
    significance_level: 0.10    # p < 0.10 (lebih longgar)
    strict_mode: false
  
  regression_tolerance:
    magnitude_accuracy: -2.0    # Max 2% drop
    large_recall: -3.0          # Max 3% drop
    false_positive_rate: 2.0    # Max 2% increase
```

---

## 9. Ringkasan Formula

### 9.1 Composite Score
```
S = 0.35×(M_acc/100) + 0.15×(A_acc/100) + 0.20×(F1/100) 
  + 0.15×((MCC+1)/2) + 0.10×(1-σ/100) + 0.05×(1-FPR/100)
```

### 9.2 Decision Criteria
```
PROMOTE jika dan hanya jika:
  (S_challenger - S_champion) ≥ min_improvement (0.005)
  AND p_value < significance_level (0.05)
  AND M_acc_diff ≥ -1.0%
  AND R_large_diff ≥ -2.0%
  AND FPR_diff ≤ +1.0%
```

### 9.3 Confidence Score
```
Confidence = 1 - p_value

Interpretasi:
- Confidence > 95%: Sangat yakin
- Confidence 90-95%: Yakin
- Confidence 80-90%: Cukup yakin
- Confidence < 80%: Kurang yakin
```

---

## 10. Referensi

1. **Guan, H., Bates, D., & Zhou, L. (2025)**. "Keeping Medical AI Healthy and Trustworthy." arXiv:2506.17442

2. **arXiv:2512.18390 (2025)**. "When Do New Data Sources Justify Switching ML Models?"

3. **Subasri, V. et al. (2025)**. "Detecting and remediating harmful data shifts." JAMA Network Open

4. **McNemar, Q. (1947)**. "Note on the sampling error of the difference between correlated proportions."

5. **Dietterich, T.G. (1998)**. "Approximate Statistical Tests for Comparing Supervised Classification Learning Algorithms."

6. **Chicco, D. & Jurman, G. (2020)**. "The advantages of the Matthews correlation coefficient (MCC)." BMC Genomics

---

*Dokumen ini dibuat untuk Auto-Update Pipeline v1.0*
*Terakhir diperbarui: Februari 2026*
