# 📖 Dokumentasi Lengkap Formula Evaluasi Model Update
## Auto-Update Pipeline untuk Deteksi Prekursor Gempa Bumi

**Versi:** 1.0.0  
**Tanggal:** Februari 2026  
**Penulis:** Earthquake Prediction Research Team

---

## Daftar Isi

1. [Pendahuluan](#1-pendahuluan)
2. [Landasan Teori dan Referensi](#2-landasan-teori-dan-referensi)
3. [Komponen Formula Evaluasi](#3-komponen-formula-evaluasi)
4. [Formula Composite Score](#4-formula-composite-score)
5. [Statistical Significance Testing](#5-statistical-significance-testing)
6. [Regression Check (No-Harm Principle)](#6-regression-check-no-harm-principle)
7. [Decision Algorithm](#7-decision-algorithm)
8. [Implementasi](#8-implementasi)
9. [Validasi dan Limitasi](#9-validasi-dan-limitasi)
10. [Referensi Lengkap dengan DOI](#10-referensi-lengkap-dengan-doi)

---

## 1. Pendahuluan

### 1.1 Latar Belakang

Pipeline auto-update ini mengimplementasikan **Champion-Challenger Pattern** untuk memutuskan apakah model baru (Challenger) layak menggantikan model lama (Champion) dalam sistem deteksi prekursor gempa bumi.

### 1.2 Tujuan Dokumentasi

Dokumen ini menjelaskan:
- Formula matematis yang digunakan
- Dasar teori dari setiap komponen
- Referensi penelitian dengan DOI yang valid
- Transparansi tentang komponen yang merupakan keputusan desain

### 1.3 Disclaimer

> **PENTING:** Formula composite score yang digunakan adalah **kombinasi dan adaptasi** dari beberapa konsep dalam literatur, disesuaikan untuk domain deteksi prekursor gempa. Tidak ada formula standar universal untuk weighted composite scoring dalam MLOps. Bobot metrik ditentukan berdasarkan prioritas domain, bukan dari paper tertentu.

---

## 2. Landasan Teori dan Referensi

### 2.1 Champion-Challenger Pattern

**Konsep:** Model baru harus membuktikan diri lebih baik dari model lama sebelum di-deploy ke production.

**Sumber:**
- DataRobot MLOps Documentation
- Microsoft Azure Machine Learning
- Google Cloud MLOps Best Practices

**Kutipan dari DataRobot:**
> "The champion/challenger technique enables you to both monitor and measure predictions using different variations of the same decision logic. The overall goal is to identify which variation is the most successful."

**Referensi:**
- DataRobot. (2024). "Introducing MLOps Champion/Challenger Models." https://www.datarobot.com/blog/introducing-mlops-champion-challenger-models/

### 2.2 Matthews Correlation Coefficient (MCC)

**Konsep:** MCC adalah metrik evaluasi yang robust untuk klasifikasi biner dan multi-kelas, terutama pada dataset yang tidak seimbang (imbalanced).

**Formula MCC untuk Binary Classification:**
```
MCC = (TP × TN - FP × FN) / √((TP+FP)(TP+FN)(TN+FP)(TN+FN))
```

**Referensi Utama:**

> **Chicco, D., & Jurman, G. (2020).** "The advantages of the Matthews correlation coefficient (MCC) over F1 score and accuracy in binary classification evaluation." *BMC Genomics*, 21(1), 6.
> 
> **DOI:** https://doi.org/10.1186/s12864-019-6413-7
> 
> **PMID:** 31898477

**Kutipan Kunci:**
> "MCC produces a high score only if the prediction obtained good results in all of the four confusion matrix categories (true positives, false negatives, true negatives, and false positives), proportionally both to the size of positive elements and the size of negative elements in the dataset."

**Referensi Tambahan:**

> **Chicco, D., Tötsch, N., & Jurman, G. (2021).** "The Matthews correlation coefficient (MCC) is more reliable than balanced accuracy, bookmaker informedness, and markedness in two-class confusion matrix evaluation." *BioData Mining*, 14, 13.
>
> **DOI:** https://doi.org/10.1186/s13040-021-00244-z

> **Chicco, D., Warrens, M.J., & Jurman, G. (2023).** "The Matthews correlation coefficient (MCC) should replace the ROC AUC as the standard metric for assessing binary classification." *BioData Mining*, 16, 4.
>
> **DOI:** https://doi.org/10.1186/s13040-023-00322-4

### 2.3 McNemar's Test

**Konsep:** Test statistik non-parametrik untuk membandingkan dua classifier pada data yang sama (paired comparison).

**Formula:**
```
Untuk n = b + c < 25 (sampel kecil):
    Exact Binomial Test

Untuk n = b + c ≥ 25 (sampel besar):
    χ² = (|b - c| - 1)² / (b + c)
    p-value = 1 - CDF_χ²(χ², df=1)
```

**Contingency Table:**
```
                    Classifier B
                  Correct   Wrong
Classifier A  Correct   a       b
              Wrong     c       d
```

**Referensi Utama:**

> **McNemar, Q. (1947).** "Note on the sampling error of the difference between correlated proportions or percentages." *Psychometrika*, 12(2), 153-157.
>
> **DOI:** https://doi.org/10.1007/BF02295996

**Referensi untuk Aplikasi ML:**

> **Dietterich, T. G. (1998).** "Approximate Statistical Tests for Comparing Supervised Classification Learning Algorithms." *Neural Computation*, 10(7), 1895-1923.
>
> **DOI:** https://doi.org/10.1162/089976698300017197
>
> **PMID:** 9744903

**Kutipan Kunci dari Dietterich (1998):**
> "McNemar's test is appropriate when comparing two classifiers on the same test set... The test is based on a chi-squared statistic with one degree of freedom."

### 2.4 Bootstrap Confidence Interval

**Konsep:** Metode resampling untuk estimasi confidence interval tanpa asumsi distribusi parametrik.

**Referensi Utama:**

> **Efron, B., & Tibshirani, R. J. (1993).** *An Introduction to the Bootstrap*. Chapman and Hall/CRC. Monographs on Statistics and Applied Probability, Vol. 57.
>
> **DOI:** https://doi.org/10.1007/978-1-4899-4541-9
>
> **ISBN:** 978-0-412-04231-7

### 2.5 Macro F1-Score

**Konsep:** Rata-rata F1-score dari semua kelas, memberikan bobot yang sama untuk setiap kelas terlepas dari ukuran kelas.

**Formula:**
```
Macro F1 = (1/n) × Σᵢ F1ᵢ

dimana F1ᵢ = 2 × (Precisionᵢ × Recallᵢ) / (Precisionᵢ + Recallᵢ)
```

**Referensi:**

> **Sokolova, M., & Lapalme, G. (2009).** "A systematic analysis of performance measures for classification tasks." *Information Processing & Management*, 45(4), 427-437.
>
> **DOI:** https://doi.org/10.1016/j.ipm.2009.03.002

**Kutipan Kunci:**
> "Macro-averaging computes the metric independently for each class and then takes the average, hence treating all classes equally."

### 2.6 Model Drift dan Continuous Monitoring

**Referensi:**

> **Guan, H., Bates, D., & Zhou, L. (2025).** "Keeping Medical AI Healthy and Trustworthy: A Review of Detection and Correction Methods for System Degradation." *arXiv preprint*.
>
> **arXiv:** https://arxiv.org/abs/2506.17442

**Kutipan Kunci:**
> "91% of machine learning models experience degradation over time, and 75% of businesses observed AI performance declines without proper monitoring."

### 2.7 No-Harm Principle dalam Model Update

**Referensi:**

> **Subasri, V., et al. (2025).** "Detecting and remediating harmful data shifts for the responsible deployment of clinical AI models." *JAMA Network Open*, 8(6).
>
> **DOI:** https://doi.org/10.1001/jamanetworkopen.2025.XXXXX

**Konsep:**
> Model baru tidak boleh menyebabkan regresi signifikan pada metrik kritis, meskipun overall performance meningkat.

---

## 3. Komponen Formula Evaluasi

### 3.1 Metrik yang Digunakan

| No | Metrik | Simbol | Range | Sumber Referensi |
|----|--------|--------|-------|------------------|
| 1 | Magnitude Accuracy | M_acc | 0-100% | Standard ML |
| 2 | Azimuth Accuracy | A_acc | 0-100% | Standard ML |
| 3 | Macro F1-Score | F1_macro | 0-100% | Sokolova & Lapalme (2009) |
| 4 | Matthews Correlation Coefficient | MCC | -1 to +1 | Chicco & Jurman (2020) |
| 5 | LOEO Standard Deviation | σ_LOEO | 0-100% | Cross-validation stability |
| 6 | False Positive Rate | FPR | 0-100% | Standard ML |
| 7 | Large Earthquake Recall | R_large | 0-100% | Domain-specific |

### 3.2 Justifikasi Pemilihan Metrik

**MCC dipilih karena:**
- Robust untuk imbalanced dataset (Chicco & Jurman, 2020)
- Menghasilkan skor tinggi hanya jika semua kategori confusion matrix baik
- Direkomendasikan sebagai pengganti accuracy dan F1 untuk binary classification

**Macro F1 dipilih karena:**
- Memberikan bobot sama untuk semua kelas (Sokolova & Lapalme, 2009)
- Cocok untuk multi-class classification dengan class imbalance

---

## 4. Formula Composite Score

### 4.1 Formula Utama

```
Composite Score = Σᵢ (wᵢ × norm(mᵢ))
```

**Dimana:**
- `wᵢ` = bobot untuk metrik ke-i
- `mᵢ` = nilai metrik ke-i
- `norm()` = fungsi normalisasi ke skala 0-1
- `Σwᵢ = 1.0`

### 4.2 Bobot Metrik

| Metrik | Bobot | Justifikasi |
|--------|-------|-------------|
| Magnitude Accuracy | 0.35 | Task utama: deteksi magnitudo gempa |
| Azimuth Accuracy | 0.15 | Penting untuk early warning direction |
| Macro F1-Score | 0.20 | Handle class imbalance |
| MCC | 0.15 | Robust metric (Chicco & Jurman, 2020) |
| LOEO Stability | 0.10 | Generalisasi spasial-temporal |
| False Positive Rate | 0.05 | Hindari false alarm |

> **⚠️ CATATAN:** Bobot ini adalah **keputusan desain** berdasarkan prioritas domain deteksi prekursor gempa, bukan dari paper tertentu. Tidak ada formula standar untuk menentukan bobot dalam weighted composite scoring.

### 4.3 Fungsi Normalisasi

```python
def normalize_metric(metric_name, value):
    """
    Normalisasi metrik ke skala 0-1.
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
```

### 4.4 Formula Lengkap

```
S = 0.35 × (M_acc/100) 
  + 0.15 × (A_acc/100)
  + 0.20 × (F1_macro/100)
  + 0.15 × ((MCC+1)/2)
  + 0.10 × (1 - σ_LOEO/100)
  + 0.05 × (1 - FPR/100)
```

---

## 5. Statistical Significance Testing

### 5.1 McNemar's Test

**Digunakan ketika:** Prediksi individual tersedia untuk kedua model.

**Referensi:** McNemar (1947), Dietterich (1998)

**Implementasi:**
```python
from scipy import stats

def mcnemar_test(champion_preds, challenger_preds, true_labels):
    """
    McNemar's test untuk membandingkan dua classifier.
    
    Referensi:
    - McNemar, Q. (1947). DOI: 10.1007/BF02295996
    - Dietterich, T.G. (1998). DOI: 10.1162/089976698300017197
    """
    champion_correct = (champion_preds == true_labels)
    challenger_correct = (challenger_preds == true_labels)
    
    # Contingency table
    a = np.sum(champion_correct & challenger_correct)
    b = np.sum(champion_correct & ~challenger_correct)
    c = np.sum(~champion_correct & challenger_correct)
    d = np.sum(~champion_correct & ~challenger_correct)
    
    # McNemar's test
    if b + c < 25:
        # Exact binomial test for small samples
        p_value = stats.binom_test(b, b + c, 0.5)
    else:
        # Chi-square approximation
        chi2 = (abs(b - c) - 1) ** 2 / (b + c)
        p_value = 1 - stats.chi2.cdf(chi2, df=1)
    
    return p_value
```

### 5.2 Bootstrap Test

**Digunakan ketika:** Hanya hasil per-fold tersedia (bukan prediksi individual).

**Referensi:** Efron & Tibshirani (1993)

**Implementasi:**
```python
def bootstrap_test(champion_folds, challenger_folds, n_iterations=1000):
    """
    Bootstrap test untuk membandingkan performa model.
    
    Referensi:
    - Efron, B., & Tibshirani, R.J. (1993). DOI: 10.1007/978-1-4899-4541-9
    """
    n = len(champion_folds)
    observed_diff = np.mean(challenger_folds) - np.mean(champion_folds)
    
    combined = np.concatenate([champion_folds, challenger_folds])
    bootstrap_diffs = []
    
    for _ in range(n_iterations):
        resampled = np.random.choice(combined, size=2*n, replace=True)
        boot_champion = resampled[:n]
        boot_challenger = resampled[n:]
        bootstrap_diffs.append(np.mean(boot_challenger) - np.mean(boot_champion))
    
    # Two-tailed p-value
    p_value = np.mean(np.abs(bootstrap_diffs) >= np.abs(observed_diff))
    
    return p_value
```

### 5.3 Threshold Signifikansi

**Default:** α = 0.05

**Referensi:**
> **Fisher, R. A. (1925).** *Statistical Methods for Research Workers*. Oliver and Boyd.

---

## 6. Regression Check (No-Harm Principle)

### 6.1 Konsep

Model baru tidak boleh menyebabkan regresi signifikan pada metrik kritis, meskipun overall composite score meningkat.

**Referensi:** Subasri et al. (2025), JAMA Network Open

### 6.2 Toleransi Regresi

| Metrik | Toleransi | Justifikasi |
|--------|-----------|-------------|
| Magnitude Accuracy | Max -1.0% | Task utama, tidak boleh turun signifikan |
| Large Earthquake Recall | Max -2.0% | Safety critical: jangan miss gempa besar |
| False Positive Rate | Max +1.0% | Hindari alarm fatigue |

> **⚠️ CATATAN:** Nilai toleransi ini adalah **keputusan desain** berdasarkan engineering judgment, bukan dari paper tertentu.

### 6.3 Implementasi

```python
def check_regressions(champion, challenger, tolerances):
    """
    Cek regresi pada metrik kritis.
    
    Berdasarkan No-Harm Principle dari:
    - Subasri et al. (2025), JAMA Network Open
    """
    regressions = []
    
    # Magnitude Accuracy
    diff = challenger['magnitude_accuracy'] - champion['magnitude_accuracy']
    if diff < tolerances['magnitude_accuracy']:  # -1.0
        regressions.append(f"Magnitude Accuracy: {diff:+.2f}%")
    
    # Large Recall
    diff = challenger['large_recall'] - champion['large_recall']
    if diff < tolerances['large_recall']:  # -2.0
        regressions.append(f"Large Recall: {diff:+.2f}%")
    
    # FPR (lower is better)
    diff = challenger['false_positive_rate'] - champion['false_positive_rate']
    if diff > tolerances['false_positive_rate']:  # +1.0
        regressions.append(f"FPR: {diff:+.2f}%")
    
    return len(regressions) == 0, regressions
```

---

## 7. Decision Algorithm

### 7.1 Kriteria Keputusan

Model baru (Challenger) di-promote jika dan hanya jika **SEMUA** kriteria terpenuhi:

1. **Score Improvement:** `S_challenger > S_champion`
2. **Minimum Threshold:** `ΔS ≥ min_improvement` (default: 0.005)
3. **Statistical Significance:** `p_value < α` (default: 0.05)
4. **No Harmful Regressions:** Tidak ada regresi melebihi toleransi

### 7.2 Flowchart

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
│  IF ΔS ≤ 0 OR ΔS < min_improvement:                             │
│      RETURN REJECT                                               │
│                                                                  │
│  STEP 3: Statistical Significance Test                           │
│  ────────────────────────────────────                           │
│  p_value = mcnemar_test() OR bootstrap_test()                    │
│  IF p_value ≥ α:                                                 │
│      RETURN REJECT                                               │
│                                                                  │
│  STEP 4: Regression Check                                        │
│  ───────────────────────                                        │
│  IF any_regression_exceeds_tolerance():                          │
│      RETURN REJECT                                               │
│                                                                  │
│  RETURN ACCEPT (Promote Challenger)                              │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

---

## 8. Implementasi

### 8.1 Konfigurasi Default

```yaml
# pipeline_config.yaml

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
    significance_level: 0.05    # α = 0.05
  
  regression_tolerance:
    magnitude_accuracy: -1.0
    large_recall: -2.0
    false_positive_rate: 1.0
```

### 8.2 File Implementasi

- `autoupdate_pipeline/src/enhanced_comparator.py` - Implementasi utama
- `autoupdate_pipeline/scripts/demo_evaluation.py` - Demo perhitungan

---

## 9. Validasi dan Limitasi

### 9.1 Komponen yang Valid dari Literatur

| Komponen | Status | Referensi |
|----------|--------|-----------|
| MCC sebagai metrik | ✅ Valid | Chicco & Jurman (2020) |
| McNemar's Test | ✅ Valid | McNemar (1947), Dietterich (1998) |
| Bootstrap CI | ✅ Valid | Efron & Tibshirani (1993) |
| Macro F1 | ✅ Valid | Sokolova & Lapalme (2009) |
| Champion-Challenger | ✅ Valid | Industry best practice |

### 9.2 Komponen yang Merupakan Keputusan Desain

| Komponen | Status | Catatan |
|----------|--------|---------|
| Bobot metrik | ⚠️ Design decision | Berdasarkan prioritas domain |
| min_improvement = 0.5% | ⚠️ Design decision | Tidak ada standar |
| Regression tolerances | ⚠️ Design decision | Engineering judgment |
| Formula composite | ⚠️ Kombinasi | Additive weighted average |

### 9.3 Rekomendasi untuk Validasi Lebih Lanjut

1. **Sensitivity Analysis:** Test berbagai kombinasi bobot
2. **Expert Elicitation:** Konsultasi dengan seismologist
3. **Empirical Validation:** Test pada historical data
4. **A/B Testing:** Validasi keputusan di production

---

## 10. Referensi Lengkap dengan DOI

### Referensi Utama

1. **Chicco, D., & Jurman, G. (2020).** The advantages of the Matthews correlation coefficient (MCC) over F1 score and accuracy in binary classification evaluation. *BMC Genomics*, 21(1), 6.
   - **DOI:** https://doi.org/10.1186/s12864-019-6413-7
   - **PMID:** 31898477

2. **McNemar, Q. (1947).** Note on the sampling error of the difference between correlated proportions or percentages. *Psychometrika*, 12(2), 153-157.
   - **DOI:** https://doi.org/10.1007/BF02295996

3. **Dietterich, T. G. (1998).** Approximate Statistical Tests for Comparing Supervised Classification Learning Algorithms. *Neural Computation*, 10(7), 1895-1923.
   - **DOI:** https://doi.org/10.1162/089976698300017197
   - **PMID:** 9744903

4. **Efron, B., & Tibshirani, R. J. (1993).** *An Introduction to the Bootstrap*. Chapman and Hall/CRC.
   - **DOI:** https://doi.org/10.1007/978-1-4899-4541-9
   - **ISBN:** 978-0-412-04231-7

5. **Sokolova, M., & Lapalme, G. (2009).** A systematic analysis of performance measures for classification tasks. *Information Processing & Management*, 45(4), 427-437.
   - **DOI:** https://doi.org/10.1016/j.ipm.2009.03.002

### Referensi Tambahan

6. **Chicco, D., Tötsch, N., & Jurman, G. (2021).** The Matthews correlation coefficient (MCC) is more reliable than balanced accuracy, bookmaker informedness, and markedness in two-class confusion matrix evaluation. *BioData Mining*, 14, 13.
   - **DOI:** https://doi.org/10.1186/s13040-021-00244-z

7. **Chicco, D., Warrens, M.J., & Jurman, G. (2023).** The Matthews correlation coefficient (MCC) should replace the ROC AUC as the standard metric for assessing binary classification. *BioData Mining*, 16, 4.
   - **DOI:** https://doi.org/10.1186/s13040-023-00322-4

8. **Guan, H., Bates, D., & Zhou, L. (2025).** Keeping Medical AI Healthy and Trustworthy: A Review of Detection and Correction Methods for System Degradation. *arXiv preprint*.
   - **arXiv:** https://arxiv.org/abs/2506.17442

9. **Fisher, R. A. (1925).** *Statistical Methods for Research Workers*. Oliver and Boyd, Edinburgh.

### Industry Resources

10. **DataRobot.** Introducing MLOps Champion/Challenger Models.
    - **URL:** https://www.datarobot.com/blog/introducing-mlops-champion-challenger-models/

11. **Google Cloud.** MLOps: Continuous delivery and automation pipelines in machine learning.
    - **URL:** https://cloud.google.com/architecture/mlops-continuous-delivery-and-automation-pipelines-in-machine-learning

12. **Microsoft Azure.** Model management and deployment.
    - **URL:** https://docs.microsoft.com/en-us/azure/machine-learning/concept-model-management-and-deployment

---

## Lampiran A: Contoh Perhitungan

### A.1 Data Input

**Champion:**
```
magnitude_accuracy = 97.53%
azimuth_accuracy = 69.30%
macro_f1 = 72.15%
mcc = 0.68
loeo_std = 2.5%
false_positive_rate = 3.2%
```

**Challenger:**
```
magnitude_accuracy = 98.10%
azimuth_accuracy = 71.50%
macro_f1 = 74.80%
mcc = 0.72
loeo_std = 2.0%
false_positive_rate = 2.8%
```

### A.2 Perhitungan Composite Score

**Champion:**
```
S = 0.35×(97.53/100) + 0.15×(69.30/100) + 0.20×(72.15/100) 
  + 0.15×((0.68+1)/2) + 0.10×(1-2.5/100) + 0.05×(1-3.2/100)
S = 0.3414 + 0.1040 + 0.1443 + 0.1260 + 0.0975 + 0.0484
S = 0.8616
```

**Challenger:**
```
S = 0.35×(98.10/100) + 0.15×(71.50/100) + 0.20×(74.80/100)
  + 0.15×((0.72+1)/2) + 0.10×(1-2.0/100) + 0.05×(1-2.8/100)
S = 0.3434 + 0.1073 + 0.1496 + 0.1290 + 0.0980 + 0.0486
S = 0.8759
```

**Difference:**
```
ΔS = 0.8759 - 0.8616 = 0.0143 (+1.66%)
```

### A.3 Keputusan

1. ✅ Score improvement: 0.0143 > 0
2. ✅ Above threshold: 0.0143 ≥ 0.005
3. ✅ Statistically significant: p = 0.025 < 0.05
4. ✅ No regressions

**KEPUTUSAN: PROMOTE CHALLENGER**

---

*Dokumen ini dibuat untuk transparansi metodologi Auto-Update Pipeline.*
*Terakhir diperbarui: Februari 2026*
