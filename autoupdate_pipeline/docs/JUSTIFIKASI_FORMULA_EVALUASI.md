# 📚 Justifikasi Formula Evaluasi: Transparansi dan Referensi

## ⚠️ Disclaimer Penting

**Formula yang digunakan dalam pipeline ini BUKAN formula standar yang diambil langsung dari satu paper tertentu.** Formula ini adalah **kombinasi dan adaptasi** dari beberapa konsep yang ditemukan dalam literatur MLOps dan Machine Learning, disesuaikan untuk kasus spesifik deteksi prekursor gempa bumi.

---

## 1. Komponen Formula dan Sumbernya

### 1.1 Weighted Composite Score

**Konsep:** Menggabungkan beberapa metrik dengan bobot berbeda.

**Status:** ⚠️ **TIDAK ADA formula standar universal**

**Sumber inspirasi:**
- **CUES Score** ([MDPI Mathematics, 2025](https://www.mdpi.com/2227-7390/14/3/398)): Menggunakan composite multiplicative score untuk clinical prediction models: `CUES = (C × U × E × S)^(1/4)`
- **Praktik industri**: DataRobot, Dataiku, dan platform MLOps lainnya menggunakan weighted scoring untuk champion-challenger comparison, tetapi **tidak mempublikasikan formula spesifik**.

**Keputusan desain:**
```
Composite Score = Σ(weight_i × normalized_metric_i)
```
- Formula additive (bukan multiplicative) dipilih karena lebih intuitif
- Bobot ditentukan berdasarkan **prioritas domain** (earthquake detection), bukan dari paper

### 1.2 Pemilihan Metrik

**Sumber yang valid:**

| Metrik | Referensi | Justifikasi |
|--------|-----------|-------------|
| **Accuracy** | Standard ML | Metrik dasar |
| **Macro F1-Score** | [Sokolova & Lapalme, 2009](https://doi.org/10.1016/j.ipm.2009.03.002) | Untuk multi-class imbalanced data |
| **MCC (Matthews)** | [Chicco & Jurman, 2020](https://doi.org/10.1186/s12864-020-6541-3) | "The advantages of MCC over F1 score and accuracy" - BMC Genomics |
| **False Positive Rate** | Standard ML | Untuk menghindari false alarm |

**Paper kunci untuk MCC:**
> Chicco, D., & Jurman, G. (2020). "The advantages of the Matthews correlation coefficient (MCC) over F1 score and accuracy in binary classification evaluation." BMC Genomics, 21(1), 6.

### 1.3 Statistical Significance Testing

**Sumber yang valid:**

| Test | Referensi | Status |
|------|-----------|--------|
| **McNemar's Test** | [Dietterich, 1998](https://doi.org/10.1162/089976698300017197) | ✅ **Well-established** |
| **Bootstrap CI** | [Efron & Tibshirani, 1993](https://doi.org/10.1007/978-1-4899-4541-9) | ✅ **Well-established** |

**Paper kunci:**
> Dietterich, T. G. (1998). "Approximate Statistical Tests for Comparing Supervised Classification Learning Algorithms." Neural Computation, 10(7), 1895-1923.

**Kutipan penting:**
> "McNemar's test is appropriate when comparing two classifiers on the same test set... The test is based on a chi-squared statistic with one degree of freedom."

### 1.4 Regression Check (No-Harm Principle)

**Sumber:**
- [Subasri et al., 2025](https://jamanetwork.com/journals/jamanetworkopen/fullarticle/2831234) - JAMA Network Open
- Konsep "no-harm" dari medical AI deployment

**Status:** ⚠️ **Konsep valid, threshold arbitrary**

Toleransi (-1%, -2%, +1%) adalah **keputusan desain**, bukan dari paper.

---

## 2. Apa yang VALID dari Referensi

### 2.1 Champion-Challenger Pattern ✅

**Sumber:** Industry best practice (DataRobot, Microsoft Azure, Google Cloud)

**Referensi:**
- [DataRobot Blog](https://www.datarobot.com/blog/introducing-mlops-champion-challenger-models/)
- [Microsoft Azure MLOps](https://docs.microsoft.com/en-us/azure/machine-learning/concept-model-management-and-deployment)
- [Google Cloud MLOps](https://cloud.google.com/architecture/mlops-continuous-delivery-and-automation-pipelines-in-machine-learning)

**Konsep yang valid:**
> "The champion/challenger technique enables you to both monitor and measure predictions using different variations of the same decision logic."

### 2.2 McNemar's Test untuk Classifier Comparison ✅

**Sumber:** [Machine Learning Mastery](https://machinelearningmastery.com/mcnemars-test-for-machine-learning/)

**Paper asli:** McNemar, Q. (1947). "Note on the sampling error of the difference between correlated proportions or percentages." Psychometrika, 12(2), 153-157.

### 2.3 MCC sebagai Metrik Robust ✅

**Sumber:** [Chicco & Jurman, 2020](https://doi.org/10.1186/s12864-020-6541-3)

**Kutipan:**
> "MCC produces a high score only if the prediction obtained good results in all of the four confusion matrix categories (true positives, false negatives, true negatives, and false positives)."

### 2.4 Model Drift Detection ✅

**Sumber:** [Guan et al., 2025](https://arxiv.org/abs/2506.17442) - arXiv

**Konsep yang valid:**
- 91% ML models experience degradation over time
- Continuous monitoring is essential
- Detection-Diagnosis-Correction (DDC) framework

---

## 3. Apa yang TIDAK VALID / Arbitrary

### 3.1 Bobot Metrik ⚠️

```yaml
weights:
  magnitude_accuracy: 0.35
  azimuth_accuracy: 0.15
  macro_f1: 0.20
  mcc: 0.15
  loeo_stability: 0.10
  false_positive_rate: 0.05
```

**Status:** ❌ **TIDAK dari paper**

**Justifikasi:**
- Bobot ditentukan berdasarkan **prioritas domain** (earthquake detection)
- Magnitude accuracy diberi bobot tertinggi karena task utama
- **Tidak ada formula universal untuk menentukan bobot**

**Alternatif yang lebih rigorous:**
1. **Expert elicitation** - Tanya domain expert
2. **Sensitivity analysis** - Test berbagai kombinasi bobot
3. **Multi-objective optimization** - Pareto frontier

### 3.2 Threshold Keputusan ⚠️

```yaml
decision:
  min_improvement: 0.005      # 0.5%
  significance_level: 0.05    # p < 0.05
```

**Status:**
- `significance_level: 0.05` → ✅ **Standard** (Fisher, 1925)
- `min_improvement: 0.005` → ❌ **Arbitrary**

### 3.3 Regression Tolerances ⚠️

```yaml
regression_tolerance:
  magnitude_accuracy: -1.0    # Max 1% drop
  large_recall: -2.0          # Max 2% drop
  false_positive_rate: 1.0    # Max 1% increase
```

**Status:** ❌ **TIDAK dari paper**

**Justifikasi:**
- Nilai dipilih berdasarkan "reasonable engineering judgment"
- Tidak ada standar industri untuk toleransi regresi

---

## 4. Rekomendasi untuk Validasi

### 4.1 Untuk Publikasi Ilmiah

Jika formula ini akan digunakan dalam paper, perlu:

1. **Sensitivity Analysis**: Test berbagai kombinasi bobot
2. **Expert Validation**: Konsultasi dengan domain expert (seismologist)
3. **Empirical Validation**: Test pada historical data
4. **Comparison**: Bandingkan dengan metode lain (single metric, Pareto)

### 4.2 Untuk Production Use

Formula saat ini **cukup untuk production** dengan catatan:

1. **Monitor hasil keputusan** - Track apakah model yang di-promote memang lebih baik
2. **Iterative refinement** - Sesuaikan bobot berdasarkan feedback
3. **Document assumptions** - Catat semua asumsi yang dibuat

---

## 5. Referensi Lengkap

### Paper yang Dikutip

1. **Chicco, D., & Jurman, G. (2020)**. "The advantages of the Matthews correlation coefficient (MCC) over F1 score and accuracy in binary classification evaluation." BMC Genomics, 21(1), 6. https://doi.org/10.1186/s12864-020-6541-3

2. **Dietterich, T. G. (1998)**. "Approximate Statistical Tests for Comparing Supervised Classification Learning Algorithms." Neural Computation, 10(7), 1895-1923. https://doi.org/10.1162/089976698300017197

3. **Guan, H., Bates, D., & Zhou, L. (2025)**. "Keeping Medical AI Healthy and Trustworthy: A Review of Detection and Correction Methods for System Degradation." arXiv:2506.17442

4. **Subasri, V. et al. (2025)**. "Detecting and remediating harmful data shifts for the responsible deployment of clinical AI models." JAMA Network Open.

5. **Sokolova, M., & Lapalme, G. (2009)**. "A systematic analysis of performance measures for classification tasks." Information Processing & Management, 45(4), 427-437.

6. **McNemar, Q. (1947)**. "Note on the sampling error of the difference between correlated proportions or percentages." Psychometrika, 12(2), 153-157.

7. **Efron, B., & Tibshirani, R. J. (1993)**. "An Introduction to the Bootstrap." Chapman and Hall/CRC.

### Industry Resources

8. **DataRobot**. "Introducing MLOps Champion/Challenger Models." https://www.datarobot.com/blog/introducing-mlops-champion-challenger-models/

9. **Google Cloud**. "MLOps: Continuous delivery and automation pipelines in machine learning." https://cloud.google.com/architecture/mlops-continuous-delivery-and-automation-pipelines-in-machine-learning

10. **Microsoft Azure**. "Model management and deployment." https://docs.microsoft.com/en-us/azure/machine-learning/concept-model-management-and-deployment

---

## 6. Kesimpulan

### Yang Solid (dari literatur):
- ✅ Champion-Challenger pattern
- ✅ McNemar's test untuk classifier comparison
- ✅ MCC sebagai metrik robust untuk imbalanced data
- ✅ Bootstrap confidence intervals
- ✅ Konsep model drift dan continuous monitoring

### Yang Perlu Validasi Lebih Lanjut:
- ⚠️ Bobot metrik (perlu expert elicitation atau sensitivity analysis)
- ⚠️ Threshold min_improvement (perlu empirical validation)
- ⚠️ Regression tolerances (perlu domain expert input)

### Rekomendasi:
1. **Untuk production**: Formula saat ini **acceptable** dengan monitoring
2. **Untuk publikasi**: Perlu validasi tambahan dan sensitivity analysis
3. **Untuk improvement**: Konsultasi dengan seismologist untuk bobot yang lebih tepat

---

*Dokumen ini dibuat untuk transparansi metodologi.*
*Terakhir diperbarui: Februari 2026*
