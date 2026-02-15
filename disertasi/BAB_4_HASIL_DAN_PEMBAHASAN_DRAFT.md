# Bab 4: Hasil dan Pembahasan (Draft Awal)

## 4.1 Pengaturan Eksperimen (Experimental Setup)

Studi ini dirancang untuk mengevaluasi efektivitas arsitektur hierarkis dalam memprediksi karakteristik prekursor gempa bumi geomagnetik. Eksperimen dilakukan dalam tiga tahap terpisah yang meniru proses kognitif "deteksi - estimasi - lokalisasi".

### 4.1.1 Dataset dan Pra-pemrosesan
Dataset terdiri dari **2.340 sampel sinyal** direkam dari 24 stasiun pengamatan BMKG selama periode 2018-2025. Data mentah (time-series) dikonversi menjadi citra spektrogram untuk analisis visual mendalam.
*   **Total Sampel:** 2.340 (800 Normal, 1.540 Prekursor).
*   **Split Data:** 80% Training (1.872), 10% Validasi (234), 10% Testing (234).
*   **Augmentasi:** Random cropping, horizontal flip, dan time warping diterapkan pada spektrogram untuk meningkatkan generalisasi model.

### 4.1.2 Konfigurasi Perangkat Keras dan Lunak
Seluruh eksperimen dijalankan menggunakan infrastruktur komputasi kinerja tinggi (HPC) dengan spesifikasi:
*   **GPU:** NVIDIA A100 (40GB VRAM) / RTX 3090 (24GB VRAM).
*   **Framework:** PyTorch 2.1 dengan pustaka `timm` (PyTorch Image Models).
*   **Model Backbone:** 
    1.  *EfficientNet-B0* (Baseline CNN)
    2.  *Vision Transformer (ViT-Small)* (Baseline Transformer)
### 4.1.3 Batasan dan Kondisi Eksperimental (Experimental Disclaimers)

Untuk memastikan transparansi ilmiah dan reprodusibilitas, penelitian ini menetapkan batasan-batasan operasional sebagai berikut:

**A. Pendekatan Hierarkis (Hierarchical Approach)**
Kami menyadari bahwa prediksi gempa bumi adalah masalah yang sangat kompleks. Oleh karena itu, kami memecahnya menjadi tiga sub-masalah independen yang diselesaikan secara berurutan:
1.  **Stage 1 (Deteksi):** Binary Classification (Apakah ada anomali?).
2.  **Stage 2 (Estimasi):** Multi-class Classification (Seberapa besar magnitudo?).
3.  **Stage 3 (Lokalisasi):** Multi-class Classification (Dari arah mana sumber gempa?).
*Disclaimer:* Model pada stage lanjut (2 & 3) hanya dilatih pada data yang **sudah terkonfirmasi sebagai prekursor** (True Positives dari Stage 1). Kesalahan pada Stage 1 akan merambat ke stage berikutnya (Error Propagation).

**B. Integritas Dataset (Data Constraints)**
*   **Sumber Data:** Data geomagnetik raw diperoleh dari jejaring stasiun BMKG.
*   **Katalog Gempa:** Menggunakan katalog terpadu 2018-2025 yang telah divalidasi dan dikurasi ulang (Total ~25.000 event).
*   **Labeling:** Pelabelan dilakukan secara semi-otomatis berdasarkan jendela waktu (time window) sebelum kejadian gempa signifikan (M>4.0) dalam radius efektif stasiun.
*   **Imbalance:** Dataset ini secara alami tidak seimbang (imbalanced), dengan jumlah sampel *Normal* jauh lebih sedikit dibanding *Prekursor* dalam eksperimen ini (karena fokus studi pada karakteristik prekursor). Teknik *Weighted Loss* digunakan untuk memitigasi bias ini.

**C. Strategi Validasi (Validation Protocol)**
*   **Split:** Menggunakan rasio 80:10:10 (Train:Val:Test) secara acak namun terstratifikasi (stratified) berdasarkan stasiun.
*   **Station Integrity:** Kami memastikan tidak ada kebocoran data (data leakage) antar set. Namun, karena keterbatasan jumlah stasiun, validasi *Leave-One-Station-Out (LOSO)* akan dibahas secara terpisah sebagai uji generalisasi ekstrem.

**D. Pemilihan Framework dan Model**
*   **Bahasa & Library:** Python 3.10 dipilih karena ekosistem Data Science yang matang. PyTorch (v2.1) dipilih karena fleksibilitas riset (dynamic computation graph) dibanding TensorFlow.
*   **Backbone:** *EfficientNet-B0* (Google) dipilih karena merupakan representasi *State-of-the-Art (SOTA)* untuk arsitektur berbasis Konvolusi yang sangat efisien untuk data geofisika.

## 4.2 Hasil Stage 1: Deteksi Anomali Biner (Noise vs. Prekursor)

Tujuan tahap ini adalah memisahkan sinyal geomagnetik normal (latar belakang) dari sinyal yang mengandung anomali prekursor potensial.

| Model | Akurasi | Precision | Recall | F1-Score | Parameter |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **EfficientNet-B0 (Champion)** | **92.5%** | **89.3%** | **95.8%** | **92.4%** | 5.3M |
| Baseline CNN | 85.2% | 72.3% | 88.5% | 79.6% | 12.5M |

**Analisis:**
EfficientNet-B0 mengungguli model baseline CNN konvensional dalam tugas deteksi ini (+7.3% Akurasi). Hal ini mengonfirmasi hipotesis bahwa fitur lokal (tepi vertikal tajam, tekstur frekuensi tinggi) ciri khas prekursor lebih efektif ditangkap oleh arsitektur yang menggunakan *Compound Scaling* dan *MBConv blocks*.

## 4.3 Hasil Stage 2: Estimasi Magnitudo (Klasifikasi Multi-Kelas)

Pada tahap ini, hanya sampel yang terklasifikasi sebagai "Prekursor" yang diproses lebih lanjut untuk mengestimasi rentang magnitudo gempa terkait (<M3, M3-4, M4-5, >M5).

**Tabel 4.2: Perbandingan Recall Berdasarkan Kelas Magnitudo (M6.0+ Priority)**
| Model | Normal (Quiet) | Moderate (M4.5+) | Medium (M5+) | **Large (M6+)** |
| :--- | :---: | :---: | :---: | :---: |
| **EfficientNet-B0 (Exp 3)** | 86.2% | 62.5% | 78.9% | **100.0%** |
| EfficientNet-B0 (Champ) | **95.3%** | **68.7%** | **82.4%** | 98.65% |
| Baseline CNN | 92.1% | 45.2% | 58.3% | 65.3% |

**Analisis:**
EfficientNet-B0 menunjukkan keunggulan signifikan dalam mengestimasi magnitudo, terutama pada gempa besar (M6.0+). Mekanisme *Inverted Bottleneck* pada EfficientNet memungkinkan model untuk mengekstraksi fitur durasi dan energi sinyal secara mendalam, menghasilkan estimasi yang lebih akurat dibandingkan CNN standar.

## 4.4 Hasil Stage 3: Lokalisasi Arah (Azimuth Classification)

Evaluasi tahap ketiga berfokus pada kemampuan model untuk menentukan arah datangnya sinyal prekursor (lokalisasi sumber). 

**Tabel 4.3: Performa Lokalisasi Azimuth (8-Arah)**
| Arsitektur | Strategi Klasifikasi | Akurasi Validasi | Stabilitas (Epoch) |
| :--- | :--- | :---: | :--- |
| **EfficientNet-B0** | **8-Arah (Octant)** | **57.39%** | **Stabil** |
| Baseline CNN | 8-Arah (Octant) | 54.93% | Cukup Stabil |

**Analisis:**
Pencapaian akurasi **57.39%** pada tugas lokalisasi 8-arah menunjukkan bahwa pola spasial arah gempa terenkode dalam data geomagnetik. Model **EfficientNet-B0** menunjukkan performa superior dibandingkan baseline, yang dikaitkan dengan efisiensi parameter dalam menangkap tekstur frekuensi lokal yang berkorelasi dengan arah datangnya gelombang.

---

## 4.5 Temuan Kritis: Ketahanan terhadap Badai Matahari (Experiment 3)

Eksperimen 3 memberikan temuan kunci bagi implementasi riil di institusi seperti BMKG:
1.  **Zero Misclassification pada M6+**: Model EfficientNet-B0 berhasil mencapai **Recall 100%** untuk gempa magnitude besar, tanpa satupun kejadian yang terlewat meskipun diuji dengan data tahun 2024-2025 yang memiliki aktivitas fluks matahari sangat tinggi.
2.  **Solar Noise Rejection**: Kemampuan model untuk menolak gangguan matahari (*Solar Storm Rejection*) meningkat signifikan melalui strategi augmentasi data yang mencakup periode aktivitas matahari maksimum.
