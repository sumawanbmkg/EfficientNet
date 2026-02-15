# Ringkasan Penelitian Disertasi

## 1. Judul Penelitian

**Pengembangan Sistem Prediksi Gempa Bumi Komprehensif Berbasis Prekusor Geomagnetik Menggunakan Arsitektur EfficientNet-B0 Hierarkis**

---

## 2. Identitas Peneliti

| Item | Detail |
|------|--------|
| Nama | Sumawan, ST, MM |
| NRP | 7009232004 |
| Program | Doktor (S3) |
| Departemen | Teknik Fisika |
| Fakultas | Teknologi Industri dan Rekayasa Sistem |
| Institusi | Institut Teknologi Sepuluh Nopember (ITS) Surabaya |

### Dosen Pembimbing
1. **Dr. Bambang Lelono Widjiantoro, S.T., M.T.** - Pembimbing 1
2. **Prof. Dr. Katherin Indriawati, S.T, M.T.** - Pembimbing 2
3. **Dr. Muhamad Syirojudin, M.Si.** - Pembimbing 3

---

## 3. Latar Belakang

Indonesia terletak di kawasan Ring of Fire Pasifik yang merupakan zona tektonik paling aktif di dunia, berada di pertemuan tiga lempeng besar: Eurasia, Indo-Australia, dan Pasifik. Data BNPB mencatat:
- **781 kejadian** gempa bumi merusak
- **16.500 korban jiwa** meninggal
- **28 kejadian** tsunami dengan **5.064 korban** meninggal
- **10.789 gempa** tercatat selama 2023

Penelitian sebelumnya menunjukkan bahwa anomali geomagnetik pada rentang frekuensi Ultra Low Frequency (ULF), khususnya pulsasi PC3 (10-45 mHz), dapat muncul **7-11 hari** sebelum gempa bumi besar terjadi. Anomali ini ditandai dengan peningkatan rasio Z/H (komponen vertikal terhadap horizontal).

BMKG telah mengoperasikan sistem deteksi prekursor dengan metode rasio Z/H, namun akurasi masih **di bawah 70%**. Diperlukan pendekatan baru menggunakan deep learning untuk meningkatkan akurasi.

---

## 4. Rumusan Masalah

1. Bagaimana cara mendeteksi prekursor seismik pada rekaman medan geomagnetik secara konsisten?

2. Bagaimana cara mengatasi keterbatasan dataset yang diperlukan untuk melabeli prekursor seismik dengan akurat?

3. Bagaimana cara mencapai sistem prediksi gempa bumi jangka pendek yang komprehensif meliputi magnitudo, lokasi, dan waktu kejadian gempa secara efektif?

---

## 5. Tujuan Penelitian

### 5.1 Tujuan Umum
Mengembangkan sistem prediksi gempa bumi yang komprehensif berbasis prekursor geomagnetik dengan mengintegrasikan data multi-parameter melalui arsitektur EfficientNet-B0 yang dioptimalkan secara hierarkis.

### 5.2 Tujuan Khusus

**Tahun Pertama:**
1. Mengumpulkan dan memproses data geomagnetik dari jaringan magnetometer BMKG
2. Mengembangkan model CNN untuk deteksi prekursor dengan akurasi > 80%
3. Membuat generator data sintetis untuk augmentasi dataset
4. Mengembangkan model dengan fitur self-updating dengan akurasi > 95%

**Tahun Kedua:**
1. Mengumpulkan dan mengintegrasikan data pendukung (seismik, ionosferik, geoatmosferik)
2. Mengembangkan model prediksi parameter gempa dengan akurasi > 85%
3. Mengimplementasikan fitur online learning
4. Mencapai akurasi sistem keseluruhan > 95%

---

## 6. Kebaharuan Penelitian (Novelty)

1. **Sistem Prediksi Komprehensif**: Berbeda dengan model CNN yang hanya melakukan klasifikasi, penelitian ini mengembangkan sistem prediksi gempa yang komprehensif dengan mengintegrasikan data seismik, ionosferik, dan geoatmosferik untuk membentuk satu framework prediksi yang lebih akurat dan holistik.

2. **Prediksi Parameter Gempa**: Memungkinkan prediksi parameter gempa (magnitudo, lokasi, waktu) yang lebih mendetail, bukan hanya deteksi ada/tidaknya prekursor.

3. **Real-time Operation**: Sistem prediksi dirancang untuk pengoperasian secara real-time dengan integrasi streaming data.

4. **Online Learning**: Pendekatan online learning memungkinkan pembaruan model secara instan sehingga sistem dapat memberikan peringatan dini yang responsif terhadap perubahan kondisi lingkungan seismik di Indonesia.

---

## 7. Metodologi Singkat

### 7.1 Data
- **Sumber**: Jaringan Magnetometer BMKG (25 stasiun)
- **Komponen**: H (horizontal), D (deklinasi), Z (vertikal)
- **Periode**: 2018-2024 (minimal 5 tahun)
- **Sampling**: 1 Hz

### 7.2 Preprocessing
- Filtering pada rentang PC3 (10-45 mHz)
- Perhitungan rasio Z/H per jam
- Ekstraksi fitur dengan Wavelet Scattering Transform (WST)
- Transformasi STFT untuk menghasilkan spektrogram
- Ukuran output: 224×224 pixels

### 7.3 Model Tahun 1
- **Arsitektur**: Convolutional Neural Network (CNN)
- **Fitur**: Self-updating untuk adaptasi kondisi seismotektonik lokal
- **Augmentasi**: Generator data sintetis (SMOTE)
- **Target**: Akurasi > 95%

### 7.4 Model Tahun 2
- **Arsitektur**: EfficientNet-B0 Hierarkis
- **Deteksi**: Binary classification (92.5% accuracy)
- **Estimasi**: Multi-task magnitude prediction (100% recall M6+)
- **Lokalisasi**: Azimuth optimization (46-57% accuracy)
- **Integrasi**: Data multi-parameter dengan PCA
- **Output**: Estimasi magnitudo, lokasi, waktu kejadian
- **Fitur**: Online learning

### 7.5 Validasi
- Leave-One-Event-Out (LOEO) Cross-Validation
- Confusion Matrix, Precision, Recall, F1-Score
- Grad-CAM untuk interpretabilitas

---

## 8. Target Luaran

### 8.1 Publikasi
| No | Judul | Target Jurnal | Timeline |
|----|-------|---------------|----------|
| 1 | Deep Learning-Based Earthquake Precursor Detection | IEEE TGRS (Q1) | Tahun 1 |
| 2 | Multi-Parameter Integration for Earthquake Prediction | Nature Scientific Reports (Q1) | Tahun 2 |
| 3 | Online Learning Framework for Earthquake Monitoring | JGR Solid Earth (Q1) | Tahun 2 |

### 8.2 Produk
1. Model CNN untuk deteksi prekursor (Production Ready)
2. Model EfficientNet-B0 Hierarkis untuk deteksi dan prediksi parameter gempa
3. Dashboard monitoring real-time
4. Auto-update pipeline dengan online learning
5. Dataset spektrogram geomagnetik Indonesia

### 8.3 HKI
1. Hak Cipta software sistem deteksi dan prediksi prekursor
2. Paten metode prediksi gempa berbasis arsitektur EfficientNet-B0 hierarkis

---

## 9. Jadwal Penelitian

### Tahun Pertama (2025-2026)
| Bulan | Kegiatan |
|-------|----------|
| 1-3 | Pengumpulan dan persiapan data geomagnetik |
| 3-5 | Ekstraksi fitur (WST, Z/H ratio, spektrogram) |
| 5-7 | Pembuatan model deteksi CNN |
| 7-9 | Pembuatan data augmentasi (SMOTE) |
| 9-12 | Pengembangan model CNN self-updating + Publikasi 1 |

### Tahun Kedua (2026-2027)
| Bulan | Kegiatan |
|-------|----------|
| 1-3 | Pengumpulan data pendukung (seismik, ionosfer, geoatmosfer) |
| 3-5 | Pre-processing dan integrasi multi-parameter |
| 5-8 | Optimasi arsitektur EfficientNet-B0 Hierarkis |
| 8-9 | Generator data sintetis untuk data pendukung |
| 9-11 | Implementasi online learning |
| 11-12 | Finalisasi + Publikasi 2 & 3 |

---

## 10. Manfaat Penelitian

### 10.1 Manfaat Ilmiah
- Memperkaya pengetahuan tentang fenomena fisik yang terjadi sebelum gempa bumi
- Hubungan antara anomali geomagnetik dan aktivitas seismik
- Framework untuk integrasi data multi-parameter dalam prediksi gempa

### 10.2 Manfaat Teknologi
- Metode dan alat analisis data yang lebih baik untuk deteksi dini gempa bumi
- Model yang dapat di-deploy di BMKG untuk operasional

### 10.3 Manfaat Masyarakat
- Meningkatkan kesiapsiagaan masyarakat terhadap bencana gempa bumi
- Sistem early warning yang lebih akurat

### 10.4 Manfaat Pemerintah
- Informasi yang berguna untuk pengambilan keputusan dalam upaya mitigasi bencana

---

## 11. Referensi Utama

1. Hayakawa, M. (2016). Earthquake prediction with electromagnetic phenomena.
2. Hattori, K. et al. (2006). ULF geomagnetic anomaly associated with earthquakes.
3. Petrescu, L. & Moldovan, I. (2022). Prospective neural network model for seismic precursory signal detection.
4. Hamidi, M. et al. (2024). Investigating ULF emissions as earthquake precursors in Sumatra.
5. Marzuki, M. et al. (2022). ULF geomagnetic anomaly associated with Sumatra-Pagai Islands earthquake swarm.
6. Yusof, K.A. et al. (2021). Correlations between earthquake properties and ULF geomagnetic precursor.
7. Bao, Z. et al. (2021). A deep learning-based electromagnetic signal for earthquake magnitude prediction.
