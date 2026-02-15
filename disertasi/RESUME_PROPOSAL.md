# Resume Proposal Disertasi

**Dokumen Sumber**: mawan-proposal-final.pdf  
**Tanggal Seminar**: 23 April 2025  
**Tempat**: Ruang Social Space Departemen Teknik Fisika ITS Surabaya

---

## 1. Identitas

| Item | Detail |
|------|--------|
| Judul | Pengembangan Sistem Prediksi Gempa Bumi Komprehensif Berbasis Prekusor Geomagnetik Menggunakan Model Hybrid Deep Learning |
| Nama | Sumawan, ST, MM |
| NRP | 7009232004 |
| Program | Doktor |
| Departemen | Teknik Fisika |
| Fakultas | Teknologi Industri dan Rekayasa Sistem |
| Institusi | Institut Teknologi Sepuluh Nopember (ITS) Surabaya |

### Dosen Pembimbing
1. Dr. Bambang Lelono Widjiantoro, S.T., M.T. (NIP 196905071995121001)
2. Prof. Dr. Katherin Indriawati, S.T, M.T. (NIP 197605232000122001)
3. Dr. Muhamad Syirojudin, M.Si. (NIP 198508092008011006)

### Dosen Penguji
1. Dr. Detak Yan Pratama, S.T., M.Sc. (NIP 198401012012121002)
2. Dr. Ir. Dwa D. Warnana S.Si., M.Si. (NIP 197601232000031001)

---

## 2. Abstrak

Prediksi gempa bumi merupakan tantangan besar dalam upaya mitigasi bencana. Studi menunjukkan bahwa sinyal prekursor pada rekaman medan geomagnetik, seperti perubahan rasio Z/H, dapat muncul **7-11 hari** sebelum gempa terjadi. Namun, keterbatasan dataset historis dan inkonsistensi dalam mendeteksi sinyal ini menghambat pengembangan sistem prediksi yang andal.

**Tujuan**: Mengembangkan sistem prediksi gempabumi yang komprehensif berbasis prekursor geomagnetik, dengan mengintegrasikan data multi-parameter melalui model hybrid deep learning.

**Tahun Pertama**:
- Deteksi sinyal prekursor dari data geomagnetik (H, D, Z) minimal 5 tahun dari BMKG
- Ekstraksi fitur menggunakan Wavelet Scattering Transform (WST) dan analisis rasio Z/H
- Model CNN dengan fitur self-updating
- Generator data sintetis untuk augmentasi dataset

**Tahun Kedua**:
- Integrasi dengan data pendukung (seismik, ionosferik, geoatmosferik)
- Fusi data level fitur dengan PCA
- Model hybrid deep learning (CNN + RNN)
- Estimasi parameter gempa: magnitudo, lokasi, waktu kejadian

**Kata Kunci**: Prediksi gempa, sinyal precursor, data geomagnetik, hybrid deep learning, integrasi multi-parameter

---

## 3. Latar Belakang

### 3.1 Kondisi Indonesia
- Terletak di antara 3 lempeng tektonik: Eurasia, Australia, Pasifik
- **781 kejadian** gempa bumi merusak (data BNPB)
- **16.500 korban jiwa** meninggal
- **28 kejadian** tsunami dengan **5.064 korban** meninggal
- **10.789 gempa** tercatat selama 2023

### 3.2 Sistem Eksisting BMKG
- 530+ unit seismograph digital
- 17+ lokasi magnetometer (sejak 2017)
- Metode: Analisis rasio Z/H
- Akurasi saat ini: **< 70%**
- Hit rate: 70-80% (7-11 hari sebelum gempa)

### 3.3 Kesenjangan Penelitian
1. Validasi di berbagai wilayah seismik Indonesia
2. Frekuensi dominan emisi ULF di Indonesia
3. Tingkat akurasi prekursor yang masih rendah
4. Belum ada sistem prediksi komprehensif (magnitudo, lokasi, waktu)

---

## 4. Rumusan Masalah

1. Bagaimana cara mendeteksi prekursor seismik pada rekaman medan geomagnetik secara konsisten?
2. Bagaimana cara mengatasi keterbatasan dataset yang diperlukan untuk melabeli prekursor seismik dengan akurat?
3. Bagaimana cara mencapai sistem prediksi gempa bumi jangka pendek yang komprehensif (magnitudo, lokasi, waktu)?

---

## 5. Tujuan Penelitian

### 5.1 Tujuan Umum
Mengembangkan model prediksi gempa bumi yang lebih akurat dengan memanfaatkan data anomali magnet bumi.

### 5.2 Tujuan Khusus
1. Merancang dan mengimplementasikan algoritma deteksi prekursor seismik secara konsisten
2. Membuat data sintetik untuk mengatasi keterbatasan dataset
3. Membangun sistem prediksi gempa komprehensif (magnitudo, lokasi, waktu)

---

## 6. Kebaharuan Penelitian (Novelty)

1. **Sistem Prediksi Komprehensif**: Berbeda dengan model CNN yang hanya klasifikasi, penelitian ini mengintegrasikan data seismik, ionosferik, dan geoatmosferik untuk prediksi parameter gempa (magnitudo, lokasi, waktu)

2. **Real-time Operation**: Sistem dirancang untuk operasi real-time dengan integrasi streaming data

3. **Online Learning**: Pendekatan online learning memungkinkan pembaruan model secara instan untuk adaptasi terhadap perubahan kondisi seismik Indonesia

---

## 7. State of the Art

### 7.1 Penelitian Terdahulu (Kelompok Statistik/Matematik)
- Hayakawa et al. (2021) - Korelasi statistik
- Ouyang et al. (2020) - Wavelet analysis
- Hamidi et al. (2024) - Polarisasi Z/H
- Marzuki et al. (2022) - Spektrum analysis

### 7.2 Penelitian Terdahulu (Machine Learning/Deep Learning)
- Petrescu & Moldovan (2022) - CNN untuk deteksi prekursor
- Pappoe et al. (2023) - Machine learning + wavelet
- Yusof et al. (2021) - AutoML dengan CNN (akurasi 83.29%)
- Bao et al. (2021) - 3D CNN untuk klasifikasi magnitudo

### 7.3 Gap yang Diisi
- Penelitian sebelumnya hanya deteksi, belum prediksi parameter
- Belum ada integrasi multi-parameter
- Belum ada fitur self-updating dan online learning

---

## 8. Dasar Teori

### 8.1 Ultra Low Frequency (ULF)
- Rentang frekuensi: f < 10 Hz
- Sumber: Solar wind, Magnetosphere, Ionosphere, Lithosphere (LAI Coupling)
- Klasifikasi: Pc3-Pc5 untuk prekursor gempa

### 8.2 Mekanisme Seismo-Elektromagnetik
1. **Efek Microfracturing**: Retakan pada batuan mengubah permitivitas dan konduktivitas dielektrik
2. **Electrokinetic Effect**: Aliran fluida dalam batuan menghasilkan arus listrik
3. **Piezoelectric Effect**: Tekanan pada mineral kuarsa menghasilkan medan listrik

### 8.3 Metode Prekursor
- Analisis rasio Z/H (komponen vertikal/horizontal)
- Window: 7-11 hari sebelum gempa
- Threshold anomali: Z/H > nilai normal

---

## 9. Metodologi

### 9.1 Tahun Pertama

#### A. Pengumpulan Data
- Sumber: Jaringan BMKG (25 stasiun)
- Komponen: H, D, Z
- Periode: 2018-2024 (minimal 5 tahun)

#### B. Ekstraksi Fitur
- Wavelet Scattering Transform (WST)
- Analisis rasio Z/H
- Spektrogram STFT

#### C. Model Deteksi CNN
- Arsitektur: Convolutional Neural Network
- Target akurasi: > 80%
- Fitur: Self-updating

#### D. Data Augmentasi
- SMOTE untuk class balancing
- Generator data sintetis

### 9.2 Tahun Kedua

#### A. Data Pendukung
- Data seismik
- Data ionosferik (TEC)
- Data geoatmosferik

#### B. Integrasi Multi-Parameter
- Fusi level fitur dengan PCA
- Vektor fitur lengkap

#### C. Model Hybrid Deep Learning
- CNN untuk ekstraksi fitur spasial
- RNN untuk analisis temporal
- Target: Estimasi magnitudo, lokasi, waktu

#### D. Online Learning
- Pembaruan model real-time
- Adaptasi kondisi seismik lokal

---

## 10. Batasan Penelitian

1. **Wilayah**: Fokus pada wilayah dengan aktivitas seismik tinggi di Indonesia
2. **Jenis Gempa**: Gempa tektonik dengan magnitudo di atas threshold tertentu
3. **Data**: Data anomali magnet bumi dari stasiun BMKG (2018-2024)
4. **Metode**: Metode analisis yang relevan dengan penelitian

---

## 11. Manfaat Penelitian

### 11.1 Ilmiah
- Memperkaya pengetahuan tentang fenomena fisik sebelum gempa
- Hubungan anomali magnet bumi dengan aktivitas seismik

### 11.2 Teknologi
- Metode dan alat analisis data yang lebih baik
- Sistem deteksi dini gempa bumi

### 11.3 Masyarakat
- Meningkatkan kesiapsiagaan terhadap bencana gempa

### 11.4 Pemerintah
- Informasi untuk pengambilan keputusan mitigasi bencana

---

## 12. Referensi Utama

1. Hayakawa, M. (2016) - Fenomena elektromagnetik dalam prediksi gempa
2. Hattori et al. (2006) - Mekanisme fisis emisi ULF
3. Petrescu & Moldovan (2022) - CNN untuk deteksi prekursor
4. Hamidi et al. (2024) - Korelasi ULF dan gempa di Sumatera
5. Marzuki et al. (2022) - Anomali ULF di Kepulauan Pagai
6. Yusof et al. (2021) - AutoML untuk prediksi gempa

---

## 13. Peta Jalan Penelitian

```
Prekursor Gempa Bumi
├── Data Geomagnet
│   ├── Variasi Geomagnet
│   ├── VHF Geomagnet
│   └── ULF Geomagnet ← [DIPILIH]
├── Medan Listrik
├── Emisi Gas Radon
└── Data Seismic

Model Analisis
├── Matematik/Statistik
│   ├── PCA dan Singular Spektrum
│   ├── Wavelet Transform Analysis
│   └── Rasio Polarisasi Z/H
└── Machine Learning ← [DIPILIH]
    ├── CNN ← [DIPILIH]
    ├── LSTM
    ├── SVM
    └── RNN ← [DIPILIH]
```

---

## 14. Jadwal Pelaksanaan

### Tahun Pertama
| Bulan | Kegiatan |
|-------|----------|
| 1-3 | Pengumpulan dan persiapan data geomagnetik |
| 3-5 | Ekstraksi fitur (WST, Z/H ratio) |
| 5-7 | Pembuatan model deteksi CNN |
| 7-9 | Pembuatan data augmentasi (SMOTE) |
| 9-12 | Pengembangan model CNN self-updating |

### Tahun Kedua
| Bulan | Kegiatan |
|-------|----------|
| 1-3 | Pengumpulan data pendukung |
| 3-5 | Pre-processing data pendukung |
| 5-6 | Integrasi data multi-parameter |
| 6-8 | Pengembangan model prediksi parameter gempa |
| 8-9 | Generator data sintetis untuk data pendukung |
| 9-11 | Pengembangan model online learning |
| 11-12 | Finalisasi dan publikasi |
