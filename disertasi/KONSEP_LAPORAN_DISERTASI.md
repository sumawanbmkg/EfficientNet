# KONSEP RENCANA LAPORAN DISERTASI

**Berdasarkan**: Panduan Penyusunan Disertasi Program Doktor ITS 2021

---

## IDENTITAS DISERTASI

| Item | Detail |
|------|--------|
| **Judul** | Pengembangan Sistem Prediksi Gempa Bumi Komprehensif Berbasis Prekusor Geomagnetik Menggunakan Arsitektur EfficientNet-B0 Hierarkis |
| **Nama** | Sumawan, ST, MM |
| **NRP** | 7009232004 |
| **Program** | Doktor |
| **Bidang Keahlian** | Instrumentasi |
| **Departemen** | Teknik Fisika |
| **Fakultas** | Teknologi Industri dan Rekayasa Sistem |
| **Institusi** | Institut Teknologi Sepuluh Nopember (ITS) Surabaya |

### Dosen Pembimbing
1. Dr. Bambang Lelono Widjiantoro, S.T., M.T. (Pembimbing 1)
2. Prof. Dr. Katherin Indriawati, S.T, M.T. (Pembimbing 2)
3. Dr. Muhamad Syirojudin, M.Si. (Pembimbing 3)

---

## STRUKTUR LAPORAN DISERTASI

Berdasarkan Panduan Penyusunan Disertasi ITS, struktur laporan terdiri dari:

### BAGIAN AWAL (Halaman Romawi)
1. Sampul Luar (Cover)
2. Sampul Dalam
3. Lembar Pengesahan
4. Abstrak (Bahasa Indonesia)
5. Abstract (Bahasa Inggris)
6. Kata Pengantar
7. Daftar Isi
8. Daftar Gambar
9. Daftar Tabel
10. Daftar Notasi/Simbol

### BAGIAN UTAMA (Halaman Arabic)
- BAB 1: PENDAHULUAN
- BAB 2: KAJIAN PUSTAKA DAN DASAR TEORI
- BAB 3: METODOLOGI PENELITIAN
- BAB 4: HASIL DAN PEMBAHASAN
- BAB 5: KESIMPULAN DAN SARAN
- DAFTAR PUSTAKA
- LAMPIRAN

---

## OUTLINE DETAIL PER BAB


### ═══════════════════════════════════════════════════════════════════════
### BAB 1: PENDAHULUAN
### ═══════════════════════════════════════════════════════════════════════

**Estimasi Halaman**: 10-15 halaman

#### 1.1 Latar Belakang
- Kondisi tektonik Indonesia (Ring of Fire, 3 lempeng)
- Statistik gempa bumi dan dampaknya (data BNPB)
- Sistem monitoring BMKG saat ini dan keterbatasannya
- Fenomena prekursor geomagnetik (ULF, rasio Z/H)
- Potensi deep learning untuk meningkatkan akurasi deteksi
- Gap penelitian yang akan diisi

**Konten yang Sudah Ada**:
- ✅ Data statistik gempa Indonesia
- ✅ Informasi sistem BMKG
- ✅ Teori prekursor geomagnetik

#### 1.2 Perumusan Masalah
1. Bagaimana cara mendeteksi prekursor seismik pada rekaman medan geomagnetik secara konsisten?
2. Bagaimana cara mengatasi keterbatasan dataset yang diperlukan untuk melabeli prekursor seismik dengan akurat?
3. Bagaimana cara mencapai sistem prediksi gempa bumi jangka pendek yang komprehensif (magnitudo, lokasi, waktu)?

#### 1.3 Tujuan Penelitian
**Tujuan Umum**:
Mengembangkan sistem prediksi gempa bumi yang komprehensif berbasis prekursor geomagnetik dengan mengintegrasikan data multi-parameter melalui arsitektur EfficientNet-B0 hierarkis.

**Tujuan Khusus**:
1. Merancang algoritma deteksi prekursor seismik yang konsisten
2. Membuat generator data sintetis untuk mengatasi keterbatasan dataset
3. Membangun sistem prediksi gempa komprehensif (magnitudo, lokasi, waktu)

#### 1.4 Batasan Penelitian
- Wilayah: Indonesia dengan aktivitas seismik tinggi
- Jenis gempa: Tektonik dengan M ≥ 5.0
- Data: Geomagnetik BMKG 2018-2025
- Metode: Deep learning (CNN, RNN)

#### 1.5 Manfaat Penelitian
- Manfaat ilmiah
- Manfaat teknologi
- Manfaat masyarakat
- Manfaat pemerintah

#### 1.6 Sistematika Penulisan
- Deskripsi singkat setiap bab

---

### ═══════════════════════════════════════════════════════════════════════
### BAB 2: KAJIAN PUSTAKA DAN DASAR TEORI
### ═══════════════════════════════════════════════════════════════════════

**Estimasi Halaman**: 25-35 halaman

#### 2.1 Gempa Bumi
- 2.1.1 Definisi dan Mekanisme Gempa Bumi
- 2.1.2 Klasifikasi Gempa Bumi
- 2.1.3 Parameter Gempa Bumi (Magnitudo, Kedalaman, Lokasi)
- 2.1.4 Seismisitas Indonesia

#### 2.2 Prekursor Gempa Bumi
- 2.2.1 Definisi Prekursor Seismik
- 2.2.2 Jenis-jenis Prekursor
  - Prekursor Geomagnetik
  - Prekursor Ionosferik
  - Prekursor Geoatmosferik
  - Prekursor Seismik
- 2.2.3 Mekanisme Seismo-Elektromagnetik
  - Efek Microfracturing
  - Electrokinetic Effect
  - Piezoelectric Effect
- 2.2.4 Lithosphere-Atmosphere-Ionosphere (LAI) Coupling

#### 2.3 Sinyal Ultra Low Frequency (ULF)
- 2.3.1 Definisi dan Karakteristik ULF
- 2.3.2 Sumber Sinyal ULF
- 2.3.3 Klasifikasi Pulsasi Geomagnetik (Pc1-Pc5)
- 2.3.4 Metode Analisis Rasio Z/H
- 2.3.5 Window Prekursor (7-11 hari)

#### 2.4 Transformasi Sinyal
- 2.4.1 Short-Time Fourier Transform (STFT)
- 2.4.2 Wavelet Transform
- 2.4.3 Wavelet Scattering Transform (WST)
- 2.4.4 Spektrogram sebagai Representasi Visual

#### 2.5 Deep Learning
- 2.5.1 Artificial Neural Network (ANN)
- 2.5.2 Convolutional Neural Network (CNN)
  - Arsitektur CNN
  - Transfer Learning
  - EfficientNet
  - ConvNeXt
- 2.5.3 Recurrent Neural Network (RNN)
  - LSTM
  - GRU
- 2.5.4 Arsitektur EfficientNet-B0 Hierarkis (Multi-Task Learning)

#### 2.6 Teknik Augmentasi Data
- 2.6.1 Image Augmentation
- 2.6.2 SMOTE (Synthetic Minority Over-sampling Technique)
- 2.6.3 Focal Loss untuk Class Imbalance

#### 2.7 Metode Validasi Model
- 2.7.1 Cross-Validation
- 2.7.2 Leave-One-Event-Out (LOEO)
- 2.7.3 Metrik Evaluasi (Accuracy, Precision, Recall, F1, MCC)
- 2.7.4 Grad-CAM untuk Interpretabilitas

#### 2.8 State of the Art
- 2.8.1 Penelitian Prekursor dengan Metode Statistik
- 2.8.2 Penelitian Prekursor dengan Machine Learning
- 2.8.3 Gap Penelitian dan Posisi Penelitian Ini

**Konten yang Sudah Ada**:
- ✅ Teori ULF dan prekursor
- ✅ Arsitektur CNN (EfficientNet, ConvNeXt)
- ✅ Metode LOEO validation
- ✅ State of the art references

---

### ═══════════════════════════════════════════════════════════════════════
### BAB 3: METODOLOGI PENELITIAN
### ═══════════════════════════════════════════════════════════════════════

**Estimasi Halaman**: 20-30 halaman

#### 3.1 Diagram Alir Penelitian
- Flowchart keseluruhan penelitian (2 tahun)
- Penjelasan setiap tahap

**Gambar yang Sudah Ada**:
- ✅ `disertasi/figures/flowchart_penelitian.png`
- ✅ `disertasi/figures/progress_flowchart.png`

#### 3.2 Data Penelitian
- 3.2.1 Sumber Data
  - Jaringan Magnetometer BMKG (25 stasiun)
  - Katalog Gempa BMKG
- 3.2.2 Spesifikasi Data
  - Komponen: H, D, Z
  - Sampling rate: 1 Hz
  - Periode: 2018-2025
- 3.2.3 Kriteria Pemilihan Event
  - Magnitudo ≥ 5.0
  - Jarak stasiun ≤ 500 km
  - Window prekursor: 7-11 hari

**Data yang Sudah Ada**:
- ✅ `earthquake_catalog_2018_2025_merged.csv`
- ✅ 25 stasiun aktif
- ✅ 2000+ spektrogram

#### 3.3 Pre-processing Data
- 3.3.1 Pembacaan Data Binary
- 3.3.2 Filtering Bandpass PC3 (10-45 mHz)
- 3.3.3 Perhitungan Rasio Z/H
- 3.3.4 Quality Control dan Outlier Removal
- 3.3.5 Normalisasi Data

**Script yang Sudah Ada**:
- ✅ `geomagnetic_dataset_generator_ssh_v2.py`

#### 3.4 Ekstraksi Fitur dan Pembuatan Spektrogram
- 3.4.1 Transformasi STFT
- 3.4.2 Parameter STFT (window, overlap, nfft)
- 3.4.3 Konversi ke Spektrogram RGB
- 3.4.4 Resize ke 224×224 pixels

#### 3.5 Arsitektur Model CNN
- 3.5.1 Pemilihan Backbone (EfficientNet-B0)
- 3.5.2 Transfer Learning dari ImageNet
- 3.5.3 Modifikasi Classifier Head
- 3.5.4 Hyperparameter Training

**Model yang Sudah Ada**:
- ✅ EfficientNet-B0 (97.47%)
- ✅ ConvNeXt-Tiny (95.2%)

#### 3.6 Augmentasi Data
- 3.6.1 Image Augmentation (rotation, flip, brightness)
- 3.6.2 SMOTE untuk Class Balancing
- 3.6.3 Focal Loss Implementation

**Script yang Sudah Ada**:
- ✅ `generate_smote_dataset.py`
- ✅ `generate_augmented_dataset.py`

#### 3.7 Pipeline Self-Updating
- 3.7.1 Arsitektur Pipeline
- 3.7.2 Data Ingestion Module
- 3.7.3 Trainer Module
- 3.7.4 Evaluator Module
- 3.7.5 Champion-Challenger Comparison
- 3.7.6 Deployer Module

**Implementasi yang Sudah Ada**:
- ✅ `autoupdate_pipeline/` (complete)

#### 3.8 Validasi Model
- 3.8.1 Strategi LOEO Cross-Validation
- 3.8.2 Metrik Evaluasi
- 3.8.3 Grad-CAM Analysis

#### 3.9 Integrasi Multi-Parameter (Tahun 2)
- 3.9.1 Data Pendukung (Ionosfer, Seismik, Geoatmosfer)
- 3.9.2 Strategi Multi-Head Output (Binary, Magnitude, Azimuth)
- 3.9.3 Arsitektur EfficientNet-B0 Hierarkis

#### 3.10 Online Learning (Tahun 2)
- 3.10.1 Konsep Online Learning
- 3.10.2 Elastic Weight Consolidation (EWC)
- 3.10.3 Implementasi Incremental Update

---

### ═══════════════════════════════════════════════════════════════════════
### BAB 4: HASIL DAN PEMBAHASAN
### ═══════════════════════════════════════════════════════════════════════

**Estimasi Halaman**: 40-60 halaman

#### 4.1 Hasil Pengumpulan Data
- 4.1.1 Statistik Data Geomagnetik
- 4.1.2 Distribusi Event Gempa
- 4.1.3 Kualitas Data per Stasiun

**Data yang Sudah Ada**:
- ✅ 25 stasiun, 7 tahun data
- ✅ 105+ gempa M≥6.0
- ✅ 2000+ spektrogram

#### 4.2 Hasil Pre-processing
- 4.2.1 Contoh Sinyal Sebelum dan Sesudah Filtering
- 4.2.2 Distribusi Rasio Z/H
- 4.2.3 Identifikasi Anomali

#### 4.3 Hasil Ekstraksi Fitur
- 4.3.1 Contoh Spektrogram Prekursor vs Normal
- 4.3.2 Karakteristik Visual Spektrogram per Kelas
- 4.3.3 Distribusi Dataset

**Gambar yang Sudah Ada**:
- ✅ Contoh spektrogram di `dataset_unified/`
- ✅ Visualisasi augmentasi

#### 4.4 Hasil Training Model CNN
- 4.4.1 Perbandingan Arsitektur (VGG16 vs EfficientNet-B0 vs ConvNeXt)
- 4.4.2 Learning Curve
- 4.4.3 Confusion Matrix
- 4.4.4 Metrik per Kelas

**Hasil yang Sudah Ada**:
| Model | Accuracy | Precision | Recall | F1 |
|-------|----------|-----------|--------|-----|
| VGG16 | 92.3% | 91.8% | 92.1% | 91.9% |
| EfficientNet-B0 | **97.47%** | 97.2% | 97.5% | 97.3% |
| ConvNeXt-Tiny | 95.2% | 94.8% | 95.1% | 94.9% |
| Xception | 91.8% | 91.2% | 91.6% | 91.4% |

#### 4.5 Hasil Validasi LOEO
- 4.5.1 Akurasi per Event
- 4.5.2 Analisis Event dengan Akurasi Rendah
- 4.5.3 Stabilitas Model

**Hasil yang Sudah Ada**:
- ✅ LOEO mean accuracy: 93.2%
- ✅ Hasil per event tersedia

#### 4.6 Hasil Augmentasi Data
- 4.6.1 Distribusi Kelas Sebelum dan Sesudah SMOTE
- 4.6.2 Pengaruh Augmentasi terhadap Akurasi
- 4.6.3 Analisis Class Imbalance

#### 4.7 Hasil Grad-CAM Analysis
- 4.7.1 Visualisasi Attention Map
- 4.7.2 Interpretasi Fitur yang Dipelajari Model
- 4.7.3 Validasi dengan Pengetahuan Domain

**Gambar yang Sudah Ada**:
- ✅ Grad-CAM visualizations

#### 4.8 Hasil Pipeline Self-Updating
- 4.8.1 Skenario Testing
- 4.8.2 Performa Champion-Challenger
- 4.8.3 Waktu Eksekusi Pipeline

**Implementasi yang Sudah Ada**:
- ✅ `autoupdate_pipeline/` tested

#### 4.9 Hasil Prediksi Parameter Gempa
- 4.9.1 Prediksi Azimuth (8 kelas)
- 4.9.2 Prediksi Magnitude (5 kelas)
- 4.9.3 Analisis Error

**Hasil yang Sudah Ada**:
- ✅ Azimuth accuracy: 96.8%
- ✅ Magnitude accuracy: 94.4%

- 4.10.1 Korelasi antar Parameter Prekursor
- 4.10.2 Hasil Prediksi Terpadu
- 4.10.3 Performa Arsitektur Hierarkis

#### 4.11 Hasil Online Learning (Tahun 2)
- 4.11.1 Adaptasi terhadap Data Baru
- 4.11.2 Pencegahan Catastrophic Forgetting
- 4.11.3 Performa Real-time

#### 4.12 Pembahasan
- 4.12.1 Perbandingan dengan Penelitian Sebelumnya
- 4.12.2 Analisis Kelebihan dan Keterbatasan
- 4.12.3 Implikasi untuk Sistem Early Warning
- 4.12.4 Potensi Implementasi di BMKG

---

### ═══════════════════════════════════════════════════════════════════════
### BAB 5: KESIMPULAN DAN SARAN
### ═══════════════════════════════════════════════════════════════════════

**Estimasi Halaman**: 3-5 halaman

#### 5.1 Kesimpulan
1. Model CNN berbasis EfficientNet-B0 berhasil mendeteksi prekursor geomagnetik dengan akurasi 97.47%, melebihi target 95%
2. Generator data sintetis dengan SMOTE efektif mengatasi class imbalance
3. Pipeline self-updating memungkinkan adaptasi model terhadap data baru
4. Prediksi parameter gempa (azimuth, magnitude) mencapai akurasi >94%
5. [Kesimpulan Tahun 2 - TBD]

#### 5.2 Saran
1. Pengembangan integrasi dengan data ionosfer dan geoatmosfer
2. Implementasi online learning untuk adaptasi real-time
3. Deployment sistem di BMKG untuk operasional
4. Penelitian lanjutan untuk prediksi lokasi dan waktu yang lebih presisi

---

### ═══════════════════════════════════════════════════════════════════════
### DAFTAR PUSTAKA
### ═══════════════════════════════════════════════════════════════════════

**Estimasi**: 50-80 referensi

**Format**: Harvard Style (sesuai panduan ITS)

**Kategori Referensi**:
1. Jurnal Ilmiah (minimal 30% dari total)
2. Buku Teks
3. Prosiding Konferensi
4. Disertasi/Tesis
5. Standar Teknis
6. Dokumen Pemerintah (BMKG, BNPB)

**Referensi Utama yang Sudah Ada**:
- Hayakawa, M. (2016) - Electromagnetic phenomena
- Hattori, K. et al. (2006) - ULF geomagnetic anomaly
- Petrescu, L. & Moldovan, I. (2022) - CNN for precursor detection
- Hamidi, M. et al. (2024) - ULF emissions in Sumatra
- Marzuki, M. et al. (2022) - ULF anomaly Pagai Islands
- Yusof, K.A. et al. (2021) - AutoML for earthquake prediction

---

### ═══════════════════════════════════════════════════════════════════════
### LAMPIRAN
### ═══════════════════════════════════════════════════════════════════════

#### Lampiran A: Daftar Stasiun Magnetometer BMKG
- Tabel 25 stasiun dengan koordinat

#### Lampiran B: Katalog Gempa Bumi 2018-2025
- Tabel event gempa yang digunakan

#### Lampiran C: Kode Program Utama
- C.1 Pre-processing Pipeline
- C.2 Model Training Script
- C.3 Auto-Update Pipeline

#### Lampiran D: Hasil LOEO per Event
- Tabel detail akurasi per event

#### Lampiran E: Confusion Matrix Detail
- Per kelas azimuth dan magnitude

#### Lampiran F: Publikasi
- Paper yang sudah/akan dipublikasikan

---


## CHECKLIST KELENGKAPAN DISERTASI

### Bagian Awal
| No | Item | Status | Catatan |
|----|------|--------|---------|
| 1 | Sampul Luar | ⏳ | Warna Fire Brick (FTIRS) |
| 2 | Sampul Dalam | ⏳ | |
| 3 | Lembar Pengesahan | ⏳ | Setelah ujian |
| 4 | Abstrak (Indonesia) | 🟡 | Draft ada |
| 5 | Abstract (English) | 🟡 | Draft ada |
| 6 | Kata Pengantar | ⏳ | |
| 7 | Daftar Isi | ⏳ | Auto-generate |
| 8 | Daftar Gambar | ⏳ | Auto-generate |
| 9 | Daftar Tabel | ⏳ | Auto-generate |
| 10 | Daftar Notasi | ⏳ | |

### Bagian Utama
| No | Bab | Status | Progress |
|----|-----|--------|----------|
| 1 | BAB 1: Pendahuluan | 🟡 | 80% - Konten ada di proposal |
| 2 | BAB 2: Kajian Pustaka | 🟡 | 70% - Perlu ekspansi |
| 3 | BAB 3: Metodologi | ✅ | 90% - Implementasi lengkap |
| 4 | BAB 4: Hasil & Pembahasan | 🟡 | 60% - Tahun 1 selesai |
| 5 | BAB 5: Kesimpulan | ⏳ | 30% - Menunggu Tahun 2 |
| 6 | Daftar Pustaka | 🟡 | 50% - Perlu tambahan |
| 7 | Lampiran | 🟡 | 40% - Data tersedia |

### Gambar yang Diperlukan
| No | Gambar | Status | File |
|----|--------|--------|------|
| 1 | Flowchart Penelitian | ✅ | `figures/flowchart_penelitian.png` |
| 2 | Progress Flowchart | ✅ | `figures/progress_flowchart.png` |
| 3 | Arsitektur CNN | ⏳ | Perlu dibuat |
| 4 | Pipeline Self-Update | ⏳ | Perlu dibuat |
| 5 | Contoh Spektrogram | ✅ | Ada di dataset |
| 6 | Confusion Matrix | ✅ | Ada di experiments |
| 7 | Learning Curve | ✅ | Ada di training logs |
| 8 | Grad-CAM | ✅ | Ada di visualization |
| 9 | Peta Stasiun BMKG | ⏳ | Perlu dibuat |
| 10 | Distribusi Dataset | ⏳ | Perlu dibuat |

### Tabel yang Diperlukan
| No | Tabel | Status |
|----|-------|--------|
| 1 | Daftar Stasiun Magnetometer | ✅ |
| 2 | Katalog Gempa | ✅ |
| 3 | Distribusi Dataset | ✅ |
| 4 | Perbandingan Model | ✅ |
| 5 | Hasil LOEO | ✅ |
| 6 | Metrik per Kelas | ✅ |
| 7 | Hyperparameter | ⏳ |
| 8 | Hasil Tahun 2 | ⏳ |

---

## FORMAT PENULISAN (Sesuai Panduan ITS)

### Spesifikasi Dokumen
- **Kertas**: A4 - 80 gram
- **Font**: Times New Roman, 12 pt
- **Spasi**: 1.5
- **Margin**:
  - Atas: 3.5 cm
  - Bawah: 3 cm
  - Kiri: 4 cm (halaman ganjil), 3 cm (halaman genap)
  - Kanan: 3 cm (halaman ganjil), 4 cm (halaman genap)

### Penomoran
- Bagian awal: Romawi kecil (i, ii, iii, ...)
- Bagian utama: Arabic (1, 2, 3, ...)
- Posisi: Bawah tengah

### Judul Bab
- Font: 14 pt, Bold, UPPERCASE
- Contoh: **BAB 1** (enter) **PENDAHULUAN**

### Sub-bab
- Font: 12 pt, Bold, Title Case
- Contoh: **1.1 Latar Belakang**

### Gambar
- Nomor dan judul di bawah gambar
- Format: Gambar X.Y Judul Gambar (Sumber, Tahun)
- Centered

### Tabel
- Nomor dan judul di atas tabel
- Format: Tabel X.Y Judul Tabel
- Sumber di bawah tabel

### Rumus
- Menggunakan Equation Editor
- Nomor di kanan: (X.Y)
- Penjelasan variabel di bawah rumus

### Kutipan
- Format Harvard: (Nama, Tahun)
- Kutipan langsung > 1 kalimat: indent 1 tab, spasi 1

---

## TIMELINE PENULISAN DISERTASI

### Fase 1: Persiapan (Bulan 1-2)
- [ ] Finalisasi outline
- [ ] Kumpulkan semua data dan hasil
- [ ] Siapkan template dokumen

### Fase 2: Penulisan Draft (Bulan 3-6)
- [ ] BAB 1: Pendahuluan
- [ ] BAB 2: Kajian Pustaka
- [ ] BAB 3: Metodologi
- [ ] BAB 4: Hasil (Tahun 1)

### Fase 3: Penelitian Tahun 2 (Bulan 7-12)
- [ ] Pengumpulan data pendukung
- [ ] Implementasi model hybrid
- [ ] Online learning

### Fase 4: Penulisan Lanjutan (Bulan 13-18)
- [ ] BAB 4: Hasil (Tahun 2)
- [ ] BAB 5: Kesimpulan
- [ ] Daftar Pustaka
- [ ] Lampiran

### Fase 5: Review & Revisi (Bulan 19-22)
- [ ] Review pembimbing
- [ ] Revisi
- [ ] Proofreading

### Fase 6: Finalisasi (Bulan 23-24)
- [ ] Ujian tertutup
- [ ] Revisi final
- [ ] Ujian terbuka
- [ ] Penjilidan

---

## CATATAN PENTING

### Kebaharuan (Novelty) yang Harus Ditonjolkan
1. **Sistem Prediksi Komprehensif** - Bukan hanya deteksi, tapi prediksi parameter
2. **Hierarchical EfficientNet-B0** - Integrasi Deteksi dan Prediksi Parameter
3. **Self-Updating Pipeline** - Adaptasi otomatis terhadap data baru
4. **Online Learning** - Real-time model update

### Kontribusi Ilmiah
1. Framework deteksi prekursor berbasis deep learning untuk Indonesia
2. Dataset spektrogram geomagnetik Indonesia (2018-2025)
3. Model CNN dengan akurasi 97.47% (state-of-the-art)
4. Pipeline auto-update untuk sistem early warning

### Target Publikasi
| No | Judul | Jurnal Target | Status |
|----|-------|---------------|--------|
| 1 | Deep Learning-Based Earthquake Precursor Detection | IEEE TGRS (Q1) | 🟡 Draft |
| 2 | Multi-Parameter Integration for Earthquake Prediction | Nature Sci. Rep. (Q1) | ⏳ Tahun 2 |
| 3 | Online Learning Framework for Earthquake Monitoring | JGR Solid Earth (Q1) | ⏳ Tahun 2 |

---

## REFERENSI PANDUAN

1. **Panduan Penyusunan Disertasi Program Doktor ITS 2021**
   - Direktorat Pascasarjana dan Pengembangan Akademik
   - Institut Teknologi Sepuluh Nopember

2. **Warna Sampul FTIRS**: Fire Brick, tulisan Putih

3. **Kode Disertasi**: Sesuai format departemen Teknik Fisika

---

*Dokumen ini dibuat sebagai panduan penyusunan laporan disertasi*
*Update terakhir: 11 Februari 2026*
