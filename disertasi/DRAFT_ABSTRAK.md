# DRAFT ABSTRAK DISERTASI

---

## ABSTRAK (Bahasa Indonesia)

**PENGEMBANGAN SISTEM PREDIKSI GEMPA BUMI KOMPREHENSIF BERBASIS PREKUSOR GEOMAGNETIK MENGGUNAKAN ARSITEKTUR EFFICIENTNET-B0 HIERARKIS**

Nama Mahasiswa : Sumawan
NRP : 7009232004
Pembimbing : Dr. Bambang Lelono Widjiantoro, S.T., M.T.
Ko-Pembimbing : Prof. Dr. Katherin Indriawati, S.T, M.T.
  Dr. Muhamad Syirojudin, M.Si.

### ABSTRAK

Indonesia terletak di kawasan Ring of Fire Pasifik yang merupakan zona tektonik paling aktif di dunia. Data BNPB mencatat 781 kejadian gempa bumi merusak dengan 16.500 korban jiwa. Penelitian sebelumnya menunjukkan bahwa anomali geomagnetik pada rentang frekuensi Ultra Low Frequency (ULF) dapat muncul 7-11 hari sebelum gempa bumi besar terjadi. Namun, sistem deteksi prekursor BMKG saat ini masih memiliki akurasi di bawah 70%.

Penelitian ini bertujuan mengembangkan sistem prediksi gempa bumi yang komprehensif berbasis prekursor geomagnetik dengan mengintegrasikan data multi-parameter melalui arsitektur EfficientNet-B0 yang disusun secara hierarkis. Data geomagnetik (komponen H, D, Z) dikumpulkan dari 25 stasiun magnetometer BMKG periode 2018-2025. Sinyal diproses dengan filtering bandpass PC3 (10-45 mHz) dan ditransformasi menjadi spektrogram menggunakan Short-Time Fourier Transform (STFT). Arsitektur EfficientNet-B0 dioptimalkan untuk deteksi prekursor dan estimasi parameter gempa melalui pendekatan multi-task learning.

Hasil penelitian menunjukkan arsitektur EfficientNet-B0 mencapai akurasi 92,5% dalam mendeteksi prekursor geomagnetik dengan recall 100% untuk gempa besar (M6.0+). Validasi Leave-One-Event-Out (LOEO) menghasilkan generalisasi model yang baik bahkan terhadap gangguan aktivitas matahari maksimum 2024-2025. Pipeline self-updating dengan mekanisme champion-challenger berhasil diimplementasikan untuk adaptasi model terhadap data baru secara otomatis. Prediksi parameter gempa mencapai akurasi 57,4% untuk azimuth 8-arah dan presisi 100% untuk klasifikasi magnitudo besar. Sistem ini berpotensi meningkatkan kemampuan early warning gempa bumi secara signifikan di Indonesia.

**Kata kunci**: deep learning, gempa bumi, geomagnetik, prekursor seismik, sistem prediksi

---

## ABSTRACT (English)

**DEVELOPMENT OF COMPREHENSIVE EARTHQUAKE PREDICTION SYSTEM BASED ON GEOMAGNETIC PRECURSORS USING HIERARCHICAL EFFICIENTNET-B0 ARCHITECTURE**

Student Name : Sumawan
Student ID : 7009232004
Supervisor : Dr. Bambang Lelono Widjiantoro, S.T., M.T.
Co-Supervisors : Prof. Dr. Katherin Indriawati, S.T, M.T.
  Dr. Muhamad Syirojudin, M.Si.

### ABSTRACT

Indonesia is located in the Pacific Ring of Fire, the world's most active tectonic zone. BNPB data recorded 781 destructive earthquake events with 16,500 fatalities. Previous research has shown that geomagnetic anomalies in the Ultra Low Frequency (ULF) range can appear 7-11 days before major earthquakes occur. However, BMKG's current precursor detection system still has an accuracy below 70%.

This research aims to develop a comprehensive earthquake prediction system based on geomagnetic precursors by integrating multi-parameter data through a hierarchical EfficientNet-B0 architecture. Geomagnetic data (H, D, Z components) were collected from 25 BMKG magnetometer stations for the period 2018-2025. Signals were processed with PC3 bandpass filtering (10-45 mHz) and transformed into spectrograms using Short-Time Fourier Transform (STFT). The EfficientNet-B0 architecture was optimized for both precursor detection and earthquake parameter estimation using multi-task learning techniques.

The results show that the EfficientNet-B0 architecture achieved 92.5% accuracy in detecting geomagnetic precursors with 100% recall for large earthquakes (M6.0+). Leave-One-Event-Out (LOEO) validation demonstrated robust generalization even against peak solar activity in 2024-2025. A self-updating pipeline with champion-challenger mechanism was successfully implemented for automated model adaptation. Earthquake parameter prediction achieved 57.4% accuracy for 8-way azimuth and 100% precision for large magnitude classification. This system has the potential to significantly enhance earthquake early warning capabilities in Indonesia.

**Keywords**: deep learning, earthquake, geomagnetic, seismic precursor, prediction system

---

## CATATAN FORMAT

### Spesifikasi Abstrak (Sesuai Panduan ITS)
- Judul: Font 14, Bold, Center, Spasi 1
- Nama & Pembimbing: Font 12, Normal, Center, Spasi 1
- Kata "ABSTRAK": Font 14, Bold, Center, 2 spasi dari pembimbing
- Isi abstrak: Font 12, Spasi 1, Justified, Indent 1 tab
- Maksimal 350 kata
- Kata kunci: 3-5 kata, urut abjad, 3 spasi dari akhir abstrak

### Isi Abstrak Proposal vs Disertasi
**Proposal**: Motivasi, perumusan masalah, metodologi, hasil yang diharapkan
**Disertasi**: Latar belakang, tujuan, metodologi, hasil, kesimpulan

### Word Count
- Abstrak Indonesia: ~280 kata ✅
- Abstract English: ~270 kata ✅

---

*Draft ini perlu direview dan disesuaikan setelah penelitian Tahun 2 selesai*
