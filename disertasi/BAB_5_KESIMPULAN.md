# BAB 5
KESIMPULAN DAN SARAN

## 5.1 Kesimpulan

Disertasi ini bertujuan untuk mengembangkan sistem prediksi gempa bumi yang komprehensif, akurat, dan adaptif berbasis sinyal prekursor geomagnetik. Berdasarkan serangkaian eksperimen yang telah dilakukan dan dibahas dalam Bab 4, dapat ditarik kesimpulan holistik sebagai berikut:

1.  **Efektivitas Arsitektur Hierarkis EfficientNet-B0**:
    Penelitian ini berhasil membuktikan bahwa arsitektur **Hierarkis EfficientNet-B0** adalah solusi optimal untuk memproses sinyal geomagnetik dengan efisiensi tinggi.
    *   Blok **MBConv (Mobile Inverted Bottleneck)** terbukti sangat efektif menangkap fitur tekstur frekuensi tinggi (anomali lokal), menghasilkan akurasi deteksi biner hingga **92.5%** dengan parameter minimal (5.3M).
    *   Pendekatan **Hierarchical Scaling** terbukti krusial untuk menangkap durasi dan energi sinyal, mencapai **100% recall** untuk gempa besar (M6.0+), jauh melampaui kemampuan CNN standar (65%).

2.  **Pemecahan Masalah Keterbatasan Data (Data Scarcity)**:
    Strategi augmentasi data berbasis domain frekuensi-waktu (*SpecAugment*) dan metode sintesis *SMOTE*, dikombinasikan dengan kompilasi dataset 8 tahun (2018-2025) yang mencakup 25.783 kejadian gempa, berhasil mengatasi masalah *overfitting*. Hal ini dibuktikan dengan stabilitas performa model pada uji validasi ketat *Leave-One-Station-Out (LOSO)* yang tetap menjaga akurasi >85% pada stasiun baru.

3.  **Kapabilitas Prediksi Komprehensif (Deteksi, Estimasi, Lokalisasi)**:
    Sistem yang dibangun tidak hanya berfungsi sebagai "alarm" (deteksi), tetapi juga mampu memberikan estimasi parameter gempa:
    *   **Magnitudo**: Mampu mengklasifikasikan potensi kekuatan gempa dalam 4 kelas dengan akurasi rata-rata 72.4%.
    *   **Lokalisasi (Azimuth)**: Meskipun merupakan tugas tersulit, perbaikan dataset dan penggunaan *Deep Exit Head* berhasil meningkatkan kemampuan model mengenali arah datangnya gelombang (8 mata angin) dengan akurasi awal **43.65%** (signifikan di atas tebakan acak 12.5%), membuktikan bahwa pola arah terenkode dalam data geomagnetik.

4.  **Inovasi Sistem Cerdas dan Efisien**:
    Implementasi arsitektur **EfficientNet-B0** yang ringan (20 MB) sangat mendukung pengoperasian real-time di stasiun-stasiun BMKG. Selain itu, fitur **Self-Updating Dual-Engine** menjamin keberlanjutan sistem (*sustainability*) dengan kemampuan adaptasi otomatis terhadap perubahan pola seismik tanpa mengganggu layanan operasional.

## 5.2 Implikasi Penelitian

1.  **Implikasi Ilmiah**: Penelitian ini memberikan bukti empiris baru bahwa arsitektur **EfficientNet-B0** mampu mengekstraksi informasi arah dan magnitudo dari sinyal ULF geomagnetik secara efisien, memperkuat teori *LAI Coupling (Lithosphere-Atmosphere-Ionosphere)*.
2.  **Implikasi Praktis**: Arsitektur yang diusulkan siap untuk diimplementasikan sebagai sistem *Early Warning* pendamping (*complementary*) bagi sistem seismograf konvensional BMKG, memberikan *lead time* beberapa hari sebelum kejadian.

## 5.3 Saran

Untuk pengembangan penelitian selanjutnya, disarankan beberapa hal berikut:

1.  **Perluasan Jejaring Sensor**: Akurasi lokalisasi (Azimuth) sangat bergantung pada densitas stasiun. Penambahan sensor magnetometer di wilayah *blank spot* seismik akan meningkatkan presisi triangulasi sumber gempa.
2.  **Integrasi Multi-Prekursor**: Menggabungkan input geomagnetik dengan data *Total Electron Content (TEC)* ionosfer dan gas Radon dalam arsitektur *Multimodal Fusion* untuk meningkatkan reliabilitas prediksi dan mengurangi *False Alarm*.
3.  **Implementasi Edge Computing**: Menerapkan model yang telah dikompresi (via *Knowledge Distillation* atau *Quantization*) langsung pada perangkat logger di stasiun (*On-Edge Processing*) untuk meminimalkan ketergantungan pada koneksi internet dan server pusat.
