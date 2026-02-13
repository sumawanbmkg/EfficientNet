# Benchmark Test Set

## Overview

Folder ini berisi fixed benchmark test set yang digunakan untuk evaluasi fair antara champion dan challenger model.

## Important

⚠️ **JANGAN PERNAH MENGUBAH FILE DI FOLDER INI!**

Benchmark test set harus tetap konsisten untuk memastikan perbandingan yang adil antar model.

## Contents

- `benchmark_test.csv` - Metadata benchmark samples
- `spectrograms/` - Spectrogram images untuk benchmark

## Structure

```
benchmark/
├── README.md
├── benchmark_test.csv
└── spectrograms/
    ├── sample_001.png
    ├── sample_002.png
    └── ...
```

## CSV Format

```csv
filename,date,station,magnitude_class,azimuth_class
sample_001.png,2024-01-15,GTO,Large,NE
sample_002.png,2024-02-20,SCN,Medium,E
...
```

## Selection Criteria

Benchmark samples dipilih berdasarkan:
1. Distribusi seimbang antar kelas magnitude
2. Distribusi seimbang antar kelas azimuth
3. Representasi dari semua stasiun
4. Kualitas spectrogram yang baik
5. Event yang sudah terverifikasi

## Usage

Benchmark digunakan oleh `ModelEvaluator` untuk:
1. Evaluasi champion model
2. Evaluasi challenger model
3. Perbandingan fair antar model

## Maintenance

- Benchmark dibuat sekali dan tidak diubah
- Jika perlu update, buat versi baru dengan nama berbeda
- Dokumentasikan setiap perubahan
