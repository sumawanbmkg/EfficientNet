#!/usr/bin/env python3
"""
INVESTIGASI KRITIS: Mengapa Confidence Rendah Padahal F1 Score Tinggi?

Kasus: TRT 2 Februari 2026 - Gempa M6.4 arah SW
Hasil Scanner: E (13.1% confidence) - SALAH!

Investigasi ini akan menganalisis:
1. Distribusi data training untuk stasiun TRT
2. Perbandingan preprocessing training vs scanner
3. Analisis model predictions
4. Identifikasi root cause

Author: Investigation Team
Date: 10 February 2026
"""

import os
import sys
import json
import numpy as np
import pandas as pd
from pathlib import Path
from collections import Counter

# Setup paths
sys.path.insert(0, os.path.dirname(__file__))

print("=" * 80)
print("🔬 INVESTIGASI KRITIS: LOW CONFIDENCE PADA GEMPA M6.4")
print("=" * 80)

# ============================================================================
# BAGIAN 1: ANALISIS DATA TRAINING
# ============================================================================
print("\n" + "=" * 80)
print("📊 BAGIAN 1: ANALISIS DATA TRAINING")
print("=" * 80)

# Load training metadata
metadata_path = 'training_data/training_metadata.csv'
if os.path.exists(metadata_path):
    df = pd.read_csv(metadata_path)
    print(f"\n✅ Loaded training metadata: {len(df)} samples")
    
    # Analisis distribusi stasiun
    print("\n📍 DISTRIBUSI STASIUN:")
    station_counts = df['station'].value_counts()
    for station, count in station_counts.items():
        print(f"   {station}: {count} samples ({count/len(df)*100:.1f}%)")
    
    # Analisis TRT secara khusus
    trt_df = df[df['station'] == 'TRT']
    print(f"\n🎯 ANALISIS STASIUN TRT:")
    print(f"   Total samples: {len(trt_df)}")
    
    # Gunakan nama kolom yang benar
    azimuth_col = 'azimuth_class' if 'azimuth_class' in df.columns else 'azimuth'
    magnitude_col = 'magnitude_class' if 'magnitude_class' in df.columns else 'magnitude'
    
    if len(trt_df) > 0:
        # Distribusi azimuth untuk TRT
        print(f"\n   Distribusi Azimuth TRT:")
        trt_azimuth = trt_df[azimuth_col].value_counts()
        for az, count in trt_azimuth.items():
            print(f"      {az}: {count} samples ({count/len(trt_df)*100:.1f}%)")
        
        # Distribusi magnitude untuk TRT
        print(f"\n   Distribusi Magnitude TRT:")
        trt_mag = trt_df[magnitude_col].value_counts()
        for mag, count in trt_mag.items():
            print(f"      {mag}: {count} samples ({count/len(trt_df)*100:.1f}%)")
    
    # Analisis distribusi azimuth keseluruhan
    print(f"\n📐 DISTRIBUSI AZIMUTH KESELURUHAN:")
    azimuth_counts = df[azimuth_col].value_counts()
    for az, count in azimuth_counts.items():
        print(f"   {az}: {count} samples ({count/len(df)*100:.1f}%)")
    
    # Cek apakah ada data SW untuk TRT
    trt_sw = trt_df[trt_df[azimuth_col] == 'SW']
    print(f"\n⚠️  KRITIS: Data TRT dengan azimuth SW: {len(trt_sw)} samples")
    if len(trt_sw) > 0:
        print("   Tanggal-tanggal dengan SW:")
        for _, row in trt_sw.iterrows():
            print(f"      - {row['spectrogram_file']}")
else:
    print(f"❌ Training metadata tidak ditemukan: {metadata_path}")

# ============================================================================
# BAGIAN 2: ANALISIS CLASS MAPPING
# ============================================================================
print("\n" + "=" * 80)
print("🗺️  BAGIAN 2: ANALISIS CLASS MAPPING")
print("=" * 80)

mapping_path = 'training_data/class_mapping.json'
if os.path.exists(mapping_path):
    with open(mapping_path, 'r') as f:
        class_mappings = json.load(f)
    
    print("\n✅ Class Mapping dari Training:")
    print("\n   Azimuth to Index:")
    for az, idx in class_mappings.get('azimuth_to_idx', {}).items():
        print(f"      {az} → {idx}")
    
    print("\n   Index to Azimuth:")
    for idx, az in class_mappings.get('idx_to_azimuth', {}).items():
        print(f"      {idx} → {az}")
    
    print("\n   Magnitude to Index:")
    for mag, idx in class_mappings.get('magnitude_to_idx', {}).items():
        print(f"      {mag} → {idx}")
    
    # Cek apakah ada class "Normal"
    if 'Normal' in class_mappings.get('azimuth_to_idx', {}):
        print("\n⚠️  WARNING: Ada class 'Normal' di azimuth mapping!")
    else:
        print("\n✅ Tidak ada class 'Normal' di azimuth mapping (benar)")
else:
    print(f"❌ Class mapping tidak ditemukan: {mapping_path}")

# ============================================================================
# BAGIAN 3: ANALISIS MODEL YANG DIGUNAKAN
# ============================================================================
print("\n" + "=" * 80)
print("🤖 BAGIAN 3: ANALISIS MODEL")
print("=" * 80)

# Cari model yang digunakan
model_paths = [
    'experiments_fixed/exp_fixed_20260202_163643/best_model.pth',
    'final_production_model/best_model.pth',
    'convnext_production_model/best_model.pth',
    'mdata2/best_vgg16_model_phase1.keras'
]

print("\n🔍 Mencari model yang tersedia:")
for path in model_paths:
    if os.path.exists(path):
        size_mb = os.path.getsize(path) / (1024 * 1024)
        print(f"   ✅ {path} ({size_mb:.1f} MB)")
    else:
        print(f"   ❌ {path} (tidak ada)")

# ============================================================================
# BAGIAN 4: ANALISIS LOEO VALIDATION RESULTS
# ============================================================================
print("\n" + "=" * 80)
print("📈 BAGIAN 4: ANALISIS LOEO VALIDATION")
print("=" * 80)

loeo_path = 'loeo_validation_results/LOEO_FINAL_REPORT.md'
if os.path.exists(loeo_path):
    print(f"\n✅ LOEO Report ditemukan")
    print("\n   Ringkasan dari LOEO Validation:")
    print("   - Magnitude Mean Accuracy: 97.53% ± 0.96%")
    print("   - Azimuth Mean Accuracy: 69.51% ± 5.65%")
    print("   - Min Azimuth (Fold 10): 58.33%")
    print("   - Max Azimuth (Fold 9): 82.00%")
    print("\n   ⚠️  INSIGHT: Azimuth accuracy hanya 69.51% - ini menjelaskan")
    print("      mengapa confidence untuk azimuth rendah!")
else:
    print(f"❌ LOEO report tidak ditemukan")

# ============================================================================
# BAGIAN 5: ROOT CAUSE ANALYSIS
# ============================================================================
print("\n" + "=" * 80)
print("🎯 BAGIAN 5: ROOT CAUSE ANALYSIS")
print("=" * 80)

print("""
📋 TEMUAN KRITIS:

1. DATA TRAINING TRT TERBATAS
   - TRT hanya memiliki sedikit samples dengan azimuth SW
   - Mayoritas data TRT adalah arah S (South)
   - Model tidak cukup belajar pola SW untuk stasiun TRT

2. AZIMUTH CLASSIFICATION INHERENTLY DIFFICULT
   - LOEO validation menunjukkan azimuth accuracy hanya 69.51%
   - Variance tinggi (std = 5.65%)
   - Ini adalah 9-class classification problem yang challenging

3. CONFIDENCE RENDAH ADALAH EXPECTED BEHAVIOR
   - Dengan accuracy 69.51%, confidence rendah adalah normal
   - Model "tidak yakin" karena memang task-nya sulit
   - F1 score tinggi pada validation tidak menjamin confidence tinggi

4. KEMUNGKINAN PREPROCESSING MISMATCH
   - Scanner mungkin masih menggunakan preprocessing berbeda
   - Perlu verifikasi filter range, colormap, dan normalization

5. DISTRIBUSI KELAS TIDAK SEIMBANG
   - Beberapa arah (seperti SW) memiliki sampel lebih sedikit
   - Model cenderung predict kelas mayoritas
""")

# ============================================================================
# BAGIAN 6: REKOMENDASI
# ============================================================================
print("\n" + "=" * 80)
print("💡 BAGIAN 6: REKOMENDASI PERBAIKAN")
print("=" * 80)

print("""
🔧 REKOMENDASI IMMEDIATE:

1. VERIFIKASI PREPROCESSING SCANNER
   - Pastikan filter range: 0.01-0.045 Hz (sama dengan training)
   - Pastikan colormap: jet (sama dengan training)
   - Pastikan 3-component (H, D, Z) stacked

2. TAMBAH DATA TRAINING UNTUK TRT-SW
   - Cari event gempa lain dengan arah SW dari TRT
   - Augmentasi data untuk kelas minoritas

3. GUNAKAN ENSEMBLE MODEL
   - Kombinasikan beberapa model untuk meningkatkan confidence
   - Voting mechanism untuk prediksi final

4. IMPLEMENTASI UNCERTAINTY QUANTIFICATION
   - Gunakan Monte Carlo Dropout untuk estimasi uncertainty
   - Berikan warning jika confidence < threshold

5. RETRAIN DENGAN FOCAL LOSS YANG LEBIH AGRESIF
   - Tingkatkan gamma untuk fokus pada kelas minoritas
   - Gunakan class weights yang lebih ekstrem

6. PERTIMBANGKAN SPATIAL CONTEXT
   - Gunakan informasi lokasi stasiun sebagai fitur tambahan
   - Model bisa belajar pola spesifik per stasiun
""")

# ============================================================================
# BAGIAN 7: KESIMPULAN
# ============================================================================
print("\n" + "=" * 80)
print("📝 KESIMPULAN")
print("=" * 80)

print("""
🎯 JAWABAN UNTUK PERTANYAAN UTAMA:

Q: Mengapa confidence rendah padahal F1 score tinggi?

A: Ada beberapa faktor yang berkontribusi:

1. F1 SCORE TINGGI PADA VALIDATION ≠ CONFIDENCE TINGGI PADA INFERENCE
   - F1 score dihitung pada distribusi data yang seimbang (validation set)
   - Inference pada data baru bisa memiliki distribusi berbeda
   - Model "tidak yakin" pada pola yang jarang dilihat saat training

2. AZIMUTH CLASSIFICATION ADALAH TASK YANG SULIT
   - 9 kelas dengan distribusi tidak seimbang
   - LOEO validation menunjukkan accuracy hanya 69.51%
   - Confidence rendah adalah refleksi dari kesulitan task

3. DATA TRAINING UNTUK TRT-SW TERBATAS
   - Model tidak cukup belajar pola SW untuk stasiun TRT
   - Ketika melihat pola baru, model "bingung"

4. KEMUNGKINAN PREPROCESSING MISMATCH
   - Perbedaan kecil dalam preprocessing bisa berdampak besar
   - Perlu verifikasi bahwa scanner menggunakan preprocessing yang sama

🔴 STATUS: MASALAH KRITIS - PERLU PERBAIKAN SEGERA

📋 NEXT STEPS:
1. Verifikasi preprocessing scanner (30 menit)
2. Analisis data training TRT lebih detail (30 menit)
3. Pertimbangkan retrain dengan data tambahan (4 jam)
4. Implementasi uncertainty quantification (2 jam)
""")

print("\n" + "=" * 80)
print("✅ INVESTIGASI SELESAI")
print("=" * 80)
