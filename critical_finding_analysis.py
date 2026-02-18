#!/usr/bin/env python3
"""
🚨 CRITICAL FINDING: ROOT CAUSE IDENTIFIED!

MASALAH UTAMA DITEMUKAN:
1. Class 'Large' hanya 32 samples (2.9%) - SANGAT TIDAK SEIMBANG!
2. Model ditraining dengan 4 magnitude classes (Normal, Moderate, Medium, Large)
3. Tapi scanner menggunakan mapping BERBEDA (3 classes tanpa Normal)!

Ini adalah MISMATCH KRITIS yang menyebabkan confidence rendah!
"""

import os
import sys
import json
import torch
import numpy as np
from pathlib import Path

print("=" * 80)
print("🚨 CRITICAL FINDING: ROOT CAUSE ANALYSIS")
print("=" * 80)

# ============================================================================
# MASALAH 1: CLASS MAPPING MISMATCH
# ============================================================================
print("\n" + "=" * 80)
print("🔴 MASALAH 1: CLASS MAPPING MISMATCH")
print("=" * 80)

print("""
📋 CLASS MAPPING DI MODEL TRAINING:
   experiments_fixed/exp_fixed_20260202_163643/class_mappings.json

   Magnitude (4 classes):
   - 0: Normal
   - 1: Moderate  
   - 2: Medium
   - 3: Large     ← Large adalah class 3!

   Azimuth (9 classes):
   - 0: Normal
   - 1: N
   - 2: NE
   - 3: E
   - 4: SE
   - 5: S
   - 6: SW
   - 7: W
   - 8: NW

📋 CLASS MAPPING DI SCANNER (prekursor_scanner.py):
   training_data/class_mapping.json

   Magnitude (3 classes - TANPA NORMAL!):
   - 0: Large     ← Large adalah class 0!
   - 1: Medium
   - 2: Moderate

   Azimuth (8 classes - TANPA NORMAL!):
   - 0: E
   - 1: N
   - 2: NE
   - 3: NW
   - 4: S
   - 5: SE
   - 6: SW
   - 7: W

🚨 MISMATCH KRITIS:
   - Model output: Large = class 3
   - Scanner expects: Large = class 0
   
   Ketika model predict class 3 (Large), scanner membacanya sebagai class 3
   tapi mapping scanner mengatakan class 3 = NW (untuk azimuth) atau tidak ada (untuk magnitude)!
""")

# ============================================================================
# MASALAH 2: CLASS IMBALANCE EKSTREM
# ============================================================================
print("\n" + "=" * 80)
print("🔴 MASALAH 2: CLASS IMBALANCE EKSTREM")
print("=" * 80)

print("""
📊 DISTRIBUSI MAGNITUDE DI TRAINING:
   Medium:   1060 samples (95.3%)
   Large:      32 samples (2.9%)
   Moderate:   20 samples (1.8%)
   Normal:      ? samples (dari dataset_unified)

⚠️  MASALAH:
   - Large hanya 2.9% dari total!
   - Ratio Medium:Large = 33:1
   - Model akan BIAS ke Medium karena dominan
   - Confidence untuk Large akan SELALU rendah!

📈 EXPECTED BEHAVIOR:
   - Jika model predict Large, confidence ~30-40% adalah NORMAL
   - Karena model "tidak yakin" dengan kelas minoritas
   - Ini bukan bug, tapi KONSEKUENSI dari class imbalance!
""")

# ============================================================================
# MASALAH 3: MODEL ARCHITECTURE MISMATCH
# ============================================================================
print("\n" + "=" * 80)
print("🔴 MASALAH 3: MODEL ARCHITECTURE MISMATCH")
print("=" * 80)

print("""
📋 MODEL DI experiments_fixed:
   - magnitude_head: 4 classes (Normal, Moderate, Medium, Large)
   - azimuth_head: 9 classes (Normal, N, NE, E, SE, S, SW, W, NW)

📋 SCANNER EXPECTS:
   - magnitude: 3 classes (Large, Medium, Moderate)
   - azimuth: 8 classes (E, N, NE, NW, S, SE, SW, W)

🚨 MISMATCH:
   - Model punya class "Normal" untuk magnitude dan azimuth
   - Scanner TIDAK punya class "Normal"
   - Index mapping BERBEDA!
""")

# ============================================================================
# VERIFIKASI: Load kedua mapping
# ============================================================================
print("\n" + "=" * 80)
print("🔍 VERIFIKASI: COMPARE MAPPINGS")
print("=" * 80)

# Load model training mapping
model_mapping_path = 'experiments_fixed/exp_fixed_20260202_163643/class_mappings.json'
if os.path.exists(model_mapping_path):
    with open(model_mapping_path, 'r') as f:
        model_mapping = json.load(f)
    print(f"\n✅ Model Training Mapping:")
    print(f"   Magnitude: {model_mapping.get('magnitude_classes', [])}")
    print(f"   Azimuth: {model_mapping.get('azimuth_classes', [])}")

# Load scanner mapping
scanner_mapping_path = 'training_data/class_mapping.json'
if os.path.exists(scanner_mapping_path):
    with open(scanner_mapping_path, 'r') as f:
        scanner_mapping = json.load(f)
    print(f"\n✅ Scanner Mapping:")
    print(f"   idx_to_magnitude: {scanner_mapping.get('idx_to_magnitude', {})}")
    print(f"   idx_to_azimuth: {scanner_mapping.get('idx_to_azimuth', {})}")

# ============================================================================
# SOLUSI
# ============================================================================
print("\n" + "=" * 80)
print("💡 SOLUSI YANG DIPERLUKAN")
print("=" * 80)

print("""
🔧 SOLUSI IMMEDIATE:

1. UPDATE SCANNER MAPPING
   Scanner harus menggunakan mapping yang SAMA dengan model training:
   
   Magnitude:
   - 0: Normal
   - 1: Moderate
   - 2: Medium
   - 3: Large
   
   Azimuth:
   - 0: Normal
   - 1: N
   - 2: NE
   - 3: E
   - 4: SE
   - 5: S
   - 6: SW
   - 7: W
   - 8: NW

2. ATAU GUNAKAN MODEL YANG SESUAI
   Jika scanner expects 3 magnitude classes, gunakan model yang ditraining
   dengan 3 classes (tanpa Normal).

3. FIX CLASS IMBALANCE
   - Retrain dengan oversampling Large
   - Gunakan focal loss dengan gamma=3 atau lebih
   - Gunakan class weights yang lebih agresif
""")

# ============================================================================
# TEST: Simulasi dengan mapping yang benar
# ============================================================================
print("\n" + "=" * 80)
print("🧪 TEST: SIMULASI DENGAN MAPPING BENAR")
print("=" * 80)

# Simulasi: jika model predict class 3 dengan 29% confidence
# Dengan mapping model training: class 3 = Large
# Dengan mapping scanner: class 3 = ??? (tidak ada untuk magnitude 3 classes)

print("""
📊 SIMULASI:

Jika model output:
   magnitude_probs = [0.10, 0.15, 0.46, 0.29]
                      ^      ^      ^      ^
                      |      |      |      |
                   Normal Moderate Medium Large

   argmax = 2 (Medium, 46%)
   
   TAPI jika scanner mapping salah:
   - Scanner thinks class 0 = Large
   - Scanner reads: Large = 10% (SALAH!)
   
   SEHARUSNYA:
   - Model says class 3 = Large = 29%
   - Ini adalah confidence yang BENAR untuk Large

🎯 KESIMPULAN:
   Confidence 29% untuk Large MUNGKIN sudah benar!
   Masalahnya adalah MAPPING yang tidak sesuai!
""")

print("\n" + "=" * 80)
print("✅ ANALISIS SELESAI")
print("=" * 80)
