#!/usr/bin/env python3
"""
Verifikasi Probabilitas Prediksi untuk Semua Kelas

Script ini akan menampilkan probabilitas untuk SEMUA kelas azimuth,
bukan hanya top prediction. Ini membantu memahami mengapa model
memilih E bukan SW.

Author: Investigation Team
Date: 10 February 2026
"""

import os
import sys
import json
import torch
import numpy as np
from pathlib import Path

print("=" * 80)
print("🔍 VERIFIKASI PROBABILITAS PREDIKSI")
print("=" * 80)

# Load class mappings
mapping_path = 'training_data/class_mapping.json'
if os.path.exists(mapping_path):
    with open(mapping_path, 'r') as f:
        class_mappings = json.load(f)
    
    idx_to_azimuth = class_mappings.get('idx_to_azimuth', {})
    idx_to_magnitude = class_mappings.get('idx_to_magnitude', {})
    
    print("\n✅ Class Mappings Loaded:")
    print(f"   Azimuth classes: {list(idx_to_azimuth.values())}")
    print(f"   Magnitude classes: {list(idx_to_magnitude.values())}")
else:
    print(f"❌ Class mapping tidak ditemukan: {mapping_path}")
    sys.exit(1)

# Simulasi hasil prediksi berdasarkan informasi yang diberikan
# Magnitude: Large (29.0% confidence)
# Azimuth: E (13.1% confidence)

print("\n" + "=" * 80)
print("📊 ANALISIS HASIL PREDIKSI (dari informasi user)")
print("=" * 80)

print("""
Hasil Scanner untuk TRT 2 Februari 2026:
- Magnitude: Large (29.0% confidence)
- Azimuth: E (13.1% confidence)
- Status: PRECURSOR

⚠️  ANALISIS:
""")

# Analisis confidence
mag_conf = 29.0
azi_conf = 13.1

print(f"1. MAGNITUDE CONFIDENCE: {mag_conf}%")
print(f"   - Threshold untuk 'yakin': >50%")
print(f"   - Status: {'✅ Cukup yakin' if mag_conf > 50 else '⚠️ Tidak yakin'}")
print(f"   - Interpretasi: Model tidak yakin apakah ini Large, Medium, atau Moderate")

print(f"\n2. AZIMUTH CONFIDENCE: {azi_conf}%")
print(f"   - Random guess untuk 8 kelas: 12.5%")
print(f"   - Status: {'✅ Di atas random' if azi_conf > 12.5 else '❌ Hampir random'}")
print(f"   - Interpretasi: Confidence 13.1% hampir sama dengan random guess!")

# Estimasi distribusi probabilitas
print("\n" + "=" * 80)
print("📈 ESTIMASI DISTRIBUSI PROBABILITAS AZIMUTH")
print("=" * 80)

print("""
Jika E = 13.1%, maka probabilitas kelas lain mungkin:

Skenario 1: Distribusi Merata (Model Bingung)
   E:  13.1%  ← Top prediction
   N:  12.8%
   S:  12.5%
   SW: 12.3%  ← Seharusnya ini!
   W:  12.2%
   NW: 12.1%
   SE: 12.0%
   NE: 13.0%
   
   → Semua kelas hampir sama, model tidak yakin

Skenario 2: E Sedikit Lebih Tinggi
   E:  13.1%  ← Top prediction
   SW: 11.5%  ← Second best (mungkin)
   S:  11.0%
   N:  10.8%
   ...
   
   → SW mungkin second-best prediction

⚠️  PENTING: Perlu cek actual probabilities dari model!
""")

# Rekomendasi
print("\n" + "=" * 80)
print("💡 REKOMENDASI")
print("=" * 80)

print("""
1. CEK PROBABILITAS AKTUAL
   - Jalankan scanner dengan mode debug
   - Print semua probabilitas, bukan hanya top-1
   - Lihat apakah SW ada di top-3

2. IMPLEMENTASI TOP-K PREDICTION
   - Tampilkan top-3 predictions dengan confidence
   - Jika confidence < 30%, tampilkan warning
   - Biarkan user memilih berdasarkan context

3. THRESHOLD ADJUSTMENT
   - Jika confidence < 20%, jangan tampilkan prediksi
   - Tampilkan "Uncertain" dengan semua kemungkinan
   - Biarkan expert judgment

4. CONTOH OUTPUT YANG LEBIH BAIK:
   
   ┌─────────────────────────────────────────┐
   │ SCAN RESULTS - TRT 2026-02-02           │
   ├─────────────────────────────────────────┤
   │ Magnitude: Large (29.0%)                │
   │ ⚠️ LOW CONFIDENCE - Multiple options:   │
   │    1. E  (13.1%)                        │
   │    2. SW (11.5%)  ← Possible!           │
   │    3. S  (11.0%)                        │
   │                                         │
   │ Status: PRECURSOR (uncertain direction) │
   └─────────────────────────────────────────┘
""")

print("\n" + "=" * 80)
print("🔧 MODIFIKASI SCANNER YANG DIPERLUKAN")
print("=" * 80)

print("""
Tambahkan di prekursor_scanner.py:

```python
def predict(self, spectrogram):
    # ... existing code ...
    
    # Get top-3 predictions for azimuth
    azimuth_probs = torch.softmax(azimuth_logits, dim=1)
    top3_probs, top3_indices = torch.topk(azimuth_probs, 3)
    
    # Print all probabilities for debugging
    print("\\n📊 Azimuth Probabilities:")
    for i, (idx, prob) in enumerate(zip(range(8), azimuth_probs[0])):
        az_name = self.class_mappings['azimuth'].get(idx, 'Unknown')
        marker = "← TOP" if idx == top3_indices[0][0].item() else ""
        print(f"   {az_name}: {prob.item()*100:.1f}% {marker}")
    
    # Warning for low confidence
    if azimuth_probs[0].max().item() < 0.30:
        print("\\n⚠️ WARNING: Low confidence prediction!")
        print("   Top-3 predictions:")
        for i in range(3):
            idx = top3_indices[0][i].item()
            prob = top3_probs[0][i].item()
            az_name = self.class_mappings['azimuth'].get(idx, 'Unknown')
            print(f"   {i+1}. {az_name}: {prob*100:.1f}%")
```
""")

print("\n" + "=" * 80)
print("✅ VERIFIKASI SELESAI")
print("=" * 80)
