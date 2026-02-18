"""
Investigate Z/H Ratio for Hybrid Classification
Menganalisis distribusi statistik rasio Z/H (Polarisasi) pada data Normal vs Prekursor
untuk menentukan threshold optimal sebagai 'Physics Gate'.

Metodologi:
1. Load sampel acak dari dataset_consolidation (jika ada, atau dataset_unified sementara)
2. Hitung rasio Z/H dari spektrogram atau data mentah (jika mungkin)
   - Karena kita hanya punya spektrogram gambar (PNG), kita akan mencoba mengestimasi
     intensitas rata-rata dari channel Z dan H.
   - Channel Spectrogram biasanya: R=H, G=D, B=Z (perlu verifikasi generate_dataset_from_scan.py)
3. Plot distribusi Z/H untuk kelas Normal vs Large/Medium.
4. Cari Threshold optimal yang meminimalkan False Negative (Prekursor terbuang).
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm
import seaborn as sns

# Konfigurasi
INPUT_DIR = 'dataset_unified'  # Gunakan data yang sudah ada dulu
METADATA_FILE = os.path.join(INPUT_DIR, 'metadata', 'unified_metadata.csv')
OUTPUT_DIR = 'investigation_zh'

def get_zh_ratio_from_image(img_path):
    """
    Estimasi rasio Z/H dari gambar spektrogram 3-channel.
    Asumsi umum dari script generate:
    - Channel 0 (Red)   : Komponen H
    - Channel 1 (Green) : Komponen D
    - Channel 2 (Blue)  : Komponen Z
    """
    try:
        with Image.open(img_path) as img:
            arr = np.array(img.convert('RGB'))
            
            # Intensitas rata-rata per channel
            h_intensity = np.mean(arr[:,:,0])
            # d_intensity = np.mean(arr[:,:,1])
            z_intensity = np.mean(arr[:,:,2])
            
            # Hindari pembagian nol
            if h_intensity < 1e-6:
                return 0.0
            
            # Rasio Z/H sederhana
            return z_intensity / h_intensity
    except Exception:
        return None

def main():
    print("="*50)
    print("INVESTIGASI Z/H RATIO")
    print("="*50)
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # 1. Load Metadata
    if not os.path.exists(METADATA_FILE):
        print(f"Metadata not found: {METADATA_FILE}")
        return
        
    df = pd.read_csv(METADATA_FILE)
    print(f"Loaded {len(df)} samples")
    
    # Filter sampel (ambil subset agar cepat)
    # Ambil semua Large/Major, dan sampel Normal secukupnya
    df_precursor = df[df['magnitude_class'].isin(['Large', 'Major', 'Medium', 'Moderate'])]
    df_normal_all = df[df['magnitude_class'] == 'Normal']
    # Sample Normal - ambil min antara 1000 atau jumlah yang ada
    sample_size = min(1000, len(df_normal_all))
    df_normal = df_normal_all.sample(n=sample_size, random_state=42) if sample_size > 0 else df_normal_all
    
    df_sample = pd.concat([df_precursor, df_normal])
    print(f"Analyzing {len(df_sample)} samples...")
    print(f"- Precursors: {len(df_precursor)}")
    print(f"- Normal: {len(df_normal)}")
    
    # 2. Hitung Ratio
    ratios = []
    valid_indices = []
    
    for idx, row in tqdm(df_sample.iterrows(), total=len(df_sample)):
        # Construct path (sesuaikan dengan struktur unified)
        # Coba find file
        path = None
        if 'unified_path' in row and pd.notna(row['unified_path']):
             p = os.path.join(INPUT_DIR, row['unified_path'])
             if os.path.exists(p): path = p
        
        if not path:
             # Coba cari manual
             fname = f"{row['station']}_{str(row['date']).replace('-','')}_H{int(row['hour']):02d}_M{row['magnitude']}_AZ{row['azimuth']}.png"
             p = os.path.join(INPUT_DIR, 'spectrograms', fname)
             if os.path.exists(p): path = p

        if path:
            ratio = get_zh_ratio_from_image(path)
            if ratio is not None:
                ratios.append(ratio)
                valid_indices.append(idx)
    
    # Update DataFrame
    df_analyzed = df_sample.loc[valid_indices].copy()
    df_analyzed['zh_ratio'] = ratios
    
    # 3. Analisis Statistik
    print("\nStatistik Z/H Ratio:")
    stats = df_analyzed.groupby('magnitude_class')['zh_ratio'].describe()
    print(stats)
    stats.to_csv(os.path.join(OUTPUT_DIR, 'zh_stats.csv'))
    
    # 4. Visualisasi
    plt.figure(figsize=(10, 6))
    sns.boxplot(x='magnitude_class', y='zh_ratio', data=df_analyzed, order=['Normal', 'Moderate', 'Medium', 'Large'])
    plt.title('Distribusi Z/H Ratio per Kelas Magnitudo')
    plt.ylabel('Z/H Intensity Ratio (from Spectrogram)')
    plt.axhline(y=1.0, color='r', linestyle='--', alpha=0.5, label='Ratio 1.0')
    plt.legend()
    plt.savefig(os.path.join(OUTPUT_DIR, 'zh_boxplot.png'))
    plt.close()
    
    plt.figure(figsize=(10, 6))
    sns.kdeplot(data=df_analyzed, x='zh_ratio', hue='magnitude_class', common_norm=False)
    plt.title('Density Plot Z/H Ratio')
    plt.xlim(0, 3) # Batasi biar kelihatan
    plt.savefig(os.path.join(OUTPUT_DIR, 'zh_density.png'))
    plt.close()
    
    # 5. Rekomendasi Threshold
    # Cari percentile ke-5 dari kelas Large/Medium (agar 95% Prekursor lolos)
    precursor_ratios = df_analyzed[df_analyzed['magnitude_class'].isin(['Large', 'Medium'])]['zh_ratio']
    
    if not precursor_ratios.empty:
        threshold_05 = np.percentile(precursor_ratios, 5)
        threshold_10 = np.percentile(precursor_ratios, 10)
        
        print("\nREKOMENDASI THRESHOLD (Physics Gate):")
        print(f"Agar 95% Prekursor Lolos (5% False Negative): Threshold Z/H > {threshold_05:.3f}")
        print(f"Agar 90% Prekursor Lolos (10% False Negative): Threshold Z/H > {threshold_10:.3f}")
        
        # Hitung berapa % Normal yang terbuang (True Negative rate dari Gatekeeper)
        normal_ratios = df_analyzed[df_analyzed['magnitude_class'] == 'Normal']['zh_ratio']
        filtered_normal = np.sum(normal_ratios < threshold_05)
        print(f"Efisiensi Filter: {filtered_normal}/{len(normal_ratios)} ({filtered_normal/len(normal_ratios):.1%}) sampel Normal akan otomatis terbuang.")
    
    # Simpan data mentah
    df_analyzed.to_csv(os.path.join(OUTPUT_DIR, 'zh_analysis_raw.csv'), index=False)
    print(f"\nHasil analisis disimpan di: {OUTPUT_DIR}")

if __name__ == "__main__":
    main()
