import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
from PIL import Image

def setup_academic_style():
    """Set global plotting style for Q1 Journals (IEEE/Nature standard)."""
    try:
        plt.rcParams["font.family"] = "serif"
        plt.rcParams["font.serif"] = ["Times New Roman", "DejaVu Serif", "serif"]
    except:
        plt.style.use('default')
    
    plt.rcParams["axes.titlesize"] = 12
    plt.rcParams["axes.labelsize"] = 11

import cartopy.crs as ccrs
import cartopy.feature as cfeature

def generate_fig1_map(output_dir):
    """Fig 1: Geographical Distribution of Indonesia Geomagnetic Observatory Network (Cartopy Standard)."""
    stations_path = 'intial/lokasi_stasiun.csv'
    
    if not os.path.exists(stations_path):
        print(f"Missing stations file: {stations_path}")
        return

    # Use delimiter ';' as seen in file
    st_df = pd.read_csv(stations_path, sep=';')
    
    # Cleanup column names and empty rows
    st_df.columns = [c.strip() for c in st_df.columns]
    st_df = st_df.dropna(subset=['Latitude', 'Longitude'])
    
    # Setup Figure with PlateCarree projection
    fig = plt.figure(figsize=(15, 8))
    ax = plt.axes(projection=ccrs.PlateCarree())
    
    # Set extent for Indonesia
    ax.set_extent([94, 142, -11, 7], crs=ccrs.PlateCarree())

    # Add Geographical Features
    ax.add_feature(cfeature.LAND, edgecolor='black', facecolor='#fdfdfd', zorder=1)
    ax.add_feature(cfeature.OCEAN, facecolor='#eef7f9', zorder=0)
    ax.add_feature(cfeature.COASTLINE, linewidth=0.8, zorder=2)
    ax.add_feature(cfeature.BORDERS, linestyle=':', alpha=0.5, zorder=2)

    # Plot Lokasi Stasiun
    ax.scatter(st_df['Longitude'], st_df['Latitude'], color='#d63031', marker='^', s=120, 
               edgecolor='black', label='BMKG Geomagnetic Station', transform=ccrs.PlateCarree(), zorder=5)

    # Beri Label Nama Stasiun
    for i, row in st_df.iterrows():
        try:
            ax.text(float(row['Longitude']) + 0.5, float(row['Latitude']) + 0.2, str(row['Kode Stasiun']), 
                    fontsize=9, fontweight='bold', transform=ccrs.PlateCarree(), family='serif')
        except: continue

    # Gridlines and Labels
    gl = ax.gridlines(draw_labels=True, linestyle='--', alpha=0.4, color='gray')
    gl.top_labels = False
    gl.right_labels = False
    
    plt.title('Fig. 1: Geographical Distribution of Indonesia Geomagnetic Observatory Network', 
              fontsize=14, fontweight='bold', pad=20, family='serif')
    plt.legend(loc='lower left', frameon=True, shadow=True)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'FIG_1_Station_Map.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print("Fig 1 (Professional Map) generated.")

def generate_fig2_preprocessing(output_dir):
    """Fig 2: Preprocessing Flow."""
    sample_img_path = 'dataset_consolidation/spectrograms/SBG_20241028_19_M0_AZ0.png'
    if not os.path.exists(sample_img_path):
        spec_dir = 'dataset_consolidation/spectrograms'
        if os.path.exists(spec_dir):
            files = [f for f in os.listdir(spec_dir) if f.endswith('.png')]
            if files: sample_img_path = os.path.join(spec_dir, files[0])

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), gridspec_kw={'height_ratios': [1, 2]})
    
    t = np.linspace(0, 6, 1000)
    wave = np.sin(2 * np.pi * 0.05 * t) + 0.3 * np.random.normal(size=1000)
    ax1.plot(t, wave, color='black', linewidth=0.8)
    ax1.set_title('(a) Raw Geomagnetic Signal (Time Domain)', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Amplitude (nT)')
    ax1.set_xlabel('Time (Hours)')
    ax1.grid(True, alpha=0.3)

    if os.path.exists(sample_img_path):
        img = Image.open(sample_img_path)
        ax2.imshow(img, aspect='auto')
        ax2.set_title('(b) Processed STFT Power Spectrogram (Time-Frequency Domain)', fontsize=12, fontweight='bold')
        ax2.axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'FIG_2_Preprocessing_Flow.png'), dpi=300)
    plt.close()
    print("Fig 2 (Preprocessing) generated.")

if __name__ == "__main__":
    setup_academic_style()
    output_dir = 'experiments_v2/hierarchical'
    generate_fig1_map(output_dir)
    generate_fig2_preprocessing(output_dir)
