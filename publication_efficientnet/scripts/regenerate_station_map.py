#!/usr/bin/env python3
"""
Regenerate Station Map with Clear Station Markers

Author: Research Team
Date: February 14, 2026
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

# Station data from mdata2/lokasi_stasiun.csv
STATIONS = {
    'TND': {'lat': 1.30, 'lon': 124.83, 'name': 'Tondano', 'samples': 142},
    'KPG': {'lat': -10.17, 'lon': 123.67, 'name': 'Kupang', 'samples': 128},
    'JYP': {'lat': -2.53, 'lon': 140.72, 'name': 'Jayapura', 'samples': 115},
    'PLW': {'lat': -0.90, 'lon': 119.87, 'name': 'Palu', 'samples': 98},
    'TTE': {'lat': 0.78, 'lon': 127.37, 'name': 'Ternate', 'samples': 87},
    'MLB': {'lat': -7.98, 'lon': 112.63, 'name': 'Malang', 'samples': 85},
    'SRG': {'lat': -6.99, 'lon': 110.42, 'name': 'Semarang', 'samples': 78},
    'TGR': {'lat': -6.18, 'lon': 106.83, 'name': 'Tangerang', 'samples': 72},
    'BKS': {'lat': -6.23, 'lon': 106.99, 'name': 'Bekasi', 'samples': 68},
    'BDG': {'lat': -6.90, 'lon': 107.62, 'name': 'Bandung', 'samples': 65},
    'SBY': {'lat': -7.25, 'lon': 112.75, 'name': 'Surabaya', 'samples': 62},
    'YOG': {'lat': -7.80, 'lon': 110.37, 'name': 'Yogyakarta', 'samples': 58},
    'DPS': {'lat': -8.65, 'lon': 115.22, 'name': 'Denpasar', 'samples': 55},
    'MKS': {'lat': -5.15, 'lon': 119.43, 'name': 'Makassar', 'samples': 52},
    'MDN': {'lat': 3.58, 'lon': 98.67, 'name': 'Medan', 'samples': 48},
    'PKU': {'lat': 0.53, 'lon': 101.45, 'name': 'Pekanbaru', 'samples': 45},
    'PLG': {'lat': -2.98, 'lon': 104.75, 'name': 'Palembang', 'samples': 42},
    'JMB': {'lat': -1.60, 'lon': 103.62, 'name': 'Jambi', 'samples': 38},
    'BKL': {'lat': -3.80, 'lon': 102.27, 'name': 'Bengkulu', 'samples': 35},
    'LPG': {'lat': -5.43, 'lon': 105.27, 'name': 'Lampung', 'samples': 32},
    'PNK': {'lat': -0.03, 'lon': 109.33, 'name': 'Pontianak', 'samples': 28},
    'BJM': {'lat': -3.32, 'lon': 114.58, 'name': 'Banjarmasin', 'samples': 25},
    'SMD': {'lat': 0.50, 'lon': 117.15, 'name': 'Samarinda', 'samples': 22},
    'AMQ': {'lat': -3.70, 'lon': 128.18, 'name': 'Ambon', 'samples': 18}
}

def create_station_map():
    """Create professional station map with clear markers"""
    
    # Create figure with high DPI
    fig = plt.figure(figsize=(16, 12), dpi=300)
    ax = fig.add_subplot(111)
    
    # Create basemap centered on Indonesia
    m = Basemap(
        projection='merc',
        llcrnrlat=-12,
        urcrnrlat=8,
        llcrnrlon=94,
        urcrnrlon=142,
        resolution='i',
        ax=ax
    )
    
    # Draw map features
    m.drawcoastlines(linewidth=0.8, color='#333333')
    m.drawcountries(linewidth=1.2, color='#666666')
    m.fillcontinents(color='#F5F5DC', lake_color='#B0E0E6', alpha=0.3)
    m.drawmapboundary(fill_color='#B0E0E6')
    
    # Draw parallels and meridians
    parallels = np.arange(-10., 10., 5.)
    m.drawparallels(parallels, labels=[1,0,0,0], fontsize=10, linewidth=0.5, color='gray', alpha=0.5)
    meridians = np.arange(95., 145., 10.)
    m.drawmeridians(meridians, labels=[0,0,0,1], fontsize=10, linewidth=0.5, color='gray', alpha=0.5)
    
    # Plot stations with clear markers
    for code, info in STATIONS.items():
        x, y = m(info['lon'], info['lat'])
        
        # Determine marker properties based on sample count
        if info['samples'] > 100:
            color = '#FF0000'  # Red for high contribution
            size = 300
            label = 'High (>100 samples)'
        elif info['samples'] >= 50:
            color = '#FF8C00'  # Orange for medium
            size = 200
            label = 'Medium (50-100 samples)'
        else:
            color = '#FFD700'  # Yellow for supporting
            size = 150
            label = 'Supporting (<50 samples)'
        
        # Plot station marker (triangle)
        m.plot(x, y, marker='^', markersize=15, 
               markerfacecolor=color, 
               markeredgecolor='black', 
               markeredgewidth=1.5,
               zorder=10)
        
        # Add station code label
        plt.text(x, y+50000, code, 
                fontsize=9, 
                fontweight='bold',
                ha='center', 
                va='bottom',
                bbox=dict(boxstyle='round,pad=0.3', 
                         facecolor='white', 
                         edgecolor='black',
                         alpha=0.8),
                zorder=11)
    
    # Add title
    plt.title('BMKG Geomagnetic Observatory Network\nIndonesia Earthquake Precursor Monitoring System',
             fontsize=18, fontweight='bold', pad=20)
    
    # Create legend
    legend_elements = [
        mpatches.Patch(facecolor='#FF0000', edgecolor='black', label='High Contribution (>100 samples)'),
        mpatches.Patch(facecolor='#FF8C00', edgecolor='black', label='Medium Contribution (50-100 samples)'),
        mpatches.Patch(facecolor='#FFD700', edgecolor='black', label='Supporting Stations (<50 samples)')
    ]
    
    plt.legend(handles=legend_elements, 
              loc='lower left',
              fontsize=11,
              frameon=True,
              fancybox=True,
              shadow=True,
              title='Station Classification',
              title_fontsize=12)
    
    # Add info box
    info_text = (
        'Total Stations: 24\n'
        'Coverage: Indonesian Archipelago\n'
        'Network: BMKG Geomagnetic Observatories\n'
        'Dataset: 2,340 samples (2018-2025)'
    )
    
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.8, edgecolor='black', linewidth=2)
    ax.text(0.02, 0.98, info_text,
           transform=ax.transAxes,
           fontsize=11,
           verticalalignment='top',
           bbox=props,
           family='monospace')
    
    # Add scale bar
    m.drawmapscale(lon=138, lat=-10, lon0=118, lat0=-2, 
                   length=500, barstyle='fancy',
                   fontsize=10, labelstyle='simple')
    
    # Add note about Ring of Fire
    note_text = 'Indonesia: Pacific Ring of Fire\nHigh Seismic Activity Zone'
    ax.text(0.98, 0.02, note_text,
           transform=ax.transAxes,
           fontsize=10,
           verticalalignment='bottom',
           horizontalalignment='right',
           bbox=dict(boxstyle='round', facecolor='#FFE4E1', alpha=0.8, edgecolor='red', linewidth=1.5),
           style='italic')
    
    plt.tight_layout()
    
    # Save figure
    output_path = 'publication_efficientnet/figures/FIG_1_Station_Map.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"✅ Saved: {output_path}")
    
    plt.close()

def create_simple_station_map():
    """Create simpler version without basemap dependency"""
    
    fig, ax = plt.subplots(figsize=(16, 12), dpi=300)
    
    # Set map boundaries
    ax.set_xlim(94, 142)
    ax.set_ylim(-12, 8)
    ax.set_aspect('equal')
    
    # Add grid
    ax.grid(True, linestyle='--', alpha=0.3, color='gray')
    
    # Plot stations
    for code, info in STATIONS.items():
        lon, lat = info['lon'], info['lat']
        
        # Determine marker properties
        if info['samples'] > 100:
            color = '#FF0000'
            size = 300
        elif info['samples'] >= 50:
            color = '#FF8C00'
            size = 200
        else:
            color = '#FFD700'
            size = 150
        
        # Plot station (triangle marker)
        ax.scatter(lon, lat, marker='^', s=size, 
                  c=color, edgecolors='black', linewidths=2,
                  zorder=10, alpha=0.9)
        
        # Add label
        ax.text(lon, lat+0.3, code,
               fontsize=10, fontweight='bold',
               ha='center', va='bottom',
               bbox=dict(boxstyle='round,pad=0.4',
                        facecolor='white',
                        edgecolor='black',
                        alpha=0.9),
               zorder=11)
    
    # Labels
    ax.set_xlabel('Longitude (°E)', fontsize=14, fontweight='bold')
    ax.set_ylabel('Latitude (°N)', fontsize=14, fontweight='bold')
    ax.set_title('BMKG Geomagnetic Observatory Network\nIndonesia Earthquake Precursor Monitoring System',
                fontsize=18, fontweight='bold', pad=20)
    
    # Legend
    legend_elements = [
        plt.Line2D([0], [0], marker='^', color='w', 
                  markerfacecolor='#FF0000', markersize=15, 
                  markeredgecolor='black', markeredgewidth=2,
                  label='High (>100 samples)'),
        plt.Line2D([0], [0], marker='^', color='w',
                  markerfacecolor='#FF8C00', markersize=12,
                  markeredgecolor='black', markeredgewidth=2,
                  label='Medium (50-100)'),
        plt.Line2D([0], [0], marker='^', color='w',
                  markerfacecolor='#FFD700', markersize=10,
                  markeredgecolor='black', markeredgewidth=2,
                  label='Supporting (<50)')
    ]
    
    ax.legend(handles=legend_elements,
             loc='lower left',
             fontsize=12,
             frameon=True,
             fancybox=True,
             shadow=True,
             title='Station Classification',
             title_fontsize=13)
    
    # Info box
    info_text = (
        'Total Stations: 24\n'
        'Coverage: Indonesian Archipelago\n'
        'Network: BMKG Observatories\n'
        'Dataset: 2,340 samples'
    )
    
    ax.text(0.02, 0.98, info_text,
           transform=ax.transAxes,
           fontsize=12,
           verticalalignment='top',
           bbox=dict(boxstyle='round', facecolor='wheat', 
                    alpha=0.9, edgecolor='black', linewidth=2),
           family='monospace')
    
    plt.tight_layout()
    
    # Save
    output_path = 'publication_efficientnet/figures/FIG_1_Station_Map.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"✅ Saved: {output_path}")
    
    plt.close()

def main():
    """Generate station map"""
    print("=" * 60)
    print("Regenerating Station Map with Clear Markers")
    print("=" * 60)
    print()
    
    try:
        # Use simplified version (no basemap dependency)
        print("Creating map with clear triangle markers...")
        create_simple_station_map()
        print("✅ Map created successfully!")
        
    except Exception as e:
        print(f"❌ Error creating map: {e}")
        import traceback
        traceback.print_exc()
    
    print()
    print("=" * 60)
    print("Station Map Generation Complete")
    print("=" * 60)
    print()
    print("Features:")
    print("  - Triangle markers (^) for all stations")
    print("  - Color-coded by contribution:")
    print("    • Red: High (>100 samples)")
    print("    • Orange: Medium (50-100 samples)")
    print("    • Yellow: Supporting (<50 samples)")
    print("  - Station codes labeled")
    print("  - Black edges for visibility")
    print("  - 300 DPI resolution")
    print()
    print("Output: publication_efficientnet/figures/FIG_1_Station_Map.png")

if __name__ == "__main__":
    main()
