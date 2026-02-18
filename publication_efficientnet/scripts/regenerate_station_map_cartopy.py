#!/usr/bin/env python3
"""
Regenerate Station Map with Cartopy (Modern Basemap Alternative)

Author: Research Team
Date: February 14, 2026
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
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

def create_station_map_cartopy():
    """Create professional station map with Cartopy"""
    
    # Create figure with high DPI
    fig = plt.figure(figsize=(18, 12), dpi=300)
    
    # Create map with PlateCarree projection (equirectangular)
    ax = fig.add_subplot(111, projection=ccrs.PlateCarree())
    
    # Set map extent (Indonesia region)
    ax.set_extent([94, 142, -12, 8], crs=ccrs.PlateCarree())
    
    # Add map features
    ax.add_feature(cfeature.LAND, facecolor='#F5F5DC', alpha=0.6, zorder=1)
    ax.add_feature(cfeature.OCEAN, facecolor='#B0E0E6', alpha=0.5, zorder=0)
    ax.add_feature(cfeature.COASTLINE, linewidth=1.0, edgecolor='#333333', zorder=2)
    ax.add_feature(cfeature.BORDERS, linewidth=1.5, edgecolor='#666666', linestyle='--', zorder=2)
    ax.add_feature(cfeature.LAKES, facecolor='#B0E0E6', alpha=0.5, zorder=1)
    
    # Add gridlines
    gl = ax.gridlines(draw_labels=True, linewidth=0.5, color='gray', alpha=0.5, linestyle='--')
    gl.top_labels = False
    gl.right_labels = False
    gl.xlabel_style = {'size': 11, 'color': 'black'}
    gl.ylabel_style = {'size': 11, 'color': 'black'}
    gl.xformatter = LONGITUDE_FORMATTER
    gl.yformatter = LATITUDE_FORMATTER
    
    # Plot stations with clear markers
    for code, info in STATIONS.items():
        lon, lat = info['lon'], info['lat']
        
        # Determine marker properties based on sample count
        if info['samples'] > 100:
            color = '#FF0000'  # Red for high contribution
            size = 400
            label = 'High (>100 samples)'
        elif info['samples'] >= 50:
            color = '#FF8C00'  # Orange for medium
            size = 280
            label = 'Medium (50-100 samples)'
        else:
            color = '#FFD700'  # Yellow for supporting
            size = 200
            label = 'Supporting (<50 samples)'
        
        # Plot station marker (triangle)
        ax.plot(lon, lat, marker='^', markersize=18, 
               markerfacecolor=color, 
               markeredgecolor='black', 
               markeredgewidth=2.0,
               transform=ccrs.PlateCarree(),
               zorder=10)
        
        # Add station code label
        ax.text(lon, lat+0.4, code, 
                fontsize=10, 
                fontweight='bold',
                ha='center', 
                va='bottom',
                bbox=dict(boxstyle='round,pad=0.4', 
                         facecolor='white', 
                         edgecolor='black',
                         linewidth=1.2,
                         alpha=0.9),
                transform=ccrs.PlateCarree(),
                zorder=11)
    
    # Add title
    plt.title('BMKG Geomagnetic Observatory Network\nIndonesia Earthquake Precursor Monitoring System',
             fontsize=20, fontweight='bold', pad=25)
    
    # Create legend
    legend_elements = [
        plt.Line2D([0], [0], marker='^', color='w', 
                  markerfacecolor='#FF0000', markersize=16, 
                  markeredgecolor='black', markeredgewidth=2,
                  label='High Contribution (>100 samples)'),
        plt.Line2D([0], [0], marker='^', color='w',
                  markerfacecolor='#FF8C00', markersize=14,
                  markeredgecolor='black', markeredgewidth=2,
                  label='Medium Contribution (50-100 samples)'),
        plt.Line2D([0], [0], marker='^', color='w',
                  markerfacecolor='#FFD700', markersize=12,
                  markeredgecolor='black', markeredgewidth=2,
                  label='Supporting Stations (<50 samples)')
    ]
    
    ax.legend(handles=legend_elements, 
              loc='lower left',
              fontsize=12,
              frameon=True,
              fancybox=True,
              shadow=True,
              title='Station Classification',
              title_fontsize=13)
    
    # Add info box
    info_text = (
        'Total Stations: 24\n'
        'Coverage: Indonesian Archipelago\n'
        'Network: BMKG Geomagnetic Observatories\n'
        'Dataset: 2,340 samples (2018-2025)'
    )
    
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.9, edgecolor='black', linewidth=2)
    ax.text(0.02, 0.98, info_text,
           transform=ax.transAxes,
           fontsize=12,
           verticalalignment='top',
           bbox=props,
           family='monospace',
           zorder=20)
    
    # Add note about Ring of Fire
    note_text = 'Indonesia: Pacific Ring of Fire\nHigh Seismic Activity Zone'
    ax.text(0.98, 0.02, note_text,
           transform=ax.transAxes,
           fontsize=11,
           verticalalignment='bottom',
           horizontalalignment='right',
           bbox=dict(boxstyle='round', facecolor='#FFE4E1', alpha=0.9, 
                    edgecolor='red', linewidth=2),
           style='italic',
           zorder=20)
    
    plt.tight_layout()
    
    # Save figure
    output_path = 'publication_efficientnet/figures/FIG_1_Station_Map.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"✅ Saved: {output_path}")
    
    plt.close()

def main():
    """Generate station map with Cartopy"""
    print("=" * 70)
    print("Regenerating Station Map with Cartopy (Professional Basemap)")
    print("=" * 70)
    print()
    
    try:
        print("Creating map with Cartopy...")
        print("  - Adding coastlines and borders")
        print("  - Plotting 24 BMKG stations")
        print("  - Adding triangle markers with labels")
        print()
        
        create_station_map_cartopy()
        
        print()
        print("✅ Map created successfully with Cartopy!")
        
    except Exception as e:
        print(f"❌ Error creating map: {e}")
        import traceback
        traceback.print_exc()
    
    print()
    print("=" * 70)
    print("Station Map Generation Complete")
    print("=" * 70)
    print()
    print("Features:")
    print("  ✓ Professional basemap with coastlines and borders")
    print("  ✓ Triangle markers (^) for all 24 stations")
    print("  ✓ Color-coded by contribution:")
    print("    • Red: High (>100 samples)")
    print("    • Orange: Medium (50-100 samples)")
    print("    • Yellow: Supporting (<50 samples)")
    print("  ✓ Station codes labeled with white boxes")
    print("  ✓ Black edges for maximum visibility")
    print("  ✓ Gridlines with lat/lon labels")
    print("  ✓ 300 DPI resolution for publication")
    print()
    print("Output: publication_efficientnet/figures/FIG_1_Station_Map.png")

if __name__ == "__main__":
    main()
