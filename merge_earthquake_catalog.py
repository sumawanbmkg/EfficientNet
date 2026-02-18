"""
Merge Earthquake Catalog 2018-2025
==================================
Script untuk menggabungkan semua file katalog gempa dari repository 2018-2025
menjadi satu file CSV dengan format yang konsisten.

Output: earthquake_catalog_2018_2025_merged.csv
"""

import os
import pandas as pd
from datetime import datetime
import glob
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

BASE_PATH = "repository gempa 2018-2025"

def read_2018_format(filepath):
    """Read 2018 format with detailed columns."""
    try:
        df = pd.read_csv(filepath)
        # Normalize column names
        df = df.rename(columns={
            'Date Time': 'datetime',
            'ID': 'event_id'
        })
        df['datetime'] = pd.to_datetime(df['datetime'], utc=True)
        df['datetime'] = df['datetime'].dt.tz_localize(None)  # Remove timezone
        return df[['event_id', 'datetime', 'Latitude', 'Longitude', 'Magnitude', 'Depth']].copy()
    except Exception as e:
        logger.error(f"Error reading {filepath}: {e}")
        return pd.DataFrame()

def read_2019_2021_format(filepath):
    """Read 2019-2021 format."""
    try:
        df = pd.read_csv(filepath)
        df = df.rename(columns={
            'Date Time': 'datetime',
            'ID': 'event_id'
        })
        df['datetime'] = pd.to_datetime(df['datetime'], utc=True)
        df['datetime'] = df['datetime'].dt.tz_localize(None)  # Remove timezone
        return df[['event_id', 'datetime', 'Latitude', 'Longitude', 'Magnitude', 'Depth']].copy()
    except Exception as e:
        logger.error(f"Error reading {filepath}: {e}")
        return pd.DataFrame()

def read_2023_2025_format(filepath):
    """Read 2023-2025 modern format."""
    try:
        df = pd.read_csv(filepath)
        df = df.rename(columns={
            'Date time': 'datetime',
            'Event ID': 'event_id',
            'Depth (km)': 'Depth'
        })
        df['datetime'] = pd.to_datetime(df['datetime'], utc=True)
        df['datetime'] = df['datetime'].dt.tz_localize(None)  # Remove timezone
        return df[['event_id', 'datetime', 'Latitude', 'Longitude', 'Magnitude', 'Depth']].copy()
    except Exception as e:
        logger.error(f"Error reading {filepath}: {e}")
        return pd.DataFrame()

def merge_all_catalogs():
    """Merge all earthquake catalogs from 2018-2025."""
    all_data = []
    
    # 2018
    logger.info("Processing 2018...")
    path_2018 = os.path.join(BASE_PATH, "1. 2018")
    for f in glob.glob(os.path.join(path_2018, "*.csv")):
        df = read_2018_format(f)
        if not df.empty:
            all_data.append(df)
            logger.info(f"  {os.path.basename(f)}: {len(df)} records")
    
    # 2019
    logger.info("Processing 2019...")
    path_2019 = os.path.join(BASE_PATH, "2. 2019")
    for f in glob.glob(os.path.join(path_2019, "*.csv")):
        df = read_2019_2021_format(f)
        if not df.empty:
            all_data.append(df)
            logger.info(f"  {os.path.basename(f)}: {len(df)} records")
    
    # 2020
    logger.info("Processing 2020...")
    path_2020 = os.path.join(BASE_PATH, "3. 2020")
    for f in glob.glob(os.path.join(path_2020, "*.csv")):
        df = read_2019_2021_format(f)
        if not df.empty:
            all_data.append(df)
            logger.info(f"  {os.path.basename(f)}: {len(df)} records")
    
    # 2021
    logger.info("Processing 2021...")
    path_2021 = os.path.join(BASE_PATH, "4. 2021")
    for f in glob.glob(os.path.join(path_2021, "*.csv")):
        df = read_2019_2021_format(f)
        if not df.empty:
            all_data.append(df)
            logger.info(f"  {os.path.basename(f)}: {len(df)} records")
    
    # 2022
    logger.info("Processing 2022...")
    path_2022_final = os.path.join(BASE_PATH, "5. 2022", "Event Tipe Final")
    for f in glob.glob(os.path.join(path_2022_final, "*.csv")):
        df = read_2019_2021_format(f)
        if not df.empty:
            all_data.append(df)
            logger.info(f"  {os.path.basename(f)}: {len(df)} records")
    
    # 2023
    logger.info("Processing 2023...")
    path_2023_pre = os.path.join(BASE_PATH, "6. 2023", "DATA GEMPA 2023 (pre)")
    for f in glob.glob(os.path.join(path_2023_pre, "*.csv")):
        df = read_2023_2025_format(f)
        if not df.empty:
            all_data.append(df)
            logger.info(f"  {os.path.basename(f)}: {len(df)} records")
    
    # 2023 final
    path_2023_final = os.path.join(BASE_PATH, "6. 2023", "SEPTEMBER (final).csv")
    if os.path.exists(path_2023_final):
        df = read_2023_2025_format(path_2023_final)
        if not df.empty:
            all_data.append(df)
            logger.info(f"  SEPTEMBER (final).csv: {len(df)} records")
    
    # 2024
    logger.info("Processing 2024...")
    path_2024_pre = os.path.join(BASE_PATH, "7. 2024", "DATA GEMPA 2024 (Pre)")
    for f in glob.glob(os.path.join(path_2024_pre, "*.csv")):
        df = read_2023_2025_format(f)
        if not df.empty:
            all_data.append(df)
            logger.info(f"  {os.path.basename(f)}: {len(df)} records")
    
    # 2024 final
    path_2024_final = os.path.join(BASE_PATH, "7. 2024", "final")
    for f in glob.glob(os.path.join(path_2024_final, "*.csv")):
        df = read_2023_2025_format(f)
        if not df.empty:
            all_data.append(df)
            logger.info(f"  {os.path.basename(f)}: {len(df)} records")
    
    # 2025
    logger.info("Processing 2025...")
    path_2025 = os.path.join(BASE_PATH, "8. 2025")
    for f in glob.glob(os.path.join(path_2025, "*.csv")):
        df = read_2023_2025_format(f)
        if not df.empty:
            all_data.append(df)
            logger.info(f"  {os.path.basename(f)}: {len(df)} records")
    
    # Merge all
    if all_data:
        merged = pd.concat(all_data, ignore_index=True)
        
        # Remove duplicates based on datetime and location
        merged = merged.drop_duplicates(subset=['datetime', 'Latitude', 'Longitude'])
        
        # Sort by datetime
        merged = merged.sort_values('datetime')
        
        # Reset index
        merged = merged.reset_index(drop=True)
        merged['No'] = merged.index + 1
        
        # Reorder columns
        merged = merged[['No', 'event_id', 'datetime', 'Latitude', 'Longitude', 'Magnitude', 'Depth']]
        
        return merged
    
    return pd.DataFrame()

def main():
    """Main function."""
    logger.info("Starting earthquake catalog merge...")
    
    merged = merge_all_catalogs()
    
    if not merged.empty:
        output_file = "earthquake_catalog_2018_2025_merged.csv"
        merged.to_csv(output_file, index=False)
        
        logger.info(f"\n{'='*50}")
        logger.info(f"Merge complete!")
        logger.info(f"Total records: {len(merged)}")
        logger.info(f"Date range: {merged['datetime'].min()} to {merged['datetime'].max()}")
        logger.info(f"Output file: {output_file}")
        
        # Statistics
        logger.info(f"\nMagnitude statistics:")
        logger.info(f"  Min: {merged['Magnitude'].min():.1f}")
        logger.info(f"  Max: {merged['Magnitude'].max():.1f}")
        logger.info(f"  Mean: {merged['Magnitude'].mean():.1f}")
        
        # Count by magnitude
        logger.info(f"\nCount by magnitude:")
        logger.info(f"  M >= 4.0: {len(merged[merged['Magnitude'] >= 4.0])}")
        logger.info(f"  M >= 5.0: {len(merged[merged['Magnitude'] >= 5.0])}")
        logger.info(f"  M >= 6.0: {len(merged[merged['Magnitude'] >= 6.0])}")
        logger.info(f"  M >= 7.0: {len(merged[merged['Magnitude'] >= 7.0])}")
        
        # Count by year
        merged['year'] = merged['datetime'].dt.year
        logger.info(f"\nCount by year:")
        for year in sorted(merged['year'].unique()):
            count = len(merged[merged['year'] == year])
            logger.info(f"  {year}: {count}")
    else:
        logger.error("No data merged!")

if __name__ == '__main__':
    main()
