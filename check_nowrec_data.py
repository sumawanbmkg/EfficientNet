#!/usr/bin/env python3
"""
Check Nowrec Data Availability on Server
Scans /home/precursor/SEISMO/DATA/{station}/Nowrec for missing data

Date: 6 February 2026
"""

import paramiko
import pandas as pd
from datetime import datetime
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

SSH_CONFIG = {
    'host': '202.90.198.224',
    'port': 4343,
    'username': 'precursor',
    'password': 'otomatismon'
}

def check_nowrec_availability():
    """Check which missing data is available in Nowrec folder"""
    
    # Load missing data list
    missing_df = pd.read_csv('missing_data.csv')
    logger.info(f"Loaded {len(missing_df)} missing records")
    
    # Get unique stations
    stations = missing_df['station'].unique()
    logger.info(f"Stations to check: {list(stations)}")
    
    # Connect to server
    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    
    try:
        ssh.connect(
            hostname=SSH_CONFIG['host'],
            port=SSH_CONFIG['port'],
            username=SSH_CONFIG['username'],
            password=SSH_CONFIG['password'],
            timeout=30
        )
        sftp = ssh.open_sftp()
        logger.info(f"Connected to {SSH_CONFIG['host']}:{SSH_CONFIG['port']}")
        
        found_files = []
        not_found = []
        
        for station in stations:
            # Check Nowrec folder
            nowrec_path = f"/home/precursor/SEISMO/DATA/{station}/Nowrec"
            
            logger.info(f"\nChecking station: {station}")
            logger.info(f"  Path: {nowrec_path}")
            
            try:
                # List files in Nowrec
                files = sftp.listdir(nowrec_path)
                logger.info(f"  Found {len(files)} files in Nowrec")
                
                # Check each missing date for this station
                station_missing = missing_df[missing_df['station'] == station]
                
                for _, row in station_missing.iterrows():
                    date_raw = row['date_raw']
                    yy = int(str(date_raw)[2:4])
                    mm = int(str(date_raw)[4:6])
                    dd = int(str(date_raw)[6:8])
                    
                    # Expected filename: SYYMMDD.STN or SYYMMDD.STN.gz
                    filename_base = f"S{yy:02d}{mm:02d}{dd:02d}.{station}"
                    
                    # Check if file exists
                    if filename_base in files or f"{filename_base}.gz" in files:
                        found_files.append({
                            'station': station,
                            'date': row['date'],
                            'hour': row['hour'],
                            'magnitude': row['magnitude'],
                            'azimuth': row['azimuth'],
                            'filename': filename_base,
                            'path': f"{nowrec_path}/{filename_base}"
                        })
                        logger.info(f"    ✓ Found: {filename_base}")
                    else:
                        not_found.append({
                            'station': station,
                            'date': row['date'],
                            'filename': filename_base
                        })
                        
            except FileNotFoundError:
                logger.warning(f"  Nowrec folder not found for {station}")
            except Exception as e:
                logger.error(f"  Error checking {station}: {e}")
        
        # Save results
        if found_files:
            found_df = pd.DataFrame(found_files)
            found_df.to_csv('nowrec_available.csv', index=False)
            logger.info(f"\n✓ Found {len(found_files)} files in Nowrec")
            logger.info(f"  Saved to: nowrec_available.csv")
        
        logger.info(f"\n{'='*60}")
        logger.info(f"SUMMARY")
        logger.info(f"{'='*60}")
        logger.info(f"Total missing: {len(missing_df)}")
        logger.info(f"Found in Nowrec: {len(found_files)}")
        logger.info(f"Still missing: {len(not_found)}")
        
        sftp.close()
        ssh.close()
        
        return found_files
        
    except Exception as e:
        logger.error(f"Connection error: {e}")
        return []


if __name__ == "__main__":
    check_nowrec_availability()
