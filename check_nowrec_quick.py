#!/usr/bin/env python3
"""
Quick Check Nowrec Data - Faster version
"""

import paramiko
import pandas as pd
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

SSH_CONFIG = {
    'host': '202.90.198.224',
    'port': 4343,
    'username': 'precursor',
    'password': 'otomatismon'
}

# Load missing data
missing_df = pd.read_csv('missing_data.csv')
logger.info(f"Total missing: {len(missing_df)}")

# Connect
ssh = paramiko.SSHClient()
ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
ssh.connect(
    hostname=SSH_CONFIG['host'],
    port=SSH_CONFIG['port'],
    username=SSH_CONFIG['username'],
    password=SSH_CONFIG['password'],
    timeout=30
)
sftp = ssh.open_sftp()
logger.info("Connected!")

found_files = []
stations_checked = set()

for _, row in missing_df.iterrows():
    station = row['station']
    date_raw = str(int(row['date_raw']))
    yy = date_raw[2:4]
    mm = date_raw[4:6]
    dd = date_raw[6:8]
    
    filename = f"S{yy}{mm}{dd}.{station}"
    nowrec_path = f"/home/precursor/SEISMO/DATA/{station}/Nowrec/{filename}"
    
    try:
        sftp.stat(nowrec_path)
        found_files.append({
            'station': station,
            'date': row['date'],
            'hour': row['hour'],
            'magnitude': row['magnitude'],
            'azimuth': row['azimuth'],
            'filename': filename,
            'path': nowrec_path
        })
        logger.info(f"✓ {filename}")
    except FileNotFoundError:
        pass
    except Exception as e:
        if station not in stations_checked:
            logger.warning(f"Error {station}: {e}")
            stations_checked.add(station)

sftp.close()
ssh.close()

# Save results
if found_files:
    found_df = pd.DataFrame(found_files)
    found_df.to_csv('nowrec_available.csv', index=False)
    logger.info(f"\n{'='*50}")
    logger.info(f"Found {len(found_files)} files in Nowrec!")
    logger.info(f"Saved to: nowrec_available.csv")
else:
    logger.info("No files found in Nowrec")

logger.info(f"Total missing: {len(missing_df)}, Found: {len(found_files)}")
