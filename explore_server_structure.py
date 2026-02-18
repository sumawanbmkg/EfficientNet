#!/usr/bin/env python3
"""
Explore Server Structure
Script untuk mengeksplorasi struktur direktori di server SSH
untuk melihat format file yang tersedia (.gz dan .STN)
"""

import paramiko
import logging
from datetime import datetime

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# SSH Configuration
SSH_CONFIG = {
    'host': '202.90.198.224',
    'port': 4343,
    'username': 'precursor',
    'password': 'otomatismon'
}

def explore_server_structure():
    """Explore server directory structure"""
    
    try:
        # Connect to server
        ssh_client = paramiko.SSHClient()
        ssh_client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        
        ssh_client.connect(
            hostname=SSH_CONFIG['host'],
            port=SSH_CONFIG['port'],
            username=SSH_CONFIG['username'],
            password=SSH_CONFIG['password'],
            timeout=30
        )
        
        sftp_client = ssh_client.open_sftp()
        logger.info(f"Connected to {SSH_CONFIG['host']}:{SSH_CONFIG['port']}")
        
        # Explore base directory
        base_path = "/home/precursor/SEISMO/DATA"
        logger.info(f"\n=== Exploring {base_path} ===")
        
        try:
            stations = sftp_client.listdir(base_path)
            logger.info(f"Available stations: {stations[:10]}...")  # Show first 10
            
            # Check a few stations for directory structure
            test_stations = ['SCN', 'GTO', 'MLB']
            
            for station in test_stations:
                if station in stations:
                    station_path = f"{base_path}/{station}"
                    logger.info(f"\n--- Station {station} ---")
                    
                    try:
                        station_contents = sftp_client.listdir(station_path)
                        logger.info(f"Contents: {station_contents}")
                        
                        # Check SData directory
                        if 'SData' in station_contents:
                            sdata_path = f"{station_path}/SData"
                            logger.info(f"\n--- {station}/SData ---")
                            
                            try:
                                sdata_contents = sftp_client.listdir(sdata_path)
                                logger.info(f"Year-month directories: {sdata_contents[:10]}...")
                                
                                # Check a specific year-month directory
                                if '1801' in sdata_contents:  # January 2018
                                    month_path = f"{sdata_path}/1801"
                                    logger.info(f"\n--- {station}/SData/1801 ---")
                                    
                                    try:
                                        files = sftp_client.listdir(month_path)
                                        logger.info(f"Files in 1801: {files[:10]}...")
                                        
                                        # Analyze file extensions
                                        extensions = {}
                                        for file in files:
                                            if '.' in file:
                                                ext = file.split('.')[-1]
                                                extensions[ext] = extensions.get(ext, 0) + 1
                                            else:
                                                extensions['no_ext'] = extensions.get('no_ext', 0) + 1
                                        
                                        logger.info(f"File extensions: {extensions}")
                                        
                                        # Show some example files
                                        logger.info(f"Example files:")
                                        for i, file in enumerate(files[:5]):
                                            try:
                                                file_path = f"{month_path}/{file}"
                                                file_stat = sftp_client.stat(file_path)
                                                logger.info(f"  {file} - Size: {file_stat.st_size} bytes")
                                            except:
                                                logger.info(f"  {file} - Could not get size")
                                        
                                    except Exception as e:
                                        logger.error(f"Error reading {month_path}: {e}")
                                
                            except Exception as e:
                                logger.error(f"Error reading {sdata_path}: {e}")
                        
                        # Check other directories
                        other_dirs = [d for d in station_contents if d != 'SData']
                        if other_dirs:
                            logger.info(f"Other directories in {station}: {other_dirs}")
                            
                            # Check first other directory
                            if other_dirs:
                                other_path = f"{station_path}/{other_dirs[0]}"
                                try:
                                    other_contents = sftp_client.listdir(other_path)
                                    logger.info(f"Contents of {other_dirs[0]}: {other_contents[:5]}...")
                                except Exception as e:
                                    logger.error(f"Error reading {other_path}: {e}")
                        
                    except Exception as e:
                        logger.error(f"Error reading station {station}: {e}")
            
        except Exception as e:
            logger.error(f"Error reading base path: {e}")
        
        # Close connections
        sftp_client.close()
        ssh_client.close()
        logger.info("Disconnected from server")
        
    except Exception as e:
        logger.error(f"Connection failed: {e}")

if __name__ == '__main__':
    explore_server_structure()