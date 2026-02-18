#!/usr/bin/env python3
"""
Check File Formats on Server
Simple script to check what file formats are available
"""

import sys
sys.path.insert(0, 'intial')
from geomagnetic_fetcher import GeomagneticDataFetcher
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def check_file_formats():
    """Check what file formats are available on server"""
    
    with GeomagneticDataFetcher() as fetcher:
        # Test different file format possibilities
        test_cases = [
            # Current format (no extension)
            {
                'date': '2018-01-17',
                'station': 'SCN',
                'path_format': '/home/precursor/SEISMO/DATA/{station}/SData/{yy:02d}{mm:02d}/S{yy:02d}{mm:02d}{dd:02d}.{station}',
                'description': 'Current format (.STN extension)'
            },
            # .gz format
            {
                'date': '2018-01-17', 
                'station': 'SCN',
                'path_format': '/home/precursor/SEISMO/DATA/{station}/SData/{yy:02d}{mm:02d}/S{yy:02d}{mm:02d}{dd:02d}.{station}.gz',
                'description': 'Compressed format (.gz extension)'
            },
            # Alternative directory structure
            {
                'date': '2018-01-17',
                'station': 'SCN', 
                'path_format': '/home/precursor/SEISMO/DATA/{station}/GZData/{yy:02d}{mm:02d}/S{yy:02d}{mm:02d}{dd:02d}.{station}.gz',
                'description': 'GZData directory with .gz files'
            }
        ]
        
        for test_case in test_cases:
            logger.info(f"\n=== Testing: {test_case['description']} ===")
            
            from datetime import datetime
            date = datetime.strptime(test_case['date'], '%Y-%m-%d')
            station = test_case['station']
            
            yy = date.year % 100
            mm = date.month
            dd = date.day
            
            test_path = test_case['path_format'].format(
                station=station, yy=yy, mm=mm, dd=dd
            )
            
            logger.info(f"Testing path: {test_path}")
            
            try:
                # Try to get file info
                file_stat = fetcher.sftp_client.stat(test_path)
                logger.info(f"✅ File exists! Size: {file_stat.st_size} bytes")
                
                # Try to read first few bytes to determine format
                with fetcher.sftp_client.open(test_path, 'rb') as f:
                    first_bytes = f.read(10)
                    logger.info(f"First 10 bytes: {first_bytes.hex()}")
                    
                    # Check if it's gzipped
                    if first_bytes[:2] == b'\x1f\x8b':
                        logger.info("🗜️  File is GZIP compressed")
                    else:
                        logger.info("📄 File is uncompressed binary")
                
            except FileNotFoundError:
                logger.info("❌ File not found")
            except Exception as e:
                logger.error(f"❌ Error: {e}")

if __name__ == '__main__':
    check_file_formats()