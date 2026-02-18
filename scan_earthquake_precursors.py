"""
Earthquake Precursor Scanner
============================
Script untuk scan raw data geomagnetik dari server SSH dan mencari nilai Z/H maksimum
pada periode 4-20 hari sebelum gempa bumi.

Author: Auto-generated
Date: 2026-02-11

Kriteria:
1. Scan raw data di server via SSH
2. Cari nilai Z/H maksimum per jam pada periode 4-20 hari sebelum gempa
3. Output format mengikuti event_list.xlsx: No, Stasiun, Tanggal, Jam, Azm, Mag
4. Gunakan binary reading dari geomagnetic_fetcher.py
5. Proses semua stasiun dari lokasi_stasiun.csv
"""

import os
import sys
import paramiko
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from io import BytesIO
import struct
import logging
import math
from concurrent.futures import ThreadPoolExecutor, as_completed
import time

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('precursor_scan.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# SSH Configuration
SSH_CONFIG = {
    'host': '202.90.198.224',
    'port': 4343,
    'username': 'precursor',
    'password': 'otomatismon'
}

# Station baselines (from geomagnetic_fetcher.py)
BASELINES = {
    'GTO': {'H': 40000, 'Z': 30000},
    'GSI': {'H': 40000, 'Z': 30000},
    'ALR': {'H': 38000, 'Z': 32000},
    'AMB': {'H': 38000, 'Z': 32000},
    'BTN': {'H': 38000, 'Z': 32000},
    'CLP': {'H': 38000, 'Z': 32000},
    'TND': {'H': 40000, 'Z': 30000},
    'SCN': {'H': 38000, 'Z': 32000},
    'MLB': {'H': 38000, 'Z': 32000},
    'SBG': {'H': 38000, 'Z': 32000},
    'YOG': {'H': 38000, 'Z': 32000},
    'MJB': {'H': 38000, 'Z': 32000},
    'LWK': {'H': 38000, 'Z': 32000},
    'SMG': {'H': 38000, 'Z': 32000},
    'SKB': {'H': 38000, 'Z': 32000},
    'TRT': {'H': 38000, 'Z': 32000},
    'PLU': {'H': 38000, 'Z': 32000},
    'LWA': {'H': 38000, 'Z': 32000},
    'KPY': {'H': 38000, 'Z': 32000},
    'LPS': {'H': 38000, 'Z': 32000},
    'SRG': {'H': 38000, 'Z': 32000},
    'LUT': {'H': 38000, 'Z': 32000},
    'SMI': {'H': 38000, 'Z': 32000},
    'TNT': {'H': 38000, 'Z': 32000},
    'SRO': {'H': 38000, 'Z': 32000},
    'TRD': {'H': 38000, 'Z': 32000},
    'JYP': {'H': 38000, 'Z': 32000},
}


class PrecursorScanner:
    """Scanner untuk mencari precursor gempa dari data geomagnetik."""
    
    def __init__(self, ssh_config=None):
        self.config = ssh_config or SSH_CONFIG
        self.ssh_client = None
        self.sftp_client = None
        self.stations = []
        self.earthquakes = []
        
    def connect(self):
        """Establish SSH connection."""
        try:
            self.ssh_client = paramiko.SSHClient()
            self.ssh_client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
            self.ssh_client.connect(
                hostname=self.config['host'],
                port=self.config['port'],
                username=self.config['username'],
                password=self.config['password'],
                timeout=60
            )
            self.sftp_client = self.ssh_client.open_sftp()
            logger.info(f"Connected to {self.config['host']}:{self.config['port']}")
            return True
        except Exception as e:
            logger.error(f"Failed to connect: {e}")
            return False
    
    def disconnect(self):
        """Close SSH connection."""
        if self.sftp_client:
            self.sftp_client.close()
        if self.ssh_client:
            self.ssh_client.close()
        logger.info("Disconnected from server")
    
    def load_stations(self, filepath='mdata2/lokasi_stasiun.csv'):
        """Load station list from CSV."""
        try:
            df = pd.read_csv(filepath, sep=';')
            self.stations = df['Kode Stasiun'].dropna().str.strip().tolist()
            logger.info(f"Loaded {len(self.stations)} stations: {self.stations}")
            return self.stations
        except Exception as e:
            logger.error(f"Failed to load stations: {e}")
            return []
    
    def load_earthquakes(self, filepath='earthquake_catalog_2018_2025_merged.csv', min_magnitude=5.0):
        """Load earthquake catalog and filter by magnitude."""
        try:
            df = pd.read_csv(filepath)
            df['datetime'] = pd.to_datetime(df['datetime'])
            df = df[df['Magnitude'] >= min_magnitude].copy()
            df = df.sort_values('datetime')
            self.earthquakes = df
            logger.info(f"Loaded {len(df)} earthquakes with M >= {min_magnitude}")
            logger.info(f"Date range: {df['datetime'].min()} to {df['datetime'].max()}")
            return df
        except Exception as e:
            logger.error(f"Failed to load earthquakes: {e}")
            return pd.DataFrame()
    
    def parse_binary_data(self, binary_data, station):
        """
        Parse binary geomagnetic data.
        
        Format:
        - Header: 32 bytes
        - Records: 17 bytes each × 86,400 (1 per second)
        - Each record: H(2), D(2), Z(2), IX(2), IY(2), TempS(2), TempP(2), Voltage(2), Spare(1)
        """
        num_seconds = 86400
        baseline = BASELINES.get(station, {'H': 38000, 'Z': 32000})
        
        H_data = np.full(num_seconds, np.nan)
        Z_data = np.full(num_seconds, np.nan)
        
        try:
            header_size = 32
            record_size = 17
            data_start = binary_data[header_size:]
            max_records = min(num_seconds, len(data_start) // record_size)
            
            for i in range(max_records):
                offset = i * record_size
                record = data_start[offset:offset+record_size]
                
                if len(record) < record_size:
                    break
                
                h_dev = struct.unpack('<h', record[0:2])[0] * 0.1
                z_dev = struct.unpack('<h', record[4:6])[0] * 0.1
                
                H_data[i] = baseline['H'] + h_dev
                Z_data[i] = baseline['Z'] + z_dev
            
            return H_data, Z_data
            
        except Exception as e:
            logger.error(f"Error parsing binary: {e}")
            return H_data, Z_data
    
    def fetch_day_data(self, date, station):
        """Fetch data for a single day."""
        if isinstance(date, str):
            date = datetime.strptime(date, '%Y-%m-%d')
        
        yy = date.year % 100
        mm = date.month
        dd = date.day
        
        filename = f"S{yy:02d}{mm:02d}{dd:02d}.{station}"
        remote_path = f"/home/precursor/SEISMO/DATA/{station}/SData/{yy:02d}{mm:02d}/{filename}"
        
        try:
            with BytesIO() as buffer:
                self.sftp_client.getfo(remote_path, buffer)
                buffer.seek(0)
                binary_data = buffer.read()
                return self.parse_binary_data(binary_data, station)
        except FileNotFoundError:
            return None, None
        except Exception as e:
            logger.debug(f"Error fetching {remote_path}: {e}")
            return None, None
    
    def calculate_hourly_zh_max(self, H_data, Z_data):
        """
        Calculate maximum Z/H ratio per hour.
        Returns dict with hour -> max_zh_ratio
        """
        hourly_max = {}
        
        for hour in range(24):
            start_idx = hour * 3600
            end_idx = (hour + 1) * 3600
            
            H_hour = H_data[start_idx:end_idx]
            Z_hour = Z_data[start_idx:end_idx]
            
            # Filter valid data
            valid_mask = ~np.isnan(H_hour) & ~np.isnan(Z_hour) & (H_hour > 0)
            
            if np.sum(valid_mask) > 0:
                H_valid = H_hour[valid_mask]
                Z_valid = Z_hour[valid_mask]
                
                # Calculate Z/H ratio
                zh_ratio = np.abs(Z_valid / H_valid)
                hourly_max[hour] = np.max(zh_ratio)
            else:
                hourly_max[hour] = np.nan
        
        return hourly_max
    
    def calculate_azimuth(self, eq_lat, eq_lon, st_lat, st_lon):
        """Calculate azimuth from station to earthquake epicenter."""
        lat1 = math.radians(st_lat)
        lat2 = math.radians(eq_lat)
        dlon = math.radians(eq_lon - st_lon)
        
        x = math.sin(dlon) * math.cos(lat2)
        y = math.cos(lat1) * math.sin(lat2) - math.sin(lat1) * math.cos(lat2) * math.cos(dlon)
        
        azimuth = math.degrees(math.atan2(x, y))
        return (azimuth + 360) % 360
    
    def scan_precursor_window(self, earthquake, station, station_coords, days_before_min=4, days_before_max=20):
        """
        Scan precursor window for a single earthquake-station pair.
        Returns the hour with maximum Z/H ratio in the precursor window.
        """
        eq_date = earthquake['datetime']
        eq_lat = earthquake['Latitude']
        eq_lon = earthquake['Longitude']
        eq_mag = earthquake['Magnitude']
        
        st_lat = station_coords.get('lat', 0)
        st_lon = station_coords.get('lon', 0)
        
        # Calculate precursor window
        start_date = eq_date - timedelta(days=days_before_max)
        end_date = eq_date - timedelta(days=days_before_min)
        
        max_zh = 0
        max_date = None
        max_hour = None
        
        current_date = start_date
        while current_date <= end_date:
            H_data, Z_data = self.fetch_day_data(current_date, station)
            
            if H_data is not None and Z_data is not None:
                hourly_zh = self.calculate_hourly_zh_max(H_data, Z_data)
                
                for hour, zh in hourly_zh.items():
                    if not np.isnan(zh) and zh > max_zh:
                        max_zh = zh
                        max_date = current_date
                        max_hour = hour
            
            current_date += timedelta(days=1)
        
        if max_date is not None:
            azimuth = self.calculate_azimuth(eq_lat, eq_lon, st_lat, st_lon)
            return {
                'station': station,
                'date': max_date.strftime('%Y-%m-%d'),
                'hour': max_hour,
                'azimuth': round(azimuth, 1),
                'magnitude': round(eq_mag, 1),
                'zh_max': max_zh,
                'eq_date': eq_date.strftime('%Y-%m-%d')
            }
        
        return None
    
    def scan_all(self, output_file='new_event_scanned.csv', min_magnitude=5.0, 
                 days_before_min=4, days_before_max=20, max_earthquakes=None):
        """
        Main scanning function.
        Scans all earthquakes for all stations.
        """
        # Load data
        self.load_stations()
        self.load_earthquakes(min_magnitude=min_magnitude)
        
        if self.earthquakes.empty:
            logger.error("No earthquakes loaded")
            return
        
        # Load station coordinates
        station_coords = {}
        try:
            df_stations = pd.read_csv('mdata2/lokasi_stasiun.csv', sep=';')
            for _, row in df_stations.iterrows():
                code = str(row['Kode Stasiun']).strip()
                try:
                    lat = float(str(row['Latitude']).strip())
                    lon = float(str(row['Longitude']).strip())
                    station_coords[code] = {'lat': lat, 'lon': lon}
                except:
                    pass
        except Exception as e:
            logger.warning(f"Could not load station coordinates: {e}")
        
        # Connect to server
        if not self.connect():
            logger.error("Failed to connect to server")
            return
        
        results = []
        event_no = 1
        
        earthquakes_to_process = self.earthquakes
        if max_earthquakes:
            earthquakes_to_process = earthquakes_to_process.head(max_earthquakes)
        
        total_eq = len(earthquakes_to_process)
        logger.info(f"Processing {total_eq} earthquakes across {len(self.stations)} stations")
        
        try:
            for eq_idx, (_, earthquake) in enumerate(earthquakes_to_process.iterrows()):
                eq_date = earthquake['datetime'].strftime('%Y-%m-%d')
                eq_mag = earthquake['Magnitude']
                
                logger.info(f"[{eq_idx+1}/{total_eq}] Processing earthquake {eq_date} M{eq_mag:.1f}")
                
                for station in self.stations:
                    coords = station_coords.get(station, {'lat': 0, 'lon': 0})
                    
                    result = self.scan_precursor_window(
                        earthquake, station, coords,
                        days_before_min, days_before_max
                    )
                    
                    if result:
                        result['No'] = event_no
                        results.append(result)
                        event_no += 1
                        logger.info(f"  Found precursor: {station} {result['date']} H{result['hour']} Z/H={result['zh_max']:.4f}")
                
                # Save intermediate results every 10 earthquakes
                if (eq_idx + 1) % 10 == 0:
                    self._save_results(results, output_file)
                    logger.info(f"Saved {len(results)} results to {output_file}")
        
        finally:
            self.disconnect()
        
        # Save final results
        self._save_results(results, output_file)
        logger.info(f"Scan complete. Total events found: {len(results)}")
        
        return results
    
    def _save_results(self, results, output_file):
        """Save results to CSV in event_list format."""
        if not results:
            return
        
        df = pd.DataFrame(results)
        
        # Format to match event_list.xlsx: No, Stasiun, Tanggal, Jam, Azm, Mag
        df_output = pd.DataFrame({
            'No': df['No'],
            'Stasiun': df['station'],
            'Tanggal': df['date'],
            'Jam': df['hour'],
            'Azm': df['azimuth'],
            'Mag': df['magnitude']
        })
        
        df_output.to_csv(output_file, index=False)
        
        # Also save detailed version
        df.to_csv(output_file.replace('.csv', '_detailed.csv'), index=False)


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Scan earthquake precursors from geomagnetic data')
    parser.add_argument('--min-mag', type=float, default=5.0, help='Minimum earthquake magnitude')
    parser.add_argument('--days-min', type=int, default=4, help='Minimum days before earthquake')
    parser.add_argument('--days-max', type=int, default=20, help='Maximum days before earthquake')
    parser.add_argument('--output', type=str, default='new_event_scanned.csv', help='Output file')
    parser.add_argument('--max-eq', type=int, default=None, help='Maximum earthquakes to process (for testing)')
    parser.add_argument('--test', action='store_true', help='Run quick test with 3 earthquakes')
    
    args = parser.parse_args()
    
    scanner = PrecursorScanner()
    
    if args.test:
        logger.info("Running test mode with 3 earthquakes")
        scanner.scan_all(
            output_file='test_precursor_scan.csv',
            min_magnitude=args.min_mag,
            days_before_min=args.days_min,
            days_before_max=args.days_max,
            max_earthquakes=3
        )
    else:
        scanner.scan_all(
            output_file=args.output,
            min_magnitude=args.min_mag,
            days_before_min=args.days_min,
            days_before_max=args.days_max,
            max_earthquakes=args.max_eq
        )


if __name__ == '__main__':
    main()
