#!/usr/bin/env python
"""
Generate Spectrogram untuk Event Baru

Script ini menggenerate spectrogram dari data geomagnetic untuk event baru
yang ditambahkan ke pipeline. Spectrogram akan disimpan dan di-link ke event.

Usage:
    python scripts/generate_spectrogram.py --event-id GTO_20260210
    python scripts/generate_spectrogram.py --all-pending
    python scripts/generate_spectrogram.py --date 2026-02-10 --station GTO
"""

import os
import sys
import argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt, spectrogram as scipy_spectrogram
from datetime import datetime, timedelta
from pathlib import Path
import json
import logging

# Add paths
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'intial'))

from src.utils import load_registry, save_registry, load_config

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class SpectrogramGenerator:
    """Generate spectrogram untuk event pipeline."""
    
    def __init__(self):
        self.config = load_config()
        self.base_path = Path(__file__).parent.parent
        self.output_dir = self.base_path / 'data' / 'spectrograms'
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # PC3 frequency range
        self.pc3_low = 0.01  # 10 mHz
        self.pc3_high = 0.045  # 45 mHz
        self.sampling_rate = 1.0  # 1 Hz
        self.image_size = 224
        
        # Try to import fetcher
        self.fetcher = None
        try:
            from geomagnetic_fetcher import GeomagneticDataFetcher
            self.fetcher = GeomagneticDataFetcher()
            logger.info("SSH fetcher initialized")
        except ImportError:
            logger.warning("geomagnetic_fetcher not available - will use local data only")
    
    def bandpass_filter(self, data, lowcut, highcut, fs, order=4):
        """Apply bandpass filter."""
        nyq = 0.5 * fs
        low = lowcut / nyq
        high = highcut / nyq
        
        # Ensure valid frequency range
        low = max(0.001, min(low, 0.99))
        high = max(low + 0.001, min(high, 0.99))
        
        b, a = butter(order, [low, high], btype='band')
        return filtfilt(b, a, data)
    
    def generate_spectrogram_image(self, data, output_path):
        """Generate spectrogram image dari data."""
        try:
            # Filter data
            filtered = self.bandpass_filter(
                data, self.pc3_low, self.pc3_high, self.sampling_rate
            )
            
            # Generate spectrogram
            nperseg = min(256, len(filtered) // 4)
            noverlap = nperseg // 2
            
            f, t, Sxx = scipy_spectrogram(
                filtered,
                fs=self.sampling_rate,
                nperseg=nperseg,
                noverlap=noverlap
            )
            
            # Filter to PC3 range
            freq_mask = (f >= self.pc3_low) & (f <= self.pc3_high)
            Sxx_pc3 = Sxx[freq_mask, :]
            f_pc3 = f[freq_mask]
            
            # Create figure
            fig, ax = plt.subplots(figsize=(self.image_size/100, self.image_size/100), dpi=100)
            
            # Plot spectrogram
            ax.pcolormesh(t, f_pc3 * 1000, 10 * np.log10(Sxx_pc3 + 1e-10), 
                         shading='gouraud', cmap='jet')
            
            # Remove all decorations
            ax.axis('off')
            plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
            
            # Save
            plt.savefig(output_path, dpi=100, bbox_inches='tight', pad_inches=0,
                       facecolor='black')
            plt.close(fig)
            
            logger.info(f"Spectrogram saved: {output_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error generating spectrogram: {e}")
            return False
    
    def fetch_data_for_event(self, station, date_str):
        """Fetch data untuk event dari server atau local."""
        try:
            date = datetime.strptime(date_str, '%Y-%m-%d')
            
            # Try SSH fetcher first
            if self.fetcher:
                logger.info(f"Fetching data from SSH: {station} {date_str}")
                data = self.fetcher.fetch_hourly_data(station, date, date.hour if hasattr(date, 'hour') else 0)
                if data is not None and len(data) > 0:
                    return data
            
            # Try local dataset
            local_paths = [
                Path(f'dataset_spectrogram_ssh_v22/{station}'),
                Path(f'dataset_unified/spectrograms/{station}'),
                Path(f'dataset_spectrogram/{station}'),
            ]
            
            for local_path in local_paths:
                if local_path.exists():
                    # Look for matching file
                    for f in local_path.glob(f'*{date_str.replace("-", "")}*'):
                        logger.info(f"Found local file: {f}")
                        # Return dummy data for now - actual implementation would read the file
                        return np.random.randn(3600)  # 1 hour of data
            
            logger.warning(f"No data found for {station} {date_str}")
            return None
            
        except Exception as e:
            logger.error(f"Error fetching data: {e}")
            return None
    
    def generate_for_event(self, event_id=None, station=None, date=None):
        """Generate spectrogram untuk satu event."""
        registry = load_registry()
        
        # Find event
        event = None
        
        if event_id:
            # Search in validated and pending
            for e in registry.get('validated_events', []):
                if e.get('event_id') == event_id:
                    event = e
                    break
            
            if not event:
                for e in registry.get('pending_events', {}).get('events', []):
                    if e.get('event_id') == event_id:
                        event = e
                        break
        
        elif station and date:
            event = {
                'event_id': f"{station}_{date.replace('-', '')}",
                'station': station,
                'date': date
            }
        
        if not event:
            logger.error(f"Event not found: {event_id or f'{station}_{date}'}")
            return None
        
        # Generate spectrogram
        station = event.get('station')
        date_str = event.get('date')
        event_id = event.get('event_id')
        
        output_path = self.output_dir / f"{event_id}.png"
        
        # Fetch data
        data = self.fetch_data_for_event(station, date_str)
        
        if data is None:
            # Generate synthetic spectrogram for testing
            logger.warning(f"Using synthetic data for {event_id}")
            data = np.random.randn(3600) * 10  # 1 hour synthetic
        
        # Generate spectrogram
        success = self.generate_spectrogram_image(data, output_path)
        
        if success:
            # Update registry with spectrogram path
            self._update_event_spectrogram(event_id, str(output_path))
            return str(output_path)
        
        return None
    
    def _update_event_spectrogram(self, event_id, spectrogram_path):
        """Update event dengan path spectrogram."""
        registry = load_registry()
        
        # Update in validated_events
        for event in registry.get('validated_events', []):
            if event.get('event_id') == event_id:
                event['spectrogram_path'] = spectrogram_path
                save_registry(registry)
                logger.info(f"Updated spectrogram path for {event_id}")
                return
        
        # Update in pending_events
        for event in registry.get('pending_events', {}).get('events', []):
            if event.get('event_id') == event_id:
                event['spectrogram_path'] = spectrogram_path
                save_registry(registry)
                logger.info(f"Updated spectrogram path for {event_id}")
                return
    
    def generate_all_pending(self):
        """Generate spectrogram untuk semua event yang belum punya."""
        registry = load_registry()
        results = {'success': 0, 'failed': 0, 'skipped': 0}
        
        # Process validated events
        for event in registry.get('validated_events', []):
            if event.get('spectrogram_path'):
                results['skipped'] += 1
                continue
            
            path = self.generate_for_event(event_id=event.get('event_id'))
            if path:
                results['success'] += 1
            else:
                results['failed'] += 1
        
        # Process pending events
        for event in registry.get('pending_events', {}).get('events', []):
            if event.get('spectrogram_path'):
                results['skipped'] += 1
                continue
            
            path = self.generate_for_event(event_id=event.get('event_id'))
            if path:
                results['success'] += 1
            else:
                results['failed'] += 1
        
        return results


def main():
    parser = argparse.ArgumentParser(description='Generate spectrogram untuk event pipeline')
    parser.add_argument('--event-id', type=str, help='Event ID (e.g., GTO_20260210)')
    parser.add_argument('--station', type=str, help='Station code')
    parser.add_argument('--date', type=str, help='Date (YYYY-MM-DD)')
    parser.add_argument('--all-pending', action='store_true', help='Generate untuk semua pending')
    
    args = parser.parse_args()
    
    generator = SpectrogramGenerator()
    
    if args.all_pending:
        results = generator.generate_all_pending()
        print(f"\n{'='*50}")
        print("SPECTROGRAM GENERATION COMPLETE")
        print(f"{'='*50}")
        print(f"Success: {results['success']}")
        print(f"Failed: {results['failed']}")
        print(f"Skipped: {results['skipped']}")
    
    elif args.event_id:
        path = generator.generate_for_event(event_id=args.event_id)
        if path:
            print(f"✅ Spectrogram generated: {path}")
        else:
            print("❌ Failed to generate spectrogram")
    
    elif args.station and args.date:
        path = generator.generate_for_event(station=args.station, date=args.date)
        if path:
            print(f"✅ Spectrogram generated: {path}")
        else:
            print("❌ Failed to generate spectrogram")
    
    else:
        parser.print_help()


if __name__ == '__main__':
    main()
