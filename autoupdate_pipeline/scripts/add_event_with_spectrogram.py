#!/usr/bin/env python3
"""
Add Event with Spectrogram Generation

Script untuk menambahkan event baru ke pipeline dengan generate spectrogram
dari data geomagnetic server.

Usage:
    # Tambah event dengan generate spectrogram dari server
    python add_event_with_spectrogram.py --date 2018-01-17 --station SCN --hour 6 --magnitude 6.2 --azimuth 45
    
    # Tambah event dengan spectrogram yang sudah ada
    python add_event_with_spectrogram.py --date 2018-01-17 --station SCN --magnitude 6.2 --azimuth 45 --spectrogram path/to/spec.png
    
    # Test dengan event dari dataset yang sudah ada
    python add_event_with_spectrogram.py --test-existing

Author: Auto-Update Pipeline
Date: February 2026
"""

import os
import sys
import argparse
import logging
from pathlib import Path
from datetime import datetime

# Add paths
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def generate_spectrogram_for_event(station: str, date: str, hour: int, output_dir: str = None) -> str:
    """
    Generate spectrogram untuk event dari SSH server.
    
    Args:
        station: Kode stasiun (e.g., 'SCN')
        date: Tanggal event (YYYY-MM-DD)
        hour: Jam event (0-23)
        output_dir: Directory output (default: autoupdate_pipeline/data/spectrograms)
        
    Returns:
        Path ke file spectrogram yang di-generate
    """
    try:
        # Import generator
        sys.path.insert(0, str(Path(__file__).parent.parent.parent))
        from geomagnetic_dataset_generator_ssh_v22 import GeomagneticDatasetGeneratorSSH_V22
        
        # Setup output directory
        if output_dir is None:
            output_dir = str(Path(__file__).parent.parent / 'data' / 'spectrograms')
        
        os.makedirs(output_dir, exist_ok=True)
        
        # Create temporary generator
        temp_output = str(Path(__file__).parent.parent / 'data' / 'temp_spectrogram')
        generator = GeomagneticDatasetGeneratorSSH_V22(output_dir=temp_output)
        
        # Parse date
        date_obj = datetime.strptime(date, '%Y-%m-%d')
        
        # Import fetcher
        sys.path.insert(0, 'intial')
        from geomagnetic_fetcher import GeomagneticDataFetcher
        
        logger.info(f"Generating spectrogram for {station} - {date} Hour {hour}")
        logger.info("Connecting to SSH server...")
        
        with GeomagneticDataFetcher(prefer_compressed=True) as fetcher:
            # Fetch data
            data = generator.fetch_hour_data(
                fetcher, date_obj.year, date_obj.month, date_obj.day, hour, station
            )
            
            if data is None:
                logger.error("Failed to fetch data from server")
                return None
            
            if len(data['H']) < 100:
                logger.error(f"Insufficient data: only {len(data['H'])} samples")
                return None
            
            # Apply PC3 filter
            h_pc3 = generator.apply_pc3_filter(data['H'])
            d_pc3 = generator.apply_pc3_filter(data['D'])
            z_pc3 = generator.apply_pc3_filter(data['Z'])
            
            # Generate spectrograms
            f_h, t_h, Sxx_h = generator.generate_spectrogram(h_pc3, 'H')
            f_d, t_d, Sxx_d = generator.generate_spectrogram(d_pc3, 'D')
            f_z, t_z, Sxx_z = generator.generate_spectrogram(z_pc3, 'Z')
            
            # Save spectrogram
            date_str = date.replace('-', '')
            filename = f"{station}_{date_str}_H{hour:02d}_3comp_spec.png"
            filepath = os.path.join(output_dir, filename)
            
            # Use generator's save method
            generator.output_dir = output_dir
            os.makedirs(os.path.join(output_dir, 'spectrograms'), exist_ok=True)
            
            saved_path = generator.save_spectrogram_image_cnn(
                f_h, t_h, Sxx_h, Sxx_d, Sxx_z,
                station, date, hour, '3comp'
            )
            
            # Move to final location
            import shutil
            if saved_path and os.path.exists(saved_path):
                final_path = os.path.join(output_dir, os.path.basename(saved_path))
                if saved_path != final_path:
                    shutil.move(saved_path, final_path)
                logger.info(f"✅ Spectrogram saved: {final_path}")
                return final_path
            
            return saved_path
            
    except ImportError as e:
        logger.error(f"Import error: {e}")
        logger.info("Make sure geomagnetic_dataset_generator_ssh_v22.py is available")
        return None
    except Exception as e:
        logger.error(f"Error generating spectrogram: {e}")
        import traceback
        traceback.print_exc()
        return None


def add_event_to_pipeline(date: str, station: str, magnitude: float, azimuth: float, 
                          spectrogram_path: str = None) -> dict:
    """
    Tambahkan event ke pipeline.
    
    Args:
        date: Tanggal event (YYYY-MM-DD)
        station: Kode stasiun
        magnitude: Nilai magnitude (0-10)
        azimuth: Nilai azimuth (0-360)
        spectrogram_path: Path ke spectrogram (optional)
        
    Returns:
        Result dictionary
    """
    try:
        from autoupdate_pipeline.src.utils import load_config
        from autoupdate_pipeline.src.data_ingestion import DataIngestion
        from autoupdate_pipeline.src.data_validator import DataValidator
        
        config = load_config()
        ingestion = DataIngestion(config)
        validator = DataValidator(config)
        
        # Convert numeric values to classes
        mag_class = validator.magnitude_to_class(magnitude)
        azi_class = validator.azimuth_to_class(azimuth)
        
        event_data = {
            'date': date,
            'station': station,
            'magnitude': mag_class,
            'azimuth': azi_class,
            'magnitude_value': magnitude,
            'azimuth_value': azimuth,
            'magnitude_class': mag_class,
            'azimuth_class': azi_class,
            'spectrogram_path': spectrogram_path
        }
        
        logger.info(f"Adding event: {station} - {date}")
        logger.info(f"  Magnitude: {magnitude} -> {mag_class}")
        logger.info(f"  Azimuth: {azimuth}° -> {azi_class}")
        if spectrogram_path:
            logger.info(f"  Spectrogram: {spectrogram_path}")
        
        result = ingestion.add_pending_event(event_data)
        
        return result
        
    except Exception as e:
        logger.error(f"Error adding event: {e}")
        import traceback
        traceback.print_exc()
        return {'success': False, 'message': str(e)}


def test_with_existing_event():
    """Test dengan event dari dataset yang sudah ada."""
    logger.info("="*60)
    logger.info("TEST: Using existing event from dataset")
    logger.info("="*60)
    
    # Cari spectrogram yang sudah ada
    dataset_path = Path(__file__).parent.parent.parent / 'dataset_unified'
    
    if not dataset_path.exists():
        logger.error(f"Dataset not found: {dataset_path}")
        return None
    
    # Cari file spectrogram
    spec_files = list(dataset_path.glob('**/*_spec.png'))
    
    if not spec_files:
        logger.error("No spectrogram files found in dataset")
        return None
    
    # Ambil satu file untuk test
    test_spec = spec_files[0]
    logger.info(f"Using existing spectrogram: {test_spec}")
    
    # Parse filename untuk mendapatkan info
    # Format: STATION_YYYYMMDD_HXX_3comp_spec.png
    filename = test_spec.stem
    parts = filename.split('_')
    
    if len(parts) >= 3:
        station = parts[0]
        date_str = parts[1]
        date = f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:8]}"
        
        # Gunakan nilai dummy untuk test
        magnitude = 5.5
        azimuth = 90.0
        
        logger.info(f"Parsed event: {station} - {date}")
        
        # Copy spectrogram ke folder pipeline
        dest_dir = Path(__file__).parent.parent / 'data' / 'spectrograms'
        dest_dir.mkdir(parents=True, exist_ok=True)
        
        import shutil
        dest_path = dest_dir / test_spec.name
        shutil.copy2(test_spec, dest_path)
        
        # Add to pipeline
        result = add_event_to_pipeline(
            date=date,
            station=station,
            magnitude=magnitude,
            azimuth=azimuth,
            spectrogram_path=str(dest_path)
        )
        
        return result
    
    return None


def main():
    parser = argparse.ArgumentParser(
        description='Add event with spectrogram to auto-update pipeline'
    )
    
    parser.add_argument('--date', '-d', type=str, help='Event date (YYYY-MM-DD)')
    parser.add_argument('--station', '-s', type=str, help='Station code')
    parser.add_argument('--hour', '-H', type=int, default=6, help='Hour of event (0-23)')
    parser.add_argument('--magnitude', '-m', type=float, help='Magnitude value (0-10)')
    parser.add_argument('--azimuth', '-a', type=float, help='Azimuth value (0-360)')
    parser.add_argument('--spectrogram', type=str, help='Path to existing spectrogram')
    parser.add_argument('--test-existing', action='store_true', 
                        help='Test with existing event from dataset')
    parser.add_argument('--generate-only', action='store_true',
                        help='Only generate spectrogram, do not add to pipeline')
    
    args = parser.parse_args()
    
    print("="*60)
    print("ADD EVENT WITH SPECTROGRAM")
    print("="*60)
    
    if args.test_existing:
        result = test_with_existing_event()
        if result:
            print(f"\n✅ Result: {result}")
        return
    
    if not args.date or not args.station:
        parser.print_help()
        print("\nExample:")
        print("  python add_event_with_spectrogram.py --date 2018-01-17 --station SCN --hour 6 --magnitude 6.2 --azimuth 45")
        return
    
    spectrogram_path = args.spectrogram
    
    # Generate spectrogram if not provided
    if not spectrogram_path:
        print(f"\n📡 Generating spectrogram from SSH server...")
        spectrogram_path = generate_spectrogram_for_event(
            station=args.station,
            date=args.date,
            hour=args.hour
        )
        
        if not spectrogram_path:
            print("❌ Failed to generate spectrogram")
            return
    
    if args.generate_only:
        print(f"\n✅ Spectrogram generated: {spectrogram_path}")
        return
    
    # Add to pipeline
    if args.magnitude is not None and args.azimuth is not None:
        print(f"\n📋 Adding event to pipeline...")
        result = add_event_to_pipeline(
            date=args.date,
            station=args.station,
            magnitude=args.magnitude,
            azimuth=args.azimuth,
            spectrogram_path=spectrogram_path
        )
        
        if result['success']:
            print(f"\n✅ Event added successfully!")
            print(f"   Event ID: {result['event_id']}")
            print(f"   Message: {result['message']}")
            
            if result.get('trigger_check', {}).get('should_trigger'):
                print(f"\n🚀 TRIGGER CONDITIONS MET!")
                print(f"   Run: python scripts/run_pipeline.py --force")
        else:
            print(f"\n❌ Failed to add event: {result['message']}")
    else:
        print(f"\n✅ Spectrogram generated: {spectrogram_path}")
        print("   To add to pipeline, provide --magnitude and --azimuth")


if __name__ == '__main__':
    main()
