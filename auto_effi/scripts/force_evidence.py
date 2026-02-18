import sys
import os
import shutil
import random
import logging
import pandas as pd
from datetime import datetime
from pathlib import Path
from PIL import Image

# Setup paths
base_dir = Path(__file__).parent.parent
pending_csv_dir = base_dir / "data" / "pending"
pending_spec_dir = base_dir / "data" / "pending_spectrograms"

# Ensure directories exist
pending_csv_dir.mkdir(parents=True, exist_ok=True)
pending_spec_dir.mkdir(parents=True, exist_ok=True)

# Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("force_evidence")

def generate_force_evidence(count=5):
    logger.info(f"🚨 FORCING EVIDENCE GENERATION: Creating {count} precursor events...")
    
    stations = ['GTO', 'SCN', 'TDR', 'PLR', 'KND']
    magnitudes = ['Moderate', 'Medium', 'Large']
    azimuths = ['N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW']
    
    generated_files = []
    
    for i in range(count):
        station = stations[i % len(stations)]
        date_str = datetime.now().strftime('%Y%m%d')
        evidence_id = f"evt_FORCE_{station}_{date_str}_{i}_{int(datetime.now().timestamp())}"
        
        # 1. Create Dummy Spectrogram (Blue Noise Image)
        # In real life this would be the actual spectrogram
        img = Image.new('RGB', (224, 224), color=(random.randint(0, 50), random.randint(0, 50), random.randint(100, 255)))
        spec_filename = f"{evidence_id}.png"
        spec_path = pending_spec_dir / spec_filename
        img.save(spec_path)
        logger.info(f"   [+] Generated spectrogram: {spec_filename}")
        
        # 2. Create Metadata CSV
        # Simulating a POSITIVE detection
        mag_cls = random.choice(magnitudes)
        azi_cls = random.choice(azimuths)
        
        evidence_data = {
            'filename': [spec_filename],
            'spectrogram_path': [str(spec_path)],
            'station': [station],
            'date': [datetime.now().strftime('%Y-%m-%d')],
            'magnitude_class': [mag_cls],
            'azimuth_class': [azi_cls],
            'confidence_mag': [random.uniform(75.0, 99.0)],
            'confidence_azi': [random.uniform(60.0, 95.0)],
            'is_precursor': [True],
            'captured_at': [datetime.now().isoformat()],
            'forced_generation': [True]
        }
        
        df = pd.DataFrame(evidence_data)
        csv_path = pending_csv_dir / f"{evidence_id}.csv"
        df.to_csv(csv_path, index=False)
        logger.info(f"   [+] Generated metadata: {csv_path.name}")
        
        generated_files.append(csv_path)
        
    logger.info("✅ Forced Evidence Generation Complete.")
    logger.info(f"   Total Pending Files: {len(list(pending_csv_dir.glob('*.csv')))}")
    
    # 3. Trigger Pipeline
    logger.info("🔄 Triggering Auto-Effi Pipeline...")
    try:
        import subprocess
        script_path = base_dir / "scripts" / "run_effi_update.py"
        subprocess.Popen([sys.executable, str(script_path)], cwd=str(base_dir))
        logger.info("   -> Auto-Update Script subprocess started.")
    except Exception as e:
        logger.error(f"   Failed to trigger auto-update: {e}")

if __name__ == "__main__":
    generate_force_evidence(5)
