
import sys
import os
import time
import shutil
import random
import logging
import pandas as pd
import subprocess
from pathlib import Path

# ==============================================================================
# CONFIGURATION
# ==============================================================================
# Paths
BASE_DIR = Path(__file__).parent.parent
ARCHIVE_ROOT = Path("d:/multi/dataset_fix") # "The Server" (Source of Truth)
ARCHIVE_IMG_DIR = ARCHIVE_ROOT / "spectrograms"
ARCHIVE_META_CSV = ARCHIVE_ROOT / "train.csv"

PENDING_DIR = BASE_DIR / "data" / "pending"
PENDING_IMG_DIR = BASE_DIR / "data" / "pending_spectrograms"

# Settings
TARGET_EVIDENCE = 5  # Berhenti setelah 5 bukti
PIPELINE_SCRIPT = BASE_DIR / "scripts" / "run_effi_update.py"

# Logging setup
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - [FORCE_SCAN] - %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger("ForceRealScan")

def setup_dirs():
    PENDING_DIR.mkdir(parents=True, exist_ok=True)
    if not PENDING_IMG_DIR.exists():
        # Jaga-jaga jika folder gambar pending terpisah, 
        # tapi biasanya sistem auto-effi bisa baca path absolut.
        PENDING_IMG_DIR.mkdir(parents=True, exist_ok=True)

def get_real_samples(count=5):
    """Mengambil sample acak dari dataset asli disertasi"""
    try:
        df = pd.read_csv(ARCHIVE_META_CSV)
        # Filter hanya yang filenya ada
        logger.info(f"Reading server archive: {len(df)} records found.")
        
        # Ambil sample acak (Stratified kalau bisa, tapi random dulu biar cepat)
        samples = df.sample(n=count)
        return samples
    except Exception as e:
        logger.error(f"Gagal membaca server archive: {e}")
        sys.exit(1)

def simulate_scanner_stream():
    """
    Mensimulasikan scanner yang mendeteksi data masuk satu per satu
    dari Server ke Local System.
    """
    setup_dirs()
    
    # 1. Fetch Data dari "Server"
    samples_df = get_real_samples(TARGET_EVIDENCE)
    
    logger.info(f"🔍 SCANNED: Detected {len(samples_df)} new valid precursors on server.")
    logger.info("⬇️  Simulating data download stream...")
    
    collected_files = []
    
    for idx, row in samples_df.iterrows():
        # Source (Server)
        src_filename = row['filename']
        src_img_path = ARCHIVE_IMG_DIR / src_filename
        
        if not src_img_path.exists():
            logger.warning(f"File missing on server: {src_filename}. Skipping.")
            continue
            
        # Destination (Local Pending)
        # Kita beri prefix FORCE_REAL agar mudah dilacak
        timestamp = int(time.time())
        dest_filename = f"REAL_{src_filename}" 
        dest_img_path = PENDING_IMG_DIR / dest_filename
        dest_csv_path = PENDING_DIR / f"evt_{timestamp}_{idx}.csv"
        
        # A. Copy Image
        shutil.copy2(src_img_path, dest_img_path)
        
        # B. Create Single Event Metadata (Format Auto-Effi)
        # Kita sesuaikan format raw archive ke format pipeline input
        single_meta = {
            'filename': [dest_filename],
            'spectrogram_path': [str(dest_img_path.absolute())], # Absolute path penting
            'station': [row.get('station', 'UNKNOWN')],
            'date': [row.get('date', '2026-01-01')],
            'magnitude_class': [row.get('magnitude_class', 'Normal')],
            'azimuth_class': [row.get('azimuth_class', 'Normal')],
            'is_precursor': [True], # Asumsi data training adalah valid events
            'ingested_at': [timestamp]
        }
        
        pd.DataFrame(single_meta).to_csv(dest_csv_path, index=False)
        
        logger.info(f"   📥 Ingested Evidence #{len(collected_files)+1}: {dest_filename}")
        collected_files.append(dest_csv_path)
        
        # Simulate Network Delay
        time.sleep(0.5)
        
        if len(collected_files) >= TARGET_EVIDENCE:
            logger.info("🛑 THRESHOLD REACHED (5 Events). Stopping Scanner.")
            break
            
    return len(collected_files)

def trigger_pipeline():
    logger.info("="*50)
    logger.info("🚀 TRIGGERING PIPELINE (AUTO-EFFI UPDATE)")
    logger.info("="*50)
    
    try:
        # Gunakan subprocess.call agar synchronous (tunggu sampai selesai)
        # atau Popen jika ingin detach. Kita pakai call biar bisa lihat lognya di sini.
        subprocess.call([sys.executable, str(PIPELINE_SCRIPT)])
    except Exception as e:
        logger.error(f"Pipeline failed to start: {e}")

if __name__ == "__main__":
    logger.info("--- FORCE REAL SCAN SIMULATION STARTED ---")
    
    # Cleaning pending folder first (Optional, agar bersih)
    for f in PENDING_DIR.glob("*.csv"):
        os.remove(f)
    for f in PENDING_IMG_DIR.glob("*REAL_*"): # Hanya hapus yg bekas simulasi
        try: os.remove(f) 
        except: pass
        
    count = simulate_scanner_stream()
    
    if count >= TARGET_EVIDENCE:
        trigger_pipeline()
    else:
        logger.warning("Not enough data collected to trigger pipeline.")
