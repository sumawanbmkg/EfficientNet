#!/usr/bin/env python3
"""
Prekursor Continuous Scanner (Simulation)
This script simulates a continuous monitoring process.
It randomly selects dates and stations to scan, looking for precursors.
If a precursor is found, it saves the evidence.
Once 5 evidences are collected, it triggers the Auto-Effi update pipeline.
"""

import sys
import time
import random
import logging
from datetime import datetime, timedelta
from pathlib import Path

# Add project root to path (d:\multi)
# Script is in d:\multi\auto_effi\scripts -> parent.parent.parent is d:\multi
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from prekursor_scanner_production import PrekursorScannerProduction

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("continuous_scanner")

def run_simulation():
    scanner = PrekursorScannerProduction()
    
    # Simulation Parameters
    start_date = datetime(2024, 1, 1)
    end_date = datetime(2024, 12, 31)
    stations = list(scanner.stations.keys())
    
    evidence_count = 0
    target_evidence = 5 # Configured in pipeline_config.yaml
    
    logger.info("="*60)
    logger.info("📡 CONTINUOUS MONITORING STARTED")
    logger.info(f"   Target: Collect {target_evidence} precursor evidences to trigger update")
    logger.info("="*60)
    
    try:
        while True:
            # Random date and station selection
            random_days = random.randint(0, (end_date - start_date).days)
            scan_date = start_date + timedelta(days=random_days)
            station = random.choice(stations)
            
            logger.info(f"\n🔍 Scanning {station} on {scan_date.date()}...")
            
            # Run Scan with Auto-Update Trigger enabled
            # The scanner itself handles saving evidence if precursor is found
            # We set trigger_auto_update=True so IF it finds one, it tries to update
            # But run_effi_update.py will actually STOP if evidence < 5
            
            result = scanner.scan(
                date=scan_date, 
                station_code=station, 
                save_results=False, # Don't flood disk with normal scans
                trigger_auto_update=True 
            )
            
            if result and result['predictions']['is_precursor']:
                evidence_count += 1
                logger.info(f"⚠️  EVIDENCE COMPLETED! Total: {evidence_count}/{target_evidence}")
                
                if evidence_count >= target_evidence:
                    logger.info("🚀 THRESHOLD REACHED! Auto-Update Pipeline should be running now...")
                    # In a real loop, we might pause here or reset counter
                    # For simulation, we break after triggering once
                    # time.sleep(60) 
            else:
                logger.info("✅ Normal condition. Continuing monitor...")
            
            # Simulate interval
            time.sleep(2) 
            
    except KeyboardInterrupt:
        logger.info("\n🛑 Monitoring stopped by user.")

if __name__ == "__main__":
    run_simulation()
