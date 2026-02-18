"""
Test Binary Reading - Validate data parsing
Menggunakan format dari read_binary_data.md
"""
import os
import sys
import struct
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

# Add intial to path
sys.path.insert(0, 'intial')
from geomagnetic_fetcher import GeomagneticDataFetcher


def parse_binary_simple(binary_data, station='GTO'):
    """
    Parse binary file menggunakan format dari read_binary_data.md
    
    Format:
    - Header: 32 bytes
    - Records: 17 bytes each × 86,400 (sequential, no timestamps)
    - Each record: H(2), D(2), Z(2), IX(2), IY(2), TempS(2), TempP(2), Voltage(2), Spare(1)
    """
    # Station baselines
    baselines = {
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
    }
    baseline = baselines.get(station, {'H': 40000, 'Z': 30000})
    
    print(f"\n{'='*80}")
    print(f"PARSING BINARY DATA - {station}")
    print(f"{'='*80}")
    print(f"Binary data size: {len(binary_data)} bytes")
    print(f"Baseline: H={baseline['H']} nT, Z={baseline['Z']} nT")
    
    # Skip 32-byte header
    header_size = 32
    record_size = 17
    data_start = binary_data[header_size:]
    
    print(f"Header size: {header_size} bytes")
    print(f"Record size: {record_size} bytes")
    print(f"Data section size: {len(data_start)} bytes")
    print(f"Expected records: {len(data_start) // record_size}")
    
    # Parse records
    h_values = []
    d_values = []
    z_values = []
    ix_values = []
    iy_values = []
    
    max_records = min(86400, len(data_start) // record_size)
    
    for i in range(max_records):
        offset = i * record_size
        record = data_start[offset:offset+record_size]
        
        if len(record) < record_size:
            break
        
        # Parse components (little-endian, signed int16)
        h_dev = struct.unpack('<h', record[0:2])[0] * 0.1
        d_dev = struct.unpack('<h', record[2:4])[0] * 0.1
        z_dev = struct.unpack('<h', record[4:6])[0] * 0.1
        ix_val = struct.unpack('<h', record[6:8])[0] * 0.01
        iy_val = struct.unpack('<h', record[8:10])[0] * 0.01
        
        # Add baseline to H and Z
        h_values.append(baseline['H'] + h_dev)
        d_values.append(d_dev)
        z_values.append(baseline['Z'] + z_dev)
        ix_values.append(ix_val)
        iy_values.append(iy_val)
    
    h_array = np.array(h_values)
    d_array = np.array(d_values)
    z_array = np.array(z_values)
    
    print(f"\n{'='*80}")
    print(f"PARSING RESULTS")
    print(f"{'='*80}")
    print(f"Records parsed: {len(h_values)}")
    print(f"\nH Component:")
    print(f"  Min: {np.min(h_array):.2f} nT")
    print(f"  Max: {np.max(h_array):.2f} nT")
    print(f"  Mean: {np.mean(h_array):.2f} nT")
    print(f"  Std: {np.std(h_array):.2f} nT")
    print(f"\nD Component:")
    print(f"  Min: {np.min(d_array):.2f} nT")
    print(f"  Max: {np.max(d_array):.2f} nT")
    print(f"  Mean: {np.mean(d_array):.2f} nT")
    print(f"  Std: {np.std(d_array):.2f} nT")
    print(f"\nZ Component:")
    print(f"  Min: {np.min(z_array):.2f} nT")
    print(f"  Max: {np.max(z_array):.2f} nT")
    print(f"  Mean: {np.mean(z_array):.2f} nT")
    print(f"  Std: {np.std(z_array):.2f} nT")
    
    # Validation checks
    print(f"\n{'='*80}")
    print(f"VALIDATION CHECKS")
    print(f"{'='*80}")
    
    checks_passed = 0
    checks_total = 0
    
    # Check 1: Coverage
    checks_total += 1
    if len(h_values) >= 3600:  # At least 1 hour
        print(f"[OK] Coverage: {len(h_values)} samples (>= 3600)")
        checks_passed += 1
    else:
        print(f"[FAIL] Coverage: {len(h_values)} samples (< 3600)")
    
    # Check 2: H mean (should be near baseline)
    checks_total += 1
    h_mean = np.mean(h_array)
    if baseline['H'] - 2000 <= h_mean <= baseline['H'] + 2000:
        print(f"[OK] H mean: {h_mean:.2f} nT (within {baseline['H']} +/- 2000)")
        checks_passed += 1
    else:
        print(f"[FAIL] H mean: {h_mean:.2f} nT (outside {baseline['H']} +/- 2000)")
    
    # Check 3: Z mean (should be near baseline)
    checks_total += 1
    z_mean = np.mean(z_array)
    if baseline['Z'] - 2000 <= z_mean <= baseline['Z'] + 2000:
        print(f"[OK] Z mean: {z_mean:.2f} nT (within {baseline['Z']} +/- 2000)")
        checks_passed += 1
    else:
        print(f"[FAIL] Z mean: {z_mean:.2f} nT (outside {baseline['Z']} +/- 2000)")
    
    # Check 4: D mean (should be near 0)
    checks_total += 1
    d_mean = np.mean(d_array)
    if -1000 <= d_mean <= 1000:
        print(f"[OK] D mean: {d_mean:.2f} nT (within +/-1000)")
        checks_passed += 1
    else:
        print(f"[FAIL] D mean: {d_mean:.2f} nT (outside +/-1000)")
    
    # Check 5: Std dev (should be reasonable)
    checks_total += 1
    h_std = np.std(h_array)
    if 100 <= h_std <= 5000:
        print(f"[OK] H std: {h_std:.2f} nT (100-5000 range)")
        checks_passed += 1
    else:
        print(f"[FAIL] H std: {h_std:.2f} nT (outside 100-5000 range)")
    
    print(f"\n{'='*80}")
    print(f"VALIDATION SUMMARY: {checks_passed}/{checks_total} checks passed")
    print(f"{'='*80}\n")
    
    return {
        'H': h_array,
        'D': d_array,
        'Z': z_array,
        'IX': np.array(ix_values),
        'IY': np.array(iy_values),
        'Time': np.arange(len(h_values)),
        'validation_passed': checks_passed >= 4  # At least 4/5 checks
    }


def test_with_ssh_data():
    """Test dengan data dari SSH server"""
    print("\n" + "="*80)
    print("TESTING BINARY READING WITH SSH DATA")
    print("="*80)
    
    # Test dengan 1 stasiun saja
    test_cases = [
        {'date': '2018-01-17', 'station': 'SCN'},
    ]
    
    with GeomagneticDataFetcher() as fetcher:
        for test_case in test_cases:
            date = datetime.strptime(test_case['date'], '%Y-%m-%d')
            station = test_case['station']
            
            print(f"\n{'='*80}")
            print(f"TEST CASE: {station} - {test_case['date']}")
            print(f"{'='*80}")
            
            # Fetch binary data
            yy = date.year % 100
            mm = date.month
            dd = date.day
            filename = f"S{yy:02d}{mm:02d}{dd:02d}.{station}"
            remote_path = f"/home/precursor/SEISMO/DATA/{station}/SData/{yy:02d}{mm:02d}/{filename}"
            
            try:
                from io import BytesIO
                with BytesIO() as file_buffer:
                    fetcher.sftp_client.getfo(remote_path, file_buffer)
                    file_buffer.seek(0)
                    binary_data = file_buffer.read()
                
                print(f"[OK] Downloaded: {len(binary_data)} bytes")
                
                # Parse dengan format baru
                result = parse_binary_simple(binary_data, station)
                
                if result['validation_passed']:
                    print(f"\n[OK] VALIDATION PASSED for {station} - {test_case['date']}")
                    
                    # Plot untuk visual check
                    plot_data(result, station, test_case['date'])
                else:
                    print(f"\n[FAIL] VALIDATION FAILED for {station} - {test_case['date']}")
                
            except Exception as e:
                print(f"[ERROR] Error: {e}")
                import traceback
                traceback.print_exc()


def plot_data(data, station, date_str):
    """Plot data untuk visual inspection"""
    fig, axes = plt.subplots(3, 1, figsize=(12, 8))
    
    time_hours = data['Time'] / 3600.0
    
    # H component
    axes[0].plot(time_hours, data['H'], 'b-', linewidth=0.5)
    axes[0].set_ylabel('H (nT)')
    axes[0].set_title(f'{station} - {date_str} - Raw Data')
    axes[0].grid(True, alpha=0.3)
    
    # D component
    axes[1].plot(time_hours, data['D'], 'r-', linewidth=0.5)
    axes[1].set_ylabel('D (nT)')
    axes[1].grid(True, alpha=0.3)
    
    # Z component
    axes[2].plot(time_hours, data['Z'], 'g-', linewidth=0.5)
    axes[2].set_ylabel('Z (nT)')
    axes[2].set_xlabel('Time (hours)')
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save plot
    filename = f"test_binary_{station}_{date_str.replace('-', '')}.png"
    plt.savefig(filename, dpi=150)
    print(f"\n[PLOT] Plot saved: {filename}")
    plt.close()


if __name__ == '__main__':
    test_with_ssh_data()
