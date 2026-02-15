
import os
import pandas as pd
import glob
from datetime import datetime
import warnings

# Suppress warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

def normalize_columns(df):
    # Mapping of possible column names to standard names
    col_map = {
        'Date time': 'datetime',
        'Time': 'datetime',
        'Origin Time': 'datetime',
        'date': 'datetime',
        'Event ID': 'event_id',
        'EventID': 'event_id',
        'ID': 'event_id',
        'Latitude': 'latitude',
        'Lat': 'latitude',
        'Longitude': 'longitude',
        'Lon': 'longitude',
        'Magnitude': 'magnitude',
        'Mag': 'magnitude',
        'Depth (km)': 'depth',
        'Depth': 'depth',
        'Mag Type': 'mag_type',
        'Type': 'mag_type',
        'Location': 'location',
        'remark': 'location'
    }
    
    # Normalize headers (strip whitespace, lower case)
    df.columns = [c.strip() for c in df.columns]
    
    # Rename columns
    new_cols = {}
    for c in df.columns:
        for k, v in col_map.items():
            if c.lower() == k.lower():
                new_cols[c] = v
                break
    
    df = df.rename(columns=new_cols)
    
    # Ensure required columns exist
    required = ['event_id', 'datetime', 'latitude', 'longitude', 'magnitude', 'depth']
    for req in required:
        if req not in df.columns:
            # Try to recover if possible or fill with NaN
            df[req] = None
            
    return df

def parse_dates(date_str):
    # Try multiple formats
    formats = [
        '%Y-%m-%dT%H:%M:%S.%fZ', # ISO 8601 with Z
        '%Y-%m-%dT%H:%M:%S',     # ISO 8601 no Z
        '%Y-%m-%d %H:%M:%S',     # Standard
        '%d-%m-%Y %H:%M:%S',     # European
        '%m/%d/%Y %H:%M:%S'      # US
    ]
    
    if pd.isna(date_str):
        return None
        
    for fmt in formats:
        try:
            return pd.to_datetime(date_str, format=fmt)
        except:
            continue
            
    # Fallback to pandas automatic parsing
    try:
        return pd.to_datetime(date_str)
    except:
        return None

def compile_catalog(base_dir, output_file):
    print(f"Scanning directory: {base_dir}")
    
    all_files = glob.glob(os.path.join(base_dir, "**", "*.csv"), recursive=True)
    print(f"Found {len(all_files)} CSV files.")
    
    dfs = []
    
    for f in all_files:
        # Skip the output file itself if it exists or other merged files
        if "merged" in f.lower() or "repository_complete" in f.lower():
            continue
            
        try:
            # Read CSV - try different encodings/separators
            try:
                df = pd.read_csv(f)
                # Check for malformed single-column CSV (double-quoted/escaped)
                if len(df.columns) == 1 and isinstance(df.columns[0], str) and ('"Event' in df.columns[0] or '"Date' in df.columns[0]):
                    # It's likely a malformed CSV where the whole line is quoted
                    print(f"  Detected malformed CSV (single column): {os.path.basename(f)}. Attempting repair...")
                    with open(f, 'r') as file:
                        lines = file.readlines()
                    
                    from io import StringIO
                    repaired_lines = []
                    for line in lines:
                        line = line.strip()
                        if line.startswith('"') and line.endswith('"'):
                            # Remove outer quotes and unescape inner quotes
                            # "No,""Event ID""..." -> No,"Event ID"...
                            # Be careful: '""' -> '"'
                            # But wait, logic: "A,""B""" -> A,"B" ?
                            # If content is: "1,""id""..."
                            # Unwrapped: 1,""id""...
                            # Replaced "" -> ": 1,"id"...
                            # This seems correct for standard CSV
                            inner = line[1:-1]
                            repaired = inner.replace('""', '"')
                            repaired_lines.append(repaired)
                        else:
                            repaired_lines.append(line)
                    
                    try:
                        df = pd.read_csv(StringIO('\n'.join(repaired_lines)))
                        print(f"  Repair successful. Columns: {list(df.columns)}")
                    except Exception as e:
                        print(f"  Repair failed: {e}. Falling back to single column df.")
            except:
                df = pd.read_csv(f, sep=';')
            
            # Normalize
            df = normalize_columns(df)
            
            # Basic validation: must have datetime and magnitude
            if df['datetime'].notna().sum() > 0:
                dfs.append(df)
                # print(f"  Loaded {len(df)} events from {os.path.basename(f)}")
            else:
                print(f"  Warning: No valid data in {os.path.basename(f)}")
                
        except Exception as e:
            print(f"  Error reading {f}: {e}")
            
    if not dfs:
        print("No valid data found!")
        return
        
    # Merge
    print("Merging dataframes...")
    full_df = pd.concat(dfs, ignore_index=True)
    
    # Clean Datetime
    print("Parsing dates using robust parser...")
    # Use helper to handle multiple formats
    full_df['datetime'] = full_df['datetime'].apply(parse_dates)
    # Ensure UTC
    full_df['datetime'] = pd.to_datetime(full_df['datetime'], utc=True, errors='coerce')
    
    # Clean numeric columns
    for col in ['latitude', 'longitude', 'magnitude', 'depth']:
        full_df[col] = pd.to_numeric(full_df[col], errors='coerce')
    
    # Drop rows with invalid key data
    before_drop = len(full_df)
    full_df = full_df.dropna(subset=['datetime', 'latitude', 'longitude', 'magnitude'])
    print(f"Dropped {before_drop - len(full_df)} rows with missing key data.")
    
    # Remove duplicates
    # Priority: Keep row with most info (least NaNs)
    full_df['nan_count'] = full_df.isna().sum(axis=1)
    full_df = full_df.sort_values('nan_count')
    
    # Deduplicate by Event ID if available
    if 'event_id' in full_df.columns and full_df['event_id'].notna().sum() > 0:
        print("Deduplicating by Event ID...")
        # Only dedupe non-null IDs
        ids = full_df[full_df['event_id'].notna()]
        no_ids = full_df[full_df['event_id'].isna()]
        
        ids = ids.drop_duplicates(subset=['event_id'], keep='first')
        full_df = pd.concat([ids, no_ids])
    
    # Deduplicate by Space-Time (to catch same events with different/missing IDs)
    # Round time to nearest minute, location to 2 decimal places for duplicate checking
    print("Deduplicating by Spacetime...")
    full_df['temp_time'] = full_df['datetime'].dt.round('1min')
    full_df['temp_lat'] = full_df['latitude'].round(2)
    full_df['temp_lon'] = full_df['longitude'].round(2)
    
    full_df = full_df.drop_duplicates(subset=['temp_time', 'temp_lat', 'temp_lon'], keep='first')
    full_df = full_df.drop(columns=['nan_count', 'temp_time', 'temp_lat', 'temp_lon'])
    
    # Sort
    full_df = full_df.sort_values('datetime', ascending=False)
    
    # Rename columns to standard output format
    final_rename = {
        'datetime': 'datetime',
        'latitude': 'Latitude',
        'longitude': 'Longitude',
        'magnitude': 'Magnitude',
        'depth': 'Depth',
        'event_id': 'EventID',
        'mag_type': 'MagType',
        'location': 'Location'
    }
    full_df = full_df.rename(columns=final_rename)
    
    # Select final columns
    cols = ['EventID', 'datetime', 'Latitude', 'Longitude', 'Magnitude', 'Depth', 'MagType', 'Location']
    # Add any extra columns that might be useful? No, keep it clean.
    full_df = full_df[cols]
    
    # Save
    full_df.to_csv(output_file, index=False)
    print(f"Successfully saved {len(full_df)} events to {output_file}")
    
    # Stats
    print("\nCatalog Statistics:")
    print(f"  Date Range: {full_df['datetime'].min()} to {full_df['datetime'].max()}")
    print(f"  Magnitude Range: {full_df['Magnitude'].min()} to {full_df['Magnitude'].max()}")
    print(f"  Events per Year:")
    print(full_df['datetime'].dt.year.value_counts().sort_index())

if __name__ == "__main__":
    base_dir = r"d:\multi\repository gempa 2018-2025"
    output_file = r"d:\multi\earthquake_catalog_2018_2025_merged_robust.csv"
    compile_catalog(base_dir, output_file)
