import pandas as pd
import os
from datetime import datetime

# Read event list
event_df = pd.read_excel('intial/event_list.xlsx')
print(f"Total events in list: {len(event_df)}")

# Read processed metadata
metadata_df = pd.read_csv('dataset_spectrogram/metadata/dataset_metadata.csv')
print(f"Total events processed: {len(metadata_df)}")

# Create success/failure report
processed_events = set()
for _, row in metadata_df.iterrows():
    key = f"{row['station']}_{row['date']}"
    processed_events.add(key)

success_list = []
failure_list = []

for idx, row in event_df.iterrows():
    station = row['Stasiun']
    date = pd.to_datetime(row['Tanggal']).strftime('%Y-%m-%d')
    key = f"{station}_{date}"
    
    event_info = {
        'No': row['No'],
        'Station': station,
        'Date': date,
        'Azimuth': row['Azm'],
        'Magnitude': row['Mag']
    }
    
    if key in processed_events:
        success_list.append(event_info)
    else:
        failure_list.append(event_info)

# Save to Excel
with pd.ExcelWriter('LAPORAN_PROSES_DATASET.xlsx', engine='openpyxl') as writer:
    # Summary sheet
    summary_data = {
        'Metric': [
            'Total Events in List',
            'Successfully Processed',
            'Failed to Process',
            'Success Rate (%)',
            'Processing Date'
        ],
        'Value': [
            len(event_df),
            len(success_list),
            len(failure_list),
            f"{len(success_list)/len(event_df)*100:.2f}%",
            datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        ]
    }
    pd.DataFrame(summary_data).to_excel(writer, sheet_name='Summary', index=False)
    
    # Success sheet
    if success_list:
        success_df = pd.DataFrame(success_list)
        success_df.to_excel(writer, sheet_name='Berhasil', index=False)
    
    # Failure sheet
    if failure_list:
        failure_df = pd.DataFrame(failure_list)
        failure_df.to_excel(writer, sheet_name='Gagal', index=False)
    
    # Station analysis
    station_success = {}
    station_total = {}
    
    for event in success_list:
        station = event['Station']
        station_success[station] = station_success.get(station, 0) + 1
    
    for _, row in event_df.iterrows():
        station = row['Stasiun']
        if pd.notna(station):
            station_total[station] = station_total.get(station, 0) + 1
    
    station_analysis = []
    for station in sorted(station_total.keys()):
        success_count = station_success.get(station, 0)
        total_count = station_total[station]
        station_analysis.append({
            'Station': station,
            'Total Events': total_count,
            'Processed': success_count,
            'Failed': total_count - success_count,
            'Success Rate (%)': f"{success_count/total_count*100:.1f}%"
        })
    
    pd.DataFrame(station_analysis).to_excel(writer, sheet_name='Per Stasiun', index=False)

print("\n✅ Laporan berhasil dibuat: LAPORAN_PROSES_DATASET.xlsx")
print(f"\n📊 RINGKASAN:")
print(f"   Total Events: {len(event_df)}")
print(f"   Berhasil: {len(success_list)} ({len(success_list)/len(event_df)*100:.2f}%)")
print(f"   Gagal: {len(failure_list)} ({len(failure_list)/len(event_df)*100:.2f}%)")
