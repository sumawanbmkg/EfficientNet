"""
Regenerate dataset dengan format CNN yang benar:
- 224x224 pixels
- RGB format
- NO axis, NO text, NO labels
- Menggunakan binary reading yang benar
"""

import os
import sys

# Jalankan generator v21 yang sudah benar
print("="*80)
print("REGENERATING DATASET WITH CORRECT CNN FORMAT")
print("="*80)
print()
print("Format:")
print("  - Size: 224x224 pixels (CNN standard)")
print("  - Mode: RGB")
print("  - NO axis, NO text, NO labels")
print("  - Binary reading: FIXED (32-byte header, 17-byte records, baseline)")
print()
print("="*80)
print()

# Import dan jalankan generator v21
from geomagnetic_dataset_generator_ssh_v21 import GeomagneticDatasetGeneratorSSH_V21

# Create generator
generator = GeomagneticDatasetGeneratorSSH_V21(output_dir='dataset_spectrogram_ssh')

# Process events
metadata_df, success_list, failure_list, skipped_list = generator.process_event_list(
    event_file='intial/event_list.xlsx',
    max_events=None  # Process all events
)

print()
print("="*80)
print("REGENERATION COMPLETE!")
print("="*80)
print(f"Newly processed: {len(success_list)}")
print(f"Skipped (already correct format): {len(skipped_list)}")
print(f"Failed: {len(failure_list)}")
print(f"Total in dataset: {len(metadata_df)}")
print()
print("Dataset is now ready for CNN training!")
print("All images are 224x224 RGB without axis/text")
