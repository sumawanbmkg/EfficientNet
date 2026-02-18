#!/usr/bin/env python3
"""
Generate Research Highlights file for IEEE TGRS submission.
Format: Word document (.docx) with bullet points only.
"""

from docx import Document
from docx.shared import Pt, Inches
from docx.enum.text import WD_ALIGN_PARAGRAPH
from pathlib import Path

# Create document
doc = Document()

# Set default font
style = doc.styles['Normal']
font = style.font
font.name = 'Times New Roman'
font.size = Pt(12)

# Research Highlights - bullet points only (no title as per IEEE guidelines)
highlights = [
    "VGG16 and EfficientNet-B0 comparison for ULF earthquake precursor detection.",
    "Achieved 98.68% magnitude accuracy using 7 years of Indonesian BMKG data.",
    "LOEO cross-validation ensures robust spatial-temporal generalization.",
    "Grad-CAM confirms model focus on physically meaningful 0.001–0.01 Hz ULF bands.",
    "EfficientNet-B0 offers 2.5x faster inference for operational early warning."
]

# Add bullet points
for highlight in highlights:
    para = doc.add_paragraph(highlight, style='List Bullet')
    para.paragraph_format.space_after = Pt(6)

# Save to publication/paper folder
output_dir = Path('publication/paper')
output_dir.mkdir(parents=True, exist_ok=True)

# Save with recommended filename
output_path = output_dir / 'Highlights_Sumawan_TGRS.docx'
doc.save(output_path)

# Also save to publication_package
pkg_dir = Path('publication_package')
pkg_dir.mkdir(exist_ok=True)
doc.save(pkg_dir / 'Highlights_Sumawan_TGRS.docx')

print("=" * 60)
print("Research Highlights file generated successfully!")
print("=" * 60)
print(f"\nOutput files:")
print(f"  - {output_path}")
print(f"  - {pkg_dir / 'Highlights_Sumawan_TGRS.docx'}")
print("\nHighlights content:")
for i, h in enumerate(highlights, 1):
    print(f"  {i}. {h}")
print("\n" + "=" * 60)
print("SUBMISSION INSTRUCTIONS:")
print("=" * 60)
print("1. Upload as separate file in ScholarOne")
print("2. Select document category: 'Research Highlights'")
print("3. File name: Highlights_Sumawan_TGRS.docx")
print("=" * 60)
