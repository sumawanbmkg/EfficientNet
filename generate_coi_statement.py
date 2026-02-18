#!/usr/bin/env python3
"""
Generate Conflict of Interest Statement PDF for IEEE TGRS submission.
"""

from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_JUSTIFY
from pathlib import Path

# Output path
output_dir = Path('publication/paper')
output_dir.mkdir(parents=True, exist_ok=True)
output_path = output_dir / 'Conflict_of_Interest_Statement.pdf'

# Create document
doc = SimpleDocTemplate(
    str(output_path),
    pagesize=letter,
    rightMargin=1*inch,
    leftMargin=1*inch,
    topMargin=1*inch,
    bottomMargin=1*inch
)

# Styles
styles = getSampleStyleSheet()
title_style = ParagraphStyle(
    'Title',
    parent=styles['Heading1'],
    fontSize=16,
    alignment=TA_CENTER,
    spaceAfter=20
)
subtitle_style = ParagraphStyle(
    'Subtitle',
    parent=styles['Normal'],
    fontSize=11,
    alignment=TA_CENTER,
    spaceAfter=30,
    textColor=colors.grey
)
heading_style = ParagraphStyle(
    'Heading',
    parent=styles['Heading2'],
    fontSize=12,
    spaceBefore=15,
    spaceAfter=10
)
body_style = ParagraphStyle(
    'Body',
    parent=styles['Normal'],
    fontSize=11,
    alignment=TA_JUSTIFY,
    spaceAfter=12,
    leading=14
)
author_style = ParagraphStyle(
    'Author',
    parent=styles['Normal'],
    fontSize=10,
    spaceAfter=6
)

# Content
story = []

# Title
story.append(Paragraph("CONFLICT OF INTEREST STATEMENT", title_style))
story.append(Paragraph(
    "Manuscript: Deep Learning-Based Earthquake Precursor Detection from Geomagnetic Data: "
    "A Comparative Study of VGG16 and EfficientNet Architectures",
    subtitle_style
))

# Journal info
story.append(Paragraph("Submitted to: IEEE Transactions on Geoscience and Remote Sensing", body_style))
story.append(Spacer(1, 20))

# Declaration
story.append(Paragraph("Declaration of Competing Interests", heading_style))
story.append(Paragraph(
    "The authors declare that they have no known competing financial interests or personal "
    "relationships that could have appeared to influence the work reported in this paper.",
    body_style
))

story.append(Spacer(1, 15))

# Funding disclosure
story.append(Paragraph("Funding Disclosure", heading_style))
story.append(Paragraph(
    "This research did not receive any specific grant from funding agencies in the public, "
    "commercial, or not-for-profit sectors. The work was conducted as part of the doctoral "
    "research program at Sepuluh Nopember Institute of Technology (ITS), Surabaya, Indonesia.",
    body_style
))

story.append(Spacer(1, 15))

# Employment disclosure
story.append(Paragraph("Employment Disclosure", heading_style))
story.append(Paragraph(
    "Sumawan and Muhamad Syirojudin are employees of the Meteorological, Climatological and "
    "Geophysical Agency (BMKG), Indonesia. The views expressed in this paper are those of the "
    "authors and do not necessarily represent the official position of BMKG.",
    body_style
))

story.append(Spacer(1, 15))

# Data disclosure
story.append(Paragraph("Data Access Disclosure", heading_style))
story.append(Paragraph(
    "The geomagnetic data used in this study were provided by BMKG Indonesia under standard "
    "institutional data-sharing agreements for academic research purposes.",
    body_style
))

story.append(Spacer(1, 30))

# Author signatures section
story.append(Paragraph("Author Confirmations", heading_style))

# Authors table
authors_data = [
    ["Author Name", "Affiliation", "Signature", "Date"],
    ["Sumawan", "ITS Surabaya / BMKG", "_____________", "_____________"],
    ["Bambang L. Widjiantoro", "ITS Surabaya", "_____________", "_____________"],
    ["Katherin Indriawati", "ITS Surabaya", "_____________", "_____________"],
    ["Muhamad Syirojudin", "BMKG Jakarta", "_____________", "_____________"],
]

table = Table(authors_data, colWidths=[2*inch, 2*inch, 1.2*inch, 1*inch])
table.setStyle(TableStyle([
    ('BACKGROUND', (0, 0), (-1, 0), colors.lightgrey),
    ('TEXTCOLOR', (0, 0), (-1, 0), colors.black),
    ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
    ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
    ('FONTSIZE', (0, 0), (-1, 0), 10),
    ('FONTSIZE', (0, 1), (-1, -1), 9),
    ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
    ('TOPPADDING', (0, 1), (-1, -1), 8),
    ('BOTTOMPADDING', (0, 1), (-1, -1), 8),
    ('GRID', (0, 0), (-1, -1), 0.5, colors.black),
]))

story.append(table)

story.append(Spacer(1, 30))

# Corresponding author
story.append(Paragraph("Corresponding Author Contact", heading_style))
story.append(Paragraph("<b>Name:</b> Sumawan", author_style))
story.append(Paragraph("<b>Email:</b> sumawanbmkg@gmail.com", author_style))
story.append(Paragraph("<b>ORCID:</b> 0009-0005-5301-6414", author_style))
story.append(Paragraph(
    "<b>Address:</b> Department of Engineering Physics, Sepuluh Nopember Institute of Technology, "
    "Kampus ITS Sukolilo, Surabaya 60111, Indonesia",
    author_style
))

# Build PDF
doc.build(story)

# Also save to publication_package
import shutil
pkg_dir = Path('publication_package')
pkg_dir.mkdir(exist_ok=True)
shutil.copy(output_path, pkg_dir / 'Conflict_of_Interest_Statement.pdf')

print("=" * 60)
print("Conflict of Interest Statement PDF generated successfully!")
print("=" * 60)
print(f"\nOutput files:")
print(f"  - {output_path}")
print(f"  - {pkg_dir / 'Conflict_of_Interest_Statement.pdf'}")
print("\nContent includes:")
print("  - Declaration of Competing Interests")
print("  - Funding Disclosure")
print("  - Employment Disclosure")
print("  - Data Access Disclosure")
print("  - Author Confirmation Table (with signature lines)")
print("  - Corresponding Author Contact")
