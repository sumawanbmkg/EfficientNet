#!/usr/bin/env python3
"""Extract text from proposal PDF"""

import PyPDF2
from pathlib import Path

def extract_pdf_text(pdf_path):
    """Extract text from PDF file"""
    try:
        with open(pdf_path, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            text = ""
            for i, page in enumerate(reader.pages):
                page_text = page.extract_text()
                if page_text:
                    text += f"\n\n=== PAGE {i+1} ===\n\n"
                    text += page_text
            return text
    except Exception as e:
        return f"Error: {e}"

if __name__ == '__main__':
    pdf_path = Path('disertasi/mawan-proposal-final.pdf')
    
    if pdf_path.exists():
        text = extract_pdf_text(pdf_path)
        
        # Save to text file
        output_path = Path('disertasi/proposal_extracted.txt')
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(text)
        
        print(f"Extracted {len(text)} characters")
        print(f"Saved to: {output_path}")
        print("\n" + "="*70)
        print("CONTENT PREVIEW (first 5000 chars):")
        print("="*70)
        print(text[:5000])
    else:
        print(f"File not found: {pdf_path}")
        # List files in disertasi folder
        print("\nFiles in disertasi folder:")
        for f in Path('disertasi').iterdir():
            print(f"  - {f.name}")
