"""
Dump actual OCR-extracted text from each bank PDF so we can see exact lines.
"""
import sys, os, logging
logging.basicConfig(level=logging.WARNING)
sys.path.insert(0, os.path.dirname(__file__))
from document_parser import DocumentParser

PDF_DIR = os.path.join(os.path.dirname(__file__), "bank statments")
PDFS = {
    "BMO":          "BMO.pdf",
    "FifthThird":   "FifthThird 53.pdf",
    "US_Bank":      "US Bank.pdf",
    "AMEX":         "AMEX.pdf",
}

dp = DocumentParser()
for label, filename in PDFS.items():
    path = os.path.join(PDF_DIR, filename)
    lines, _, _ = dp.parse_pdf(path)
    out = os.path.join(os.path.dirname(__file__), f"ocr_{label}_raw.txt")
    with open(out, "w", encoding="utf-8") as f:
        f.write(f"=== {label} ({filename}) ===\n")
        for i, line in enumerate(lines, 1):
            f.write(f"{i:3}: {line}\n")
    print(f"Saved {len(lines)} lines to {out}")
