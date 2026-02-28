import pdfplumber

def inspect_chase():
    pdf_path = 'bank statments/20 Dec Chase.pdf'
    with pdfplumber.open(pdf_path) as pdf:
        text = "\n".join(p.extract_text() or "" for p in pdf.pages)
        lines = text.split('\n')
        print(f"Total lines: {len(lines)}")
        print("\n=== First 100 lines ===")
        for i, line in enumerate(lines[:100]):
            print(f"{i:3d}: {line}")

if __name__ == "__main__":
    inspect_chase()
