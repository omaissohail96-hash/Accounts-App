from document_parser import DocumentParser

def inspect_bmo():
    pdf_path = 'bank statments/BMO.pdf'
    dp = DocumentParser()
    with open(pdf_path, 'rb') as f:
        text_lines, ok, unreadable = dp.parse_document(f.read(), 'BMO.pdf')
    
    print(f"Total lines extracted: {len(text_lines)}")
    print("\n=== First 200 lines ===")
    for i, line in enumerate(text_lines[:200]):
        print(f"{i:3d}: {line}")

if __name__ == "__main__":
    inspect_bmo()
