
import pdfplumber
import os

pdf_dir = r"c:\Users\Farooq\Desktop\New Folder (4)\Accounts-App\bank statments"
output_dir = r"c:\Users\Farooq\Desktop\New Folder (4)\Accounts-App\pdf_dumps"

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

pdfs = [
    "20 Dec Chase.pdf",
    "AMEX.pdf",
    "BMO.pdf",
    "BoA.pdf",
    "FifthThird 53.pdf",
    "US Bank.pdf"
]

for pdf_name in pdfs:
    pdf_path = os.path.join(pdf_dir, pdf_name)
    output_path = os.path.join(output_dir, pdf_name.replace(".pdf", ".txt"))
    
    print(f"Extracting {pdf_name}...")
    try:
        with pdfplumber.open(pdf_path) as pdf:
            full_text = ""
            for page in pdf.pages:
                full_text += page.extract_text() + "\n"
                
            with open(output_path, "w", encoding="utf-8") as f:
                f.write(full_text)
        print(f"Saved to {output_path}")
    except Exception as e:
        print(f"Error extracting {pdf_name}: {e}")
