import re

def fuzzy_find_4100():
    with open(r"c:\Users\Farooq\Desktop\New Folder (4)\Accounts-App\ocr_dumps\BoA.txt", 'r', encoding='utf-8') as f:
        text = f.read()
    
    # 1. Search for 4, 1, 0, 0 with potential noise
    pattern = r'4\s*[1I|]\s*0\s*0'
    matches = list(re.finditer(pattern, text))
    
    print(f"Found {len(matches)} potential fuzzy matches for '4100':")
    for m in matches:
        start = max(0, m.start() - 40)
        end = min(len(text), m.end() + 40)
        snippet = text[start:end].replace('\n', ' ')
        print(f"  Match at pos {m.start()}: '{snippet}'")

if __name__ == "__main__":
    fuzzy_find_4100()
