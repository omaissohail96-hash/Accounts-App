import sys

def get_context(filename, search_str, lines_before=10, lines_after=10):
    try:
        with open(filename, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        for i, line in enumerate(lines):
            if search_str.lower() in line.lower():
                print(f"--- Context for '{search_str}' (Line {i+1}) ---")
                start = max(0, i - lines_before)
                end = min(len(lines), i + lines_after + 1)
                for j in range(start, end):
                    prefix = ">>> " if j == i else "    "
                    print(f"{j+1:4}: {prefix}{lines[j].strip()}")
                print("-" * 40)
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    get_context(sys.argv[1], sys.argv[2])
