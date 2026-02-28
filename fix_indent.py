
import sys

def indent_file(filepath, start_line):
    with open(filepath, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    # Lines before the streamlit part
    new_lines = lines[:start_line]
    
    # Indent the rest
    for line in lines[start_line:]:
        new_lines.append('    ' + line)
    
    # Add main guard
    new_lines.append('\n\nif __name__ == "__main__":\n    main()\n')
    
    with open(filepath, 'w', encoding='utf-8') as f:
        f.writelines(new_lines)
    print(f"Indented {len(lines) - start_line} lines and added main guard.")

if __name__ == "__main__":
    indent_file('bank_data_analysis.py', 2735)
