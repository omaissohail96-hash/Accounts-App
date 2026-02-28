
import sys

def wrap_main(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    new_lines = []
    ui_started = False
    
    for line in lines:
        if "# Streamlit UI" in line and not ui_started:
            new_lines.append(line)
            new_lines.append("def main():\n")
            new_lines.append("    global _STATEMENT_YEAR\n")
            ui_started = True
            continue
            
        if ui_started:
            if line.strip() == "":
                new_lines.append("\n")
            else:
                new_lines.append("    " + line)
        else:
            new_lines.append(line)
            
    # Add the guard at the end
    new_lines.append("\n")
    new_lines.append("if __name__ == \"__main__\":\n")
    new_lines.append("    main()\n")
    
    with open(filepath, 'w', encoding='utf-8') as f:
        f.writelines(new_lines)
    print(f"Wrapped UI in main() for {filepath}")

if __name__ == "__main__":
    wrap_main('bank_data_analysis.py')
