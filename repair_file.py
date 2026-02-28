
import sys
import re

def repair_file(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()
    
    lines = content.splitlines()
    new_lines = []
    
    # Target imports to correctly comment out later
    targets = [
        "import pandas as pd",
        "import re",
        "import json",
        "from datetime import datetime, date",
        "from calendar import monthrange",
        "from pathlib import Path",
        "import os",
        "import shutil",
        "import platform",
        "import streamlit as st",
        "import pdfplumber",
        "import pytesseract",
        "from PIL import Image",
        "from pdf2image import convert_from_bytes",
        "import docx",
        "from bank_statement_parser import parse_bank_statement, detect_bank",
        "from bank_statement_parser import TransactionType, TransactionCategory",
        "from schedule_c_categorizer import ScheduleCCategorizer",
        "from account_code_mapper import AccountCodeMapper"
    ]
    
    # 1. First, strip the corruption but keep the top-level consolidation
    # The consolidated imports are at the top, we keep them.
    # We need to find the "--- END CONSOLIDATED IMPORTS ---" marker.
    
    found_marker = False
    index_marker = 0
    for i, line in enumerate(lines):
        if "--- END CONSOLIDATED IMPORTS ---" in line:
            found_marker = True
            index_marker = i
            break
            
    if not found_marker:
        print("Marker not found, cannot repair safely.")
        return

    # Keep everything up to the marker
    new_lines.extend(lines[:index_marker+1])
    
    # Process the rest of the file
    for line in lines[index_marker+1:]:
        # Remove the corruption: "    # " prefix and " (moved to top level)" suffix
        if line.strip().startswith("#") and "(moved to top level)" in line:
            # Extract the original part
            # Expected format: "    # original code (moved to top level)"
            match = re.search(r'^(?P<indent>\s*)#\s*(?P<content>.*?)\s*\(moved to top level\)$', line)
            if match:
                indent = match.group('indent')
                original_content = match.group('content')
                
                # If it's one of the imports we WANT to leave commented out, keep it commented
                is_target = False
                for t in targets:
                    if original_content == t:
                        is_target = True
                        break
                
                if is_target:
                    # Keep as a comment but maybe clean it up
                    new_lines.append(f"{indent}# {original_content} (moved to top level)")
                else:
                    # RESTORE THE CODE
                    new_lines.append(f"{indent}{original_content}")
            else:
                # If regex fails for some reason, just keep the line
                new_lines.append(line)
        else:
            new_lines.append(line)

    with open(filepath, 'w', encoding='utf-8') as f:
        f.write('\n'.join(new_lines))
    print(f"Repaired {filepath}")

if __name__ == "__main__":
    repair_file('bank_data_analysis.py')
