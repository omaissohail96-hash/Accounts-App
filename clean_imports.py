
import sys
import re

def consolidate_imports(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # 1. Identify all library imports to move to top level
    # We'll specifically target the ones causing issues: pandas, re, json, datetime, date, calendar, pathlib, os, etc.
    imports_to_move = [
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
    
    lines = content.splitlines()
    new_lines = []
    
    # Pre-process imports to move: remove indentation and trailing whitespace
    patterns = [re.escape(imp.strip()) for imp in imports_to_move]
    
    # 2. Collect all active top-level imports and remove local ones
    for line in lines:
        stripped = line.strip()
        is_local_import = False
        
        # Check if it's one of our target imports
        for pattern in patterns:
            # Match strictly as a whole line (optionally indented)
            if re.match(r'^\s+' + re.escape(line.strip()) + r'$', line):
                is_local_import = True
                break
        
        if is_local_import:
            new_lines.append("    # " + line.strip() + " (moved to top level)")
        else:
            new_lines.append(line)
            
    # 3. Ensure top-level imports are exhaustive
    # We'll replace the existing top-level imports block (around lines 5-25)
    # with a clean consolidated block.
    
    # Find the end of the top-level imports (usually before the first class or def)
    top_import_end = 0
    for i, line in enumerate(new_lines):
        if line.startswith('class ') or line.startswith('def ') or line.startswith('@'):
            top_import_end = i
            break
            
    final_imports = [
        "import io",
        "import re",
        "import json",
        "import logging",
        "import tempfile",
        "import platform",
        "import shutil",
        "import os",
        "import platform",
        "from dataclasses import dataclass, asdict",
        "from datetime import datetime, date",
        "from typing import List, Tuple, Dict, Any, Optional",
        "from pathlib import Path",
        "from calendar import monthrange",
        "",
        "import pdfplumber",
        "import pandas as pd",
        "import streamlit as st",
        "import pytesseract",
        "from PIL import Image",
        "from pdf2image import convert_from_bytes"
    ]
    # Add optional ones that might not exist in all environments but are used in the file
    final_imports.extend([
        "try:",
        "    import docx",
        "except ImportError:",
        "    docx = None",
        "",
        "from bank_statement_parser import (",
        "    parse_bank_statement, ",
        "    detect_bank, ",
        "    TransactionType, ",
        "    TransactionCategory",
        ")",
        "from schedule_c_categorizer import ScheduleCCategorizer",
        "from account_code_mapper import AccountCodeMapper"
    ])
    
    # Remove old top-level imports (rough range 6-30)
    # Actually, let's just insert them at the top after the docstring
    # and comment out any existing ones to be safe.
    
    result = lines[:6] + final_imports + ["", "# --- END CONSOLIDATED IMPORTS ---", ""]
    
    # Filter out the old top-level imports (lines 6-30 approximately)
    # and also all the local ones we commented out
    for line in new_lines[6:]:
        # Skip lines that are just imports we already consolidated
        stripped = line.strip()
        matched = False
        for imp in final_imports:
            if stripped == imp:
                matched = True
                break
        if matched and not line.startswith('    '):
            continue
        result.append(line)

    with open(filepath, 'w', encoding='utf-8') as f:
        f.write('\n'.join(result))
    print(f"Consolidated imports in {filepath}")

if __name__ == "__main__":
    consolidate_imports('bank_data_analysis.py')
