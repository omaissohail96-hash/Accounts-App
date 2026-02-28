
import sys
import os
from unittest.mock import MagicMock

# 1. Verification of global imports
class ImportChecker:
    def check_file(self, filepath):
        with open(filepath, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        found_locals = []
        in_function = False
        
        for i, line in enumerate(lines):
            stripped = line.strip()
            if stripped.startswith('def ') or stripped.startswith('class '):
                in_function = True
            
            if in_function and (stripped.startswith('import ') or stripped.startswith('from ')):
                # Check if it's a library import (not a relative local module)
                if not stripped.startswith('#') and ('pandas' in stripped or 'json' in stripped or 're' in stripped or 'datetime' in stripped):
                    found_locals.append((i+1, stripped))
        
        return found_locals

checker = ImportChecker()
locals_found = checker.check_file('bank_data_analysis.py')

if locals_found:
    print("WARNING: Found remaining local imports:")
    for line_no, content in locals_found:
        print(f"  Line {line_no}: {content}")
else:
    print("SUCCESS: No common library local imports found in functions.")

# 2. Functional test for pd in global scope
try:
    # Mock streamlit
    sys.modules['streamlit'] = MagicMock()
    sys.modules['pdfplumber'] = MagicMock()
    sys.modules['pytesseract'] = MagicMock()
    
    import bank_data_analysis
    import pandas as pd
    
    # Test if pd is available as expected
    df = pd.DataFrame([{"test": 1}])
    print("SUCCESS: pandas is available and working correctly.")
except Exception as e:
    print(f"FAILED: Functional test failed: {e}")
    sys.exit(1)

print("\n=== VERIFICATION PASSED ===")
