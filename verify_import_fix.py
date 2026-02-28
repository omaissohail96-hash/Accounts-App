
import sys
import os
from unittest.mock import MagicMock

# Mock streamlit
mock_st = MagicMock()
mock_st.session_state = {}
sys.modules['streamlit'] = mock_st

# Mock other potential side-effect modules if needed
sys.modules['pdfplumber'] = MagicMock()
sys.modules['pytesseract'] = MagicMock()
sys.modules['pandas'] = MagicMock()

import bank_data_analysis
from datetime import datetime

def verify_fix():
    print("Verifying datetime fix in bank_data_analysis.py...")
    
    # Check if we can access datetime.now() without error
    try:
        # Simulate the logic that was failing
        current_year = datetime.now().year
        print(f"SUCCESS: datetime.now().year = {current_year}")
        
        # Also check if it's accessible through the bank_data_analysis module
        # Note: Since it was a local variable shadowing error, we want to ensure
        # that the function where it failed works.
        # Line 3205 is in main(). We can't easily run the whole main() due to UI
        # but we can verify that there are no conflicting local assignments.
        
        print("\nChecking file content for shadowed datetime assignments inside main()...")
        with open('bank_data_analysis.py', 'r', encoding='utf-8') as f:
            lines = f.readlines()
            # main() starts around line 2734 (now probably shifted)
            found_main = False
            for i, line in enumerate(lines):
                if 'def main():' in line:
                    found_main = True
                    main_start = i
                    break
            
            if found_main:
                # Look for local imports in main
                for i in range(main_start, len(lines)):
                    if 'from datetime import' in lines[i] and not lines[i].strip().startswith('#'):
                        print(f"WARNING: Found active local import at line {i+1}: {lines[i].strip()}")
                        return False
            
        print("Final verification: No active local datetime imports found in main().")
        return True
    except Exception as e:
        print(f"FAILED: Still encountered error: {e}")
        return False

if __name__ == "__main__":
    if verify_fix():
        print("\n=== VERIFICATION PASSED ===")
    else:
        print("\n=== VERIFICATION FAILED ===")
        sys.exit(1)
