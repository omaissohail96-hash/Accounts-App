import re

def test_chase_check_regex():
    # Current regex in bank_statement_parser.py
    # check_regex=r'(\d+)\s+[\^]?\s+(\d{1,2}/\d{1,2})\s+([-+]?\$?[\d,]+\.\d{2}|[\d,]+\.\d{2})$'
    
    current_regex = r'(\d+)\s+[\^]?\s+(\d{1,2}/\d{1,2})\s+([-+]?\$?[\d,]+\.\d{2}|[\d,]+\.\d{2})$'
    
    # Proposed regex
    # Added [\*\^/ ]+ to handle noise symbols like * and combined ^
    proposed_regex = r'(\d+)\s+[\*\^/ ]+\s+(\d{1,2}/\d{1,2})\s+([-+]?\$?[\d,]+\.\d{2}|[\d,]+\.\d{2})$'
    
    test_line = "6478 * ^ 04/09 1,235.25"
    
    print(f"Testing line: '{test_line}'")
    
    m_current = re.search(current_regex, test_line)
    if m_current:
        print("[SUCCESS] Current regex matched!")
        print(f"Groups: {m_current.groups()}")
    else:
        print("[FAILURE] Current regex failed to match.")
        
    m_proposed = re.search(proposed_regex, test_line)
    if m_proposed:
        print("[SUCCESS] Proposed regex matched!")
        print(f"Groups: {m_proposed.groups()}")
    else:
        print("[FAILURE] Proposed regex failed to match.")

if __name__ == "__main__":
    test_chase_check_regex()
