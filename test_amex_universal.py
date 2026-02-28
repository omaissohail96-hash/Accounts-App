import sys
class DummySessionState(dict):
    def __getattr__(self, name): return "test_user"
class DummySt:
    session_state = DummySessionState()
    def __getattr__(self, name): return lambda *args, **kwargs: None
sys.modules['streamlit'] = DummySt()

from bank_data_analysis import UniversalParser
import re

with open("amex_raw_full.txt", "r", encoding="utf-8") as f:
    lines = f.read().splitlines()

parser = UniversalParser()
txs = parser.parse(lines)
print(f"Total parsed by UniversalParser: {len(txs)}")
for t in txs[:5]:
    print(t)
