from bank_data_analysis import MainBankParser

parser = MainBankParser()
try:
    res, doc_type, is_readable = parser.parse("bank statments/AMEX.pdf")
    print(f"Doc type: {doc_type} Is readable: {is_readable}")
    for t in res:
        if "UNKNOWN" in t.vendor or t.amount in [314.0, 142.0, 643.0, 550.0, 944.0, 20.0, 334.0, 151.0, 275.0, 807.0]:
            print(t)
    print(f"Total extracted: {len(res)}")
except Exception as e:
    import traceback
    traceback.print_exc()
