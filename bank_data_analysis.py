import streamlit as st
import pandas as pd

st.set_page_config(page_title="Accounting App Gen 1", layout="wide")
st.title("💼 Accounting App — Gen 1 (Enhanced Reports)")

# ---------- Upload CSV ----------
uploaded = st.file_uploader("Upload transactions CSV", type=["csv"])
if not uploaded:
    st.info("Upload the sample_transactions.csv file to start.")
    st.stop()

df = pd.read_csv(uploaded)
st.subheader("📋 Preview of Uploaded Data")
st.dataframe(df.head())

# ---------- Categorization Rules ----------
rules = [
    {"merchant_contains": "Starbucks", "category": "Meals"},
    {"merchant_contains": "Amazon", "category": "Office Supplies"},
    {"merchant_contains": "Uber", "category": "Travel"},
    {"merchant_contains": "PayPal", "category": "Software Subscription"},
    {"merchant_contains": "Upwork", "category": "Client Payment"},
    {"merchant_contains": "Netflix", "category": "Entertainment"},
    {"merchant_contains": "Walmart", "category": "Groceries"},
    {"merchant_contains": "Shell", "category": "Fuel"},
    {"merchant_contains": "FedEx", "category": "Shipping"},
    {"merchant_contains": "Google", "category": "Advertising"},
    {"merchant_contains": "Canva", "category": "Design Tool"},
    {"merchant_contains": "Zoom", "category": "Communication"},
    {"merchant_contains": "Shopify", "category": "E-commerce Fees"},
]

def apply_rules(row):
    for r in rules:
        if r["merchant_contains"].lower() in str(row["merchant_name"]).lower():
            return r["category"]
    return "Uncategorized"

df["category"] = df.apply(apply_rules, axis=1)

# ---------- Signed Amounts ----------
df["signed_amount"] = df.apply(
    lambda x: x["amount"] if x["direction"] == "in" else -x["amount"], axis=1
)

# ---------- P&L Report ----------
st.subheader("📊 Profit & Loss Report (by Category)")

pnl = df.groupby("category").agg(
    Total_Income = ("signed_amount", lambda x: x[x > 0].sum()),
    Total_Expense = ("signed_amount", lambda x: -x[x < 0].sum()),
    Net_Amount = ("signed_amount", "sum"),
).reset_index()

pnl.fillna(0, inplace=True)
st.dataframe(pnl)

# ---------- Schedule C (Tax View) ----------
st.subheader("💰 Schedule C (Tax View)")

mapping = pd.read_csv("tax_mapping_schedule_c.csv")

tax_view = pnl.merge(mapping, left_on="category", right_on="category", how="left")

def deductible(row):
    amt = row["Net_Amount"]
    if amt < 0:
        return amt * (row["deductibility_pct"] / 100)
    return amt

tax_view["Deductible Amount"] = tax_view.apply(deductible, axis=1)

tax_report = tax_view.groupby(
    ["tax_line_code", "tax_line_name", "deductibility_pct"]
)["Deductible Amount"].sum().reset_index()

tax_report.rename(columns={"Deductible Amount": "Total Deductible ($)"}, inplace=True)
st.dataframe(tax_report)

# ---------- Downloads ----------
st.download_button(
    "⬇️ Download P&L Report (CSV)",
    pnl.to_csv(index=False).encode("utf-8"),
    "pnl_report.csv",
    "text/csv"
)
st.download_button(
    "⬇️ Download Schedule C Report (CSV)",
    tax_report.to_csv(index=False).encode("utf-8"),
    "schedule_c_report.csv",
    "text/csv"
)

st.success("✅ Both reports generated successfully with detailed columns!")
