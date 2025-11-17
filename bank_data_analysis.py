"""
Bank Statement Analysis Application
Main Streamlit application for processing bank statements
"""
import streamlit as st
import pandas as pd
import logging
from document_parser import DocumentParser
from bank_statement_parser import BankStatementParser
from transaction_categorizer import TransactionCategorizer
from report_generator import ReportGenerator

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

st.set_page_config(
    page_title="Bank Statement Analyzer",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        margin-bottom: 1rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .warning-box {
        background-color: #fff3cd;
        border-left: 4px solid #ffc107;
        padding: 1rem;
        margin: 1rem 0;
    }
    .success-box {
        background-color: #d4edda;
        border-left: 4px solid #28a745;
        padding: 1rem;
        margin: 1rem 0;
    }
    </style>
""", unsafe_allow_html=True)

st.markdown('<p class="main-header">💼 Bank Statement Analyzer</p>', unsafe_allow_html=True)
st.markdown("Upload bank statements (PDF or Word) to automatically generate categorized income and expense summaries for accounting and tax reporting.")

# Initialize session state
if 'transactions' not in st.session_state:
    st.session_state.transactions = []
if 'parsing_metadata' not in st.session_state:
    st.session_state.parsing_metadata = {}
if 'statistics' not in st.session_state:
    st.session_state.statistics = {}

# Sidebar
with st.sidebar:
    st.header("📤 Upload Statement")
    uploaded_file = st.file_uploader(
        "Choose a bank statement file",
        type=['pdf', 'doc', 'docx'],
        help="Supported formats: PDF and Word documents"
    )
    
    if uploaded_file:
        st.info(f"📄 File: {uploaded_file.name}")
        st.info(f"📊 Size: {uploaded_file.size / 1024:.2f} KB")
        
        if st.button("🔄 Process Statement", type="primary", use_container_width=True):
            with st.spinner("Processing statement..."):
                try:
                    # Step 1: Parse document
                    doc_parser = DocumentParser()
                    file_bytes = uploaded_file.read()
                    lines, is_readable, unreadable_pages = doc_parser.parse_document(file_bytes, uploaded_file.name)
                    
                    if not is_readable or len(lines) < 10:
                        st.error("⚠️ Document appears unreadable or has insufficient text content.")
                        if unreadable_pages:
                            st.warning(f"Unreadable pages: {unreadable_pages}. The document may need OCR processing or better image quality.")
                        st.session_state.transactions = []
                        st.stop()
                    
                    # Step 2: Parse bank statement
                    statement_parser = BankStatementParser()
                    transactions, metadata = statement_parser.parse_statement(lines)
                    
                    if not transactions:
                        st.warning("⚠️ No transactions found in the statement. Please verify the document format.")
                        st.session_state.transactions = []
                        st.stop()

                    # Step 3: Categorize transactions
                    categorizer = TransactionCategorizer()
                    transactions = categorizer.process_transactions(transactions)
                    transactions = categorizer.detect_duplicates(transactions)
                    
                    # Step 4: Generate statistics
                    report_gen = ReportGenerator()
                    statistics = report_gen.generate_summary_statistics(transactions)
                    
                    # Store in session state
                    st.session_state.transactions = transactions
                    st.session_state.parsing_metadata = {
                        **metadata,
                        'total_lines': len(lines),
                        'unreadable_pages': unreadable_pages,
                        'is_readable': is_readable
                    }
                    st.session_state.statistics = statistics
                    
                    st.success(f"✅ Processed {len(transactions)} transactions successfully!")
                    
                except Exception as e:
                    st.error(f"❌ Error processing document: {str(e)}")
                    logger.exception("Error processing document")
                    st.session_state.transactions = []

# Main content
if not st.session_state.transactions:
    st.info("👆 Please upload a bank statement file to begin analysis.")
    st.stop()

transactions = st.session_state.transactions
metadata = st.session_state.parsing_metadata
statistics = st.session_state.statistics

# Display warnings if needed
if metadata.get('unreadable_pages'):
    st.warning(f"⚠️ Some pages ({metadata['unreadable_pages']}) were unreadable. Please review the 'Needs Review' section.")

needs_review_count = statistics.get('Transactions Needing Review', 0)
if needs_review_count > 0:
    st.warning(f"⚠️ {needs_review_count} transaction(s) need manual review. Check the 'Needs Review' section below.")

# Summary Statistics
st.header("📊 Summary Statistics")
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Total Deposits", f"${statistics.get('Total Deposit Amount', 0):,.2f}", 
              f"{statistics.get('Total Deposits', 0)} transactions")

with col2:
    st.metric("Total Withdrawals", f"${statistics.get('Total Withdrawal Amount', 0):,.2f}",
              f"{statistics.get('Total Withdrawals', 0)} transactions")

with col3:
    net_income = statistics.get('Net Income', 0)
    color = "normal" if net_income >= 0 else "inverse"
    st.metric("Net Income", f"${net_income:,.2f}", delta=None)

with col4:
    st.metric("Total Transactions", statistics.get('Total Transactions', 0),
              f"{needs_review_count} need review" if needs_review_count > 0 else "All clear")

# Tabs for different views
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "💰 Deposits Summary",
    "💸 Withdrawals Summary",
    "📈 P&L Report",
    "⚠️ Needs Review",
    "📋 All Transactions"
])

report_gen = ReportGenerator()

with tab1:
    st.subheader("Deposits Summary (Amount In)")
    deposits_df = report_gen.generate_deposits_summary(transactions)
    
    if not deposits_df.empty:
        # Display summary table (without transaction details)
        display_df = deposits_df[['Source/Vendor', 'Transaction Count', 'Subtotal ($)']].copy()
        st.dataframe(display_df, use_container_width=True, hide_index=True)
        
        # Expandable detailed view
        with st.expander("View Detailed Transaction List"):
            st.dataframe(deposits_df, use_container_width=True, hide_index=True)
        
        # Verify totals
        total_from_statement = deposits_df[deposits_df['Source/Vendor'] == 'TOTAL DEPOSITS']['Subtotal ($)'].values[0]
        st.info(f"💡 Total Deposits: ${total_from_statement:,.2f} (Please verify this matches your statement)")
    else:
        st.info("No deposits found in the statement.")

with tab2:
    st.subheader("Withdrawals Summary (Amount Out)")
    withdrawals_df = report_gen.generate_withdrawals_summary(transactions)
    
    if not withdrawals_df.empty:
        # Display summary table
        display_df = withdrawals_df[['Category', 'Vendor', 'Transaction Count', 'Subtotal ($)']].copy()
        st.dataframe(display_df, use_container_width=True, hide_index=True)
        
        # Expandable detailed view
        with st.expander("View Detailed Transaction List"):
            st.dataframe(withdrawals_df, use_container_width=True, hide_index=True)
        
        # Verify totals
        total_withdrawals = withdrawals_df[withdrawals_df['Category'] == 'TOTAL WITHDRAWALS']['Subtotal ($)'].values[0]
        st.info(f"💡 Total Withdrawals: ${total_withdrawals:,.2f} (Please verify this matches your statement)")
    else:
        st.info("No withdrawals found in the statement.")

with tab3:
    st.subheader("Profit & Loss Report")
    pl_df = report_gen.generate_pl_report(transactions)
    
    if not pl_df.empty:
        st.dataframe(pl_df, use_container_width=True, hide_index=True)
        
        # Highlight net income
        net_income_row = pl_df[pl_df['Category'] == 'NET INCOME']
        if not net_income_row.empty:
            net_income_val = net_income_row['Amount ($)'].values[0]
            if net_income_val >= 0:
                st.success(f"✅ Net Income: ${net_income_val:,.2f}")
            else:
                st.error(f"❌ Net Loss: ${net_income_val:,.2f}")
        
        # Chart visualization
        expense_df = pl_df[(pl_df['Type'] == 'Expense') & (pl_df['Category'] != 'TOTAL EXPENSES')]
        if not expense_df.empty:
            st.subheader("Expenses by Category")
            expense_chart_df = expense_df[['Category', 'Amount ($)']].copy()
            expense_chart_df['Amount ($)'] = expense_chart_df['Amount ($)'].abs()
            st.bar_chart(expense_chart_df.set_index('Category'))
    else:
        st.info("No data available for P&L report.")

with tab4:
    st.subheader("⚠️ Transactions Needing Review")
    review_df = report_gen.generate_needs_review_section(transactions)
    
    if not review_df.empty:
        st.warning(f"Found {len(review_df)} transaction(s) that need manual review.")
        st.dataframe(review_df, use_container_width=True, hide_index=True)
        st.info("💡 These transactions had unclear formatting, missing amounts, or incomplete descriptions. Please review and categorize manually if needed.")
    else:
        st.success("✅ No transactions need review! All transactions were successfully parsed and categorized.")

with tab5:
    st.subheader("All Transactions")
    all_trans_df = pd.DataFrame([
        {
            'Date': t.date or 'N/A',
            'Type': t.transaction_type.title(),
            'Vendor': t.vendor or 'N/A',
            'Category': t.category or 'Uncategorized',
            'Amount': f"${t.amount:,.2f}" if t.amount != 0 else 'N/A',
            'Description': t.description[:100] if t.description else 'N/A',
            'Needs Review': '⚠️ Yes' if t.needs_review else '✅ No'
        }
        for t in transactions
    ])
    
    # Filters
    col1, col2, col3 = st.columns(3)
    with col1:
        type_filter = st.selectbox("Filter by Type", ['All', 'Deposit', 'Withdrawal'], key='type_filter')
    with col2:
        category_filter = st.selectbox("Filter by Category", 
                                       ['All'] + sorted(all_trans_df['Category'].unique().tolist()),
                                       key='category_filter')
    with col3:
        review_filter = st.selectbox("Review Status", ['All', 'Needs Review', 'OK'], key='review_filter')
    
    # Apply filters
    filtered_df = all_trans_df.copy()
    if type_filter != 'All':
        filtered_df = filtered_df[filtered_df['Type'] == type_filter]
    if category_filter != 'All':
        filtered_df = filtered_df[filtered_df['Category'] == category_filter]
    if review_filter == 'Needs Review':
        filtered_df = filtered_df[filtered_df['Needs Review'] == '⚠️ Yes']
    elif review_filter == 'OK':
        filtered_df = filtered_df[filtered_df['Needs Review'] == '✅ No']
    
    st.dataframe(filtered_df, use_container_width=True, hide_index=True)
    st.info(f"Showing {len(filtered_df)} of {len(all_trans_df)} transactions")

# Download Section
st.header("📥 Download Reports")

col1, col2, col3 = st.columns(3)

with col1:
    # CSV downloads
    deposits_csv = report_gen.generate_deposits_summary(transactions).to_csv(index=False)
    st.download_button(
        "⬇️ Download Deposits Summary (CSV)",
        deposits_csv.encode('utf-8'),
        "deposits_summary.csv",
        "text/csv",
        use_container_width=True
    )

with col2:
    withdrawals_csv = report_gen.generate_withdrawals_summary(transactions).to_csv(index=False)
    st.download_button(
        "⬇️ Download Withdrawals Summary (CSV)",
        withdrawals_csv.encode('utf-8'),
        "withdrawals_summary.csv",
        "text/csv",
        use_container_width=True
    )

with col3:
    pl_csv = report_gen.generate_pl_report(transactions).to_csv(index=False)
st.download_button(
    "⬇️ Download P&L Report (CSV)",
        pl_csv.encode('utf-8'),
        "pl_report.csv",
        "text/csv",
        use_container_width=True
    )

# Excel download
st.markdown("---")
if st.button("📊 Download Complete Report (Excel)", type="primary", use_container_width=True):
    try:
        excel_filename = report_gen.export_to_excel(transactions, "bank_statement_report.xlsx")
        with open(excel_filename, 'rb') as f:
            st.download_button(
                "⬇️ Download Excel Report",
                f.read(),
                excel_filename,
                "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                use_container_width=True
            )
    except Exception as e:
        st.error(f"Error generating Excel report: {str(e)}")

st.success("✅ Reports generated successfully! Please verify totals match your bank statement.")
