import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from pypdf import PdfReader
import re
from sklearn.ensemble import IsolationForest
from collections import Counter

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="AI Fraud Guard: Benford & ML",
    page_icon="🕵️‍♂️",
    layout="wide"
)

# --- HELPER FUNCTIONS ---
def extract_financial_numbers(pdf_file):
    """
    Reads a PDF and extracts all numbers that look like financial amounts.
    Filters out years (2020, 2021) and small percentages.
    """
    reader = PdfReader(pdf_file)
    text = ""
    for page in reader.pages:
        text += page.extract_text() + " "
    
    # Regex to find numbers: e.g., 1,234.56 or 5000
    # We ignore dates like 2019-2025 by checking context or value range if needed
    # For simplicity, we grab all numbers with digits, commas, and dots
    matches = re.findall(r'\b\d{1,3}(?:,\d{3})*(?:\.\d+)?\b', text)
    
    clean_numbers = []
    for m in matches:
        # Remove commas to convert to float
        clean_str = m.replace(',', '')
        try:
            val = float(clean_str)
            # Filter: Ignore integers that look like Years (1990-2030) or small integers (1-10) often used for bullets
            if (val > 10 and val < 1900) or (val > 2100): 
                clean_numbers.append(val)
        except:
            continue
            
    return clean_numbers

def calculate_benford_stats(numbers):
    """
    Calculates the First Digit distribution of the extracted numbers.
    """
    first_digits = [int(str(abs(n))[0]) for n in numbers if n != 0]
    total_count = len(first_digits)
    counts = Counter(first_digits)
    
    # Calculate percentages for digits 1-9
    actual_freq = []
    for d in range(1, 10):
        actual_freq.append((counts[d] / total_count) * 100 if total_count > 0 else 0)
        
    return actual_freq, first_digits

# --- DASHBOARD LAYOUT ---
st.title("🕵️‍♂️ AI Fraud Detection System")
st.markdown("""
**Methodology:** 1. **Benford's Law Analysis:** Checks if the distribution of numbers follows the natural "First Digit Law".
2. **AI Anomaly Detection:** Uses an `Isolation Forest` model to find statistical outliers in the dataset.
""")

# Sidebar
with st.sidebar:
    st.header("Upload Financial Report")
    uploaded_file = st.file_uploader("Upload Annual Report (PDF)", type="pdf")
    st.info("The system will extract numerical data to verify accounting irregularities.")

if uploaded_file:
    with st.spinner("Extracting Financial Data & Running AI Models..."):
        # 1. Extraction
        numbers = extract_financial_numbers(uploaded_file)
        
        if len(numbers) < 50:
            st.error("Not enough data found in PDF to perform reliable analysis. Please upload a detailed financial report.")
        else:
            # 2. Benford Calculation
            actual_freq, first_digits_list = calculate_benford_stats(numbers)
            benford_freq = [30.1, 17.6, 12.5, 9.7, 7.9, 6.7, 5.8, 5.1, 4.6] # Standard Benford Percentages
            digits = [str(x) for x in range(1, 10)]

            # 3. AI Anomaly Detection (Isolation Forest)
            # Reshape for Sklearn
            X = np.array(numbers).reshape(-1, 1)
            clf = IsolationForest(contamination=0.05, random_state=42) # Assume 5% outliers
            preds = clf.fit_predict(X)
            
            # Create DataFrame for results
            df_nums = pd.DataFrame({'Amount': numbers, 'Anomaly_Score': preds})
            # -1 means Anomaly, 1 means Normal
            anomalies = df_nums[df_nums['Anomaly_Score'] == -1]

            # --- VISUALIZATION: BENFORD'S LAW ---
            st.subheader("1. Benford's Law Analysis (Macro Check)")
            
            fig = go.Figure()
            # Expected Bar
            fig.add_trace(go.Bar(x=digits, y=benford_freq, name='Expected (Benford)', marker_color='lightgrey'))
            # Actual Line
            fig.add_trace(go.Scatter(x=digits, y=actual_freq, name='Actual Data', line=dict(color='red', width=4)))
            
            fig.update_layout(title="First Digit Distribution: Actual vs. Expected", 
                              xaxis_title="First Digit", yaxis_title="Frequency (%)")
            st.plotly_chart(fig, use_container_width=True)

            # Benford Mean Absolute Deviation (MAD) Check
            mad = np.mean(np.abs(np.array(actual_freq) - np.array(benford_freq)))
            col1, col2 = st.columns(2)
            col1.metric("Data Points Analyzed", len(numbers))
            col1.metric("Deviation Score (MAD)", f"{mad:.2f}")
            
            if mad > 2.0:
                col2.error("⚠️ **High Irregularity Detected!** The data deviates significantly from Benford's Law. This suggests potential manipulation or data quality issues.")
            else:
                col2.success("✅ **Pass:** The data roughly conforms to Benford's Law.")

            # --- VISUALIZATION: AI ANOMALY DETECTION ---
            st.markdown("---")
            st.subheader("2. AI-Based Specific Anomaly Detection (Micro Check)")
            st.write(f"The **Isolation Forest** algorithm identified **{len(anomalies)} specific values** that are statistical outliers compared to the rest of the report.")

            # Histogram of distribution with outliers highlighted
            fig_dist = go.Figure()
            fig_dist.add_trace(go.Histogram(x=df_nums[df_nums['Anomaly_Score']==1]['Amount'], name='Normal Data', marker_color='blue', opacity=0.6))
            fig_dist.add_trace(go.Histogram(x=anomalies['Amount'], name='AI Flagged Anomalies', marker_color='red', opacity=1.0))
            fig_dist.update_layout(title="Distribution of Financial Amounts (Log Scale)", xaxis_type="log", yaxis_title="Count")
            st.plotly_chart(fig_dist, use_container_width=True)

            # Detailed Table
            with st.expander("🚩 View Detailed Suspicious Transactions"):
                st.dataframe(anomalies['Amount'].sort_values(ascending=False).style.format("{:,.2f}"), use_container_width=True)

else:
    st.info("Waiting for PDF upload...")
