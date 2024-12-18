import streamlit as st
import pandas as pd
from transformers import pipeline

# Initialize Streamlit App
st.title("Dataset Polisher App")
st.write("Upload your dataset to polish it without losing any data quality.")

# Upload Dataset
uploaded_file = st.file_uploader("Upload your dataset (.csv or .xlsx)", type=["csv", "xlsx"])

if uploaded_file:
    # Load Dataset
    if uploaded_file.name.endswith('.csv'):
        df = pd.read_csv(uploaded_file)
    elif uploaded_file.name.endswith('.xlsx'):
        df = pd.read_excel(uploaded_file)

    st.subheader("Raw Dataset Preview")
    st.write(df)

    # Dropdown to select the question column
    question_column = st.selectbox("Select the column containing questions:", df.columns)

    # Polishing Options
    st.subheader("Polishing Options")
    polish_questions = st.checkbox("Rephrase Questions")
    detect_intent = st.checkbox("Add Intent Field")

    # Load AI Model (only if required)
    if polish_questions:
        st.write("Loading AI model for rephrasing...")
        rephraser = pipeline("text2text-generation", model="t5-small")

    # Rephrase Questions
    if polish_questions:
        if question_column in df.columns:
            st.write("Rephrasing questions...")
            try:
                df['Polished_Question'] = df[question_column].apply(
                    lambda x: rephraser(x, max_length=50, num_return_sequences=1)[0]['generated_text'] if pd.notna(x) else x
                )
                st.success("Questions rephrased successfully!")
            except Exception as e:
                st.error(f"An error occurred during rephrasing: {e}")
        else:
            st.error("Please select a valid column for rephrasing.")

    # Add Intent Field
    if detect_intent:
        st.write("Detecting intents...")
        def detect_intent_logic(question):
            if pd.isna(question):
                return "unknown"
            if "requirement" in question.lower():
                return "eligibility"
            elif "mathematics" in question.lower():
                return "subject_requirement"
            else:
                return "general"

        df['Intent'] = df[question_column].apply(detect_intent_logic)
        st.success("Intent field added!")

    # Display Polished Dataset
    st.subheader("Polished Dataset Preview")
    st.write(df)

    # Download Option
    st.subheader("Download Polished Dataset")
    polished_file = st.selectbox("Select export format", ["CSV", "Excel"])
    if polished_file == "CSV":
        csv = df.to_csv(index=False).encode('utf-8')
        st.download_button(
            "Download CSV",
            csv,
            "polished_dataset.csv",
            "text/csv"
        )
    elif polished_file == "Excel":
        excel_buffer = pd.ExcelWriter("polished_dataset.xlsx", engine='xlsxwriter')
        df.to_excel(excel_buffer, index=False, sheet_name='Polished Dataset')
        st.download_button(
            "Download Excel",
            excel_buffer,
            "polished_dataset.xlsx"
        )
