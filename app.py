import streamlit as st
import pandas as pd
from io import StringIO
from transformers import pipeline

# Initialize Streamlit App
st.title("Dataset Polisher App")
st.write("Upload your dataset to polish it while ensuring data integrity.")

# Upload Dataset
uploaded_file = st.file_uploader("Upload your dataset (.csv or .xlsx)", type=["csv", "xlsx"])

if uploaded_file:
    # Load the dataset
    if uploaded_file.name.endswith('.csv'):
        df = pd.read_csv(uploaded_file)
    elif uploaded_file.name.endswith('.xlsx'):
        df = pd.read_excel(uploaded_file)

    st.subheader("Raw Dataset Preview")
    st.write(df)

    # Select the column for questions
    question_column = st.selectbox("Select the column containing questions:", df.columns)

    if polish_questions:
        if question_column in df.columns:
            st.write("Rephrasing questions...")
            df['Polished_Question'] = df[question_column].apply(
                lambda x: rephraser(x, max_length=50, num_return_sequences=1)[0]['generated_text']
            )
            st.success("Questions rephrased!")
        else:
            st.error("Please select a valid column for rephrasing.")

    # Add Intent Field
    if detect_intent:
        st.write("Detecting intents...")
        def detect_intent_logic(question):
            if "requirement" in question.lower():
                return "eligibility"
            elif "mathematics" in question.lower():
                return "subject_requirement"
            else:
                return "general"

        df['Intent'] = df['Question'].apply(detect_intent_logic)
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
        excel_buffer = StringIO()
        with pd.ExcelWriter(excel_buffer, engine='xlsxwriter') as writer:
            df.to_excel(writer, index=False, sheet_name='Polished Dataset')
        st.download_button(
            "Download Excel",
            excel_buffer.getvalue(),
            "polished_dataset.xlsx"
        )
