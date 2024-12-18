import streamlit as st
from dataset_generator import DatasetGenerator  # Assuming you saved the class code in dataset_generator.py
from together import api as together
import pandas as pd
import random
import json
from datetime import datetime
from sklearn.metrics.pairwise import cosine_similarity

# Streamlit UI Setup
st.set_page_config(page_title="AgriGo Dataset Generation", page_icon="📊", layout="wide")

st.title("🚜 AgriGo Dataset Generator")
st.markdown("""
This tool allows you to generate high-quality conversational datasets for your AgriGo project, 
providing detailed quality metrics and enabling real-time progress monitoring.
""")

# File upload section
st.sidebar.title("🗂️ Upload Files")
uploaded_files = st.sidebar.file_uploader("Upload your TXT, CSV, or JSON files", type=["txt", "csv", "json", "jsonl"], accept_multiple_files=True)

if uploaded_files:
    # Initialize Dataset Generator class
    embedding_model = "your_embedding_model_here"  # Initialize your embedding model
    model_name = "gpt-3.5-turbo"  # Use appropriate model name
    temperature = 0.7
    dataset_generator = DatasetGenerator(embedding_model, model_name, temperature)
    
    # Process uploaded files
    st.sidebar.write("Processing your files...")
    num_files_processed = dataset_generator.add_context_data(uploaded_files)
    st.sidebar.write(f"Processed {num_files_processed} data points from your files.")
    
    # Topic and Concept Input
    topic = st.text_input("Enter the topic for your dataset", value="Agriculture")
    concepts_input = st.text_area("Enter the concepts (comma-separated)", value="Crop Growth, Pest Control, Weather Forecasting")
    
    if concepts_input:
        concepts = [concept.strip() for concept in concepts_input.split(",")]
    
    # Number of samples input
    num_samples = st.number_input("Enter the number of samples to generate", min_value=1, max_value=1000, value=10)
    
    # Generate dataset button
    if st.button("Generate Dataset"):
        try:
            st.write("🔄 Generating dataset... Please wait.")
            dataset = dataset_generator.generate_dataset(topic=topic, concepts=concepts, num_samples=num_samples)
            
            # Display the dataset
            st.write(f"✅ Successfully generated {len(dataset)} samples!")
            if dataset:
                df = pd.DataFrame([
                    {
                        "Prompt": entry["prompt"],
                        "Response": entry["response"],
                        "Quality Metrics": entry["metadata"]["quality_metrics"]
                    } for entry in dataset
                ])
                st.write(df)
                
                # Provide download option
                def to_csv(df):
                    return df.to_csv(index=False).encode('utf-8')
                
                st.download_button(
                    label="Download Dataset as CSV",
                    data=to_csv(df),
                    file_name="generated_dataset.csv",
                    mime="text/csv"
                )
                
                st.success("Dataset generation complete!")
                
        except Exception as e:
            st.error(f"❌ Error: {str(e)}")

else:
    st.warning("Please upload your source files to begin.")

