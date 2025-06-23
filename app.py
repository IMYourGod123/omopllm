import sys
import os

# Patch sqlite3 with pysqlite3 if available
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
try:
    import pysqlite3
    sys.modules["sqlite3"] = pysqlite3
except ImportError:
    pass

import streamlit as st
import json
import pandas as pd
import torch
import chromadb
from sentence_transformers import SentenceTransformer
from openai import OpenAI
import fitz  # PyMuPDF for PDF reading
import plotly.express as px

# ---------------------------
# Initialization
# ---------------------------
st.set_page_config(page_title="OMOP LLM Concept Validator", layout="wide")
st.title("🧠 OMOP LLM Concept Validator")

# Load embedding model
device = "cuda" if torch.cuda.is_available() else "cpu"
st.info(f"Using device: {device}")
model = SentenceTransformer("neuml/pubmedbert-base-embeddings", device=device)

# Load ChromaDB collections
chroma_client = chromadb.PersistentClient()
domain_collections = {
    "Observation": chroma_client.get_or_create_collection("concept_embeddings_observation_domain"),
    "Condition": chroma_client.get_or_create_collection("concept_embeddings_condition_domain"),
    "Procedure": chroma_client.get_or_create_collection("concept_embeddings_procedure_domain"),
    "Drug": chroma_client.get_or_create_collection("concept_embeddings_drug_domain"),
    "Measurement": chroma_client.get_or_create_collection("concept_embeddings_measurement_domain"),
}

category_to_domain = {
    "Diagnosis": "Condition", "Diagnoses": "Condition",
    "Medication": "Drug", "Medications": "Drug",
    "Measurement": "Measurement", "Measurements": "Measurement",
    "Procedure": "Procedure",
    "Observation": "Observation", "Observations": "Observation"
}

# Load OpenRouter LLM Client
client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=st.secrets["OPENROUTER_API_KEY"]
)

# ---------------------------
# Functions
# ---------------------------
def call_llm(messages):
    completion = client.chat.completions.create(
        model="deepseek/deepseek-r1:free",
        messages=messages,
        extra_headers={
            "HTTP-Referer": "https://yourapp.site",  # replace with your site
            "X-Title": "OMOP LLM Validator"
        }
    )
    return completion.choices[0].message.content

def get_edited_note(note):
    messages = [
        {"role": "system", "content": "You are a medical note editor for a hospital in Malaysia."},
        {"role": "user", "content": f"""
            Your task is to rephrase medical notes into full and proper sentences and structured, expand all abbreviations, and make the language more formal and suitable for official documentation based on the Dataframe column called UnformattedText. Maintain the original meaning of the note while ensuring clarity, accuracy, and a professional tone.
            Return only the revised and structured medical note without extra commentary and avoid Markdown-style formatting.
            Medical Note: 
            {note}
        """}
    ]
    return call_llm(messages)

def get_formatted_json(note):
    messages = [
        {"role": "system", "content": "You are a clinical coding assistant."},
        {"role": "user", "content": f"""
        From the medical note below, perform the following tasks:
        1. Extract and classify all relevant clinical information into the following five categories:
        Diagnosis, Medication, Measurement, Procedures, Observations.

        2. For each item, return the best-matching standard code and description using:
        ICD-10-CM (Diagnosis, Procedures), RxNorm (Medication), LOINC (Measurement, Observations).

        3. Capture associated values in a 'Value' field.

        4. Return raw JSON array with fields:
        Category | Clinical Item | Value | Code | Coding System | Description

        ⚠️ Return only the raw JSON array. No Markdown.
        Medical Note: 
        {note}
        """}
    ]
    return call_llm(messages)

def validate_and_update_dataframe(df):
    try:
        if "UnformattedText" in df.columns:
            with st.spinner("Preprocessing medical notes with LLM..."):
                df["EditedText"] = df["UnformattedText"].apply(get_edited_note)
                df["Jsonformatted"] = df["EditedText"].apply(get_formatted_json)
        return df
    except Exception as e:
        st.error(f"Validation error: {e}")
        return df

# ---------------------------
# Streamlit App
# ---------------------------
uploaded_file = st.file_uploader("📤 Upload clinical data (CSV, Excel, PDF)", type=["csv", "xlsx", "xls", "pdf"])
if uploaded_file:
    file_type = uploaded_file.name.split(".")[-1].lower()
    df = None

    if file_type in ["csv", "xlsx", "xls"]:
        try:
            df = pd.read_csv(uploaded_file) if file_type == "csv" else pd.read_excel(uploaded_file)
            df = df.drop(columns=[
                'HuggingFace - Llama3-OpenBioLLM-70B',
                'Anthropic- Claude 3.7 Sonnet',
                'OpenAI -GPT-4o (Clinical Note)',
                'OpenAI -GPT-4o (Json)',
                'Google - Med-PaLM'
            ], errors='ignore')

            st.subheader("📄 Uploaded Table")
            st.dataframe(df[["UnformattedText"]])

            if st.button("🚀 Validate Full Table with LLM"):
                with st.spinner("Validating entire table with LLM..."):
                    validated_df = validate_and_update_dataframe(df)

                st.subheader("✅ Validated Table")
                for idx, row in validated_df.iterrows():
                    st.markdown("---")
                    st.markdown(f"**Original Note:**\n\n{row['UnformattedText']}")
                    st.markdown(f"**Edited Note:**\n\n{row['EditedText']}")
                    st.markdown("**🧾 Extracted Concepts:**")
                    try:
                        concepts = json.loads(row['Jsonformatted'])
                        concept_text = "\n\n".join([json.dumps(c, indent=2) for c in concepts])
                        st.markdown(f"```json\n{concept_text}\n```")
                    except Exception as e:
                        st.warning(f"Failed to parse concepts: {e}")

                st.download_button("📥 Download Validated Table", validated_df.to_json(orient="records", indent=2),
                                   file_name="validated_output.json", mime="application/json")

        except Exception as e:
            st.error(f"Error processing file: {e}")
