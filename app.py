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
from langchain_openai import ChatOpenAI
from langchain.schema import SystemMessage, HumanMessage
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

# Load LLM
llm = ChatOpenAI(model="gpt-4", temperature=0)

# ---------------------------
# Functions
# ---------------------------
def get_edited_note(note):
    messages = [
        SystemMessage(content="You are a medical note editor for a hospital in Malaysia."),
        HumanMessage(
            content=f"""
            Your task is to rephrase medical notes into full and proper sentences and structured, expand all abbreviations, and make the language more formal and suitable for official documentation based on the Dataframe column called UnformattedText. Maintain the original meaning of the note while ensuring clarity, accuracy, and a professional tone. 

            Return only the revised and structured medical note without extra commentary and avoid Markdown-style formatting.
            Medical Note: 
            {note}
            """
        ),
    ]
    response = llm.invoke(messages)
    return response.content

def get_formatted_json(note):
    messages = [
        SystemMessage(content="You are a clinical coding assistant."),
        HumanMessage(
            content=f"""
        From the medical note below, perform the following tasks:
        1. Extract and classify all relevant clinical information into the following five categories:
        Diagnosis: All identified conditions or diseases (past or present).
        Medication: Any prescribed or recommended drugs or supplements.
        Measurement: Any clinical findings or test results (e.g., lab values, imaging findings, physical exam results).
        Procedures: Any performed or planned medical procedures, tests, or imaging.
        Observations: Clinical symptoms, patient-reported issues, or clinician-noted changes in condition.

        2. For each extracted item, return the best-matching standard medical code and description using the appropriate coding system.
        Use ICD-10-CM for Diagnosis and Procedures.
        Use RxNorm for Medication.
        Use LOINC for Measurement and Observations.

        3. When applicable, capture and return the associated value or quantity for Medication (e.g., dose, frequency) and Measurement (e.g., lab values, vital signs) in a separate field named Value.

        4. Return the output in a JSON format with the following columns:
        Category | Clinical Item | Value | Code | Coding System | Description

        ⚠️ IMPORTANT: Return only the raw JSON array. Do NOT wrap it in triple backticks or use markdown formatting (e.g., ```json).

        Medical Note: 
        {note}
        """
        ),
    ]
    response = llm.invoke(messages)
    return response.content

def llm_validated_concept_match(clinical_item, original_category, description):
    try:
        expected_domain = category_to_domain.get(original_category)
        if not expected_domain:
            return None

        collection = domain_collections.get(expected_domain)
        if not collection:
            return None

        query_text = f"{clinical_item} - {description}: {expected_domain}"
        embedding = model.encode(query_text).tolist()
        results = collection.query(query_embeddings=[embedding], n_results=30)

        if not results.get("documents") or not results["documents"][0]:
            return None

        candidate_concepts = [json.loads(doc) for doc in results["documents"][0]]

        messages = [
            SystemMessage(content="You are a clinical coding expert using OMOP CDM."),
            HumanMessage(content=f"""
Select the best OMOP concept.
Clinical Term: "{clinical_item}"
Description: "{description}"
Expected Domain: "{expected_domain}"
Candidates:
{json.dumps(candidate_concepts)}
Return only ONE JSON object inside an array. No markdown or explanation.
""")
        ]

        response = llm.invoke(messages)
        selected = json.loads(response.content.strip())
        return selected[0] if selected else None

    except Exception as e:
        st.error(f"LLM validation error: {e}")
        return None

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
            st.dataframe(df[["UnformattedText", "EditedText"]])

            if st.button("🚀 Validate Full Table with LLM"):
                with st.spinner("Validating entire table with LLM..."):
                    validated_df = validate_and_update_dataframe(df)

                st.subheader("✅ Validated Table")
                for idx, row in validated_df.iterrows():
    st.markdown("---")
    st.markdown(f"**Original Note:**

{row['UnformattedText']}")
    st.markdown(f"**Edited Note:**

{row['EditedText']}")
    st.markdown("**🧾 Extracted Concepts:**")
    try:
        concepts = json.loads(row['Jsonformatted'])
        for concept in concepts:
            st.json(concept)
    except Exception as e:
        st.warning(f"Failed to parse concepts: {e}")

                st.download_button("📥 Download Validated Table", validated_df.to_json(orient="records", indent=2),
                                   file_name="validated_output.json", mime="application/json")

        except Exception as e:
            st.error(f"Error processing file: {e}")
