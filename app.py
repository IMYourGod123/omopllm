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
        items = df.to_dict(orient="records")
        updated_items = []

        for item in items:
            clinical_item = item.get("Clinical Item")
            category = item.get("Category")
            description = item.get("Description", "")

            match = llm_validated_concept_match(clinical_item, category, description)

            if match:
                item.update({
                    "Clinical Item": match["concept_name"],
                    "Code": match["concept_id"],
                    "Coding System": "OMOPCDM",
                    "Category": match["domain_id"]
                })
            updated_items.append(item)

        return pd.DataFrame(updated_items)

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
            st.dataframe(df)

            if st.button("🚀 Validate Full Table with LLM"):
                with st.spinner("Validating entire table with LLM..."):
                    validated_df = validate_and_update_dataframe(df)

                st.subheader("✅ Validated Table")
                st.dataframe(validated_df)

                with st.expander("📊 Preview Concept Distribution"):
                    if "Category" in validated_df.columns:
                        fig = px.histogram(validated_df, x="Category", title="Concept Category Distribution")
                        st.plotly_chart(fig)

                st.download_button("📥 Download Validated Table", validated_df.to_json(orient="records", indent=2),
                                   file_name="validated_output.json", mime="application/json")

        except Exception as e:
            st.error(f"Error processing file: {e}")

    elif file_type == "pdf":
        try:
            pdf = fitz.open(stream=uploaded_file.read(), filetype="pdf")
            extracted_texts = [page.get_text() for page in pdf]
            st.subheader("📄 Extracted PDF Text")
            for i, txt in enumerate(extracted_texts):
                st.text_area(f"Page {i+1}", txt, height=200)

            if st.button("🚀 Process PDF Notes with LLM"):
                notes_json = []
                for text in extracted_texts:
                    messages = [
                        SystemMessage(content="Extract key structured concepts from clinical note into JSON."),
                        HumanMessage(content=text)
                    ]
                    try:
                        response = llm.invoke(messages)
                        extracted = json.loads(response.content.strip())
                        if isinstance(extracted, list):
                            notes_json.extend(extracted)
                    except Exception as e:
                        st.warning(f"Extraction failed for a note: {e}")

                if notes_json:
                    validated_df = validate_and_update_dataframe(pd.DataFrame(notes_json))
                    st.subheader("✅ Validated Concepts from PDF")
                    st.dataframe(validated_df)

                    with st.expander("📊 Preview Concept Distribution"):
                        if "Category" in validated_df.columns:
                            fig = px.histogram(validated_df, x="Category", title="Concept Category Distribution")
                            st.plotly_chart(fig)

                    st.download_button("📥 Download Validated Results", validated_df.to_json(orient="records", indent=2),
                                       file_name="validated_output.json", mime="application/json")

        except Exception as e:
            st.error(f"Error extracting PDF text: {e}")
