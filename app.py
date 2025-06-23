import sys
import os

# Patch sqlite3 with pysqlite3 if available
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"  # Optional protobuf fix
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

# Force CPU device to avoid GPU issues
device = "cpu"
st.info(f"Using device: {device}")

# Load embedding model
model = SentenceTransformer("neuml/pubmedbert-base-embeddings", device=device)

# Connect ChromaDB collections
chroma_client = chromadb.PersistentClient()

collection_concept_embeddings_observation_domain = chroma_client.get_or_create_collection(name='concept_embeddings_observation_domain')
collection_concept_embeddings_condition_domain = chroma_client.get_or_create_collection(name='concept_embeddings_condition_domain')
collection_concept_embeddings_procedure_domain = chroma_client.get_or_create_collection(name='concept_embeddings_procedure_domain')
collection_concept_embeddings_drug_domain = chroma_client.get_or_create_collection(name='concept_embeddings_drug_domain')
collection_concept_embeddings_measurement_domain = chroma_client.get_or_create_collection(name='concept_embeddings_measurement_domain')

# Load LLM
llm = ChatOpenAI(model="gpt-4", temperature=0)

category_to_domain = {
    "Diagnosis": "Condition", "Diagnoses": "Condition",
    "Medication": "Drug", "Medications": "Drug",
    "Measurement": "Measurement", "Measurements": "Measurement",
    "Procedure": "Procedure",
    "Observation": "Observation", "Observations": "Observation"
}

def llm_validated_concept_match(clinical_item, original_category, description):
    try:
        expected_domain = category_to_domain.get(original_category)
        if expected_domain is None:
            return None

        domain_to_collection = {
            "Observation": collection_concept_embeddings_observation_domain,
            "Condition": collection_concept_embeddings_condition_domain,
            "Procedure": collection_concept_embeddings_procedure_domain,
            "Drug": collection_concept_embeddings_drug_domain,
            "Measurement": collection_concept_embeddings_measurement_domain
        }

        collection = domain_to_collection.get(expected_domain)
        if collection is None:
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
You are an expert in mapping clinical terms to OMOP CDM concepts.

You are given:
- A **clinical term** from a doctor’s note
- A **description** that provides important clinical context (often a brand name or clarification)
- An **expected OMOP domain**
- A list of 30 **OMOP candidate concepts** (already filtered semantically)

🎯 Your goal is to select the **single most appropriate concept** that matches the intended meaning.

📌 Format requirement:
- Return a **raw JSON array with ONE object only**
- No markdown, no extra explanation

Clinical Term: "{clinical_item}"
Description: "{description}"
Expected Domain: "{expected_domain}"

Candidates:
{json.dumps(candidate_concepts)}
""")
        ]

        response = llm.invoke(messages)
        selected = json.loads(response.content.strip())
        return selected[0]

    except Exception as e:
        st.error(f"LLM validation error: {e}")
        return None

def validate_and_update_items(items):
    updated_items = []

    for item in items:
        clinical_item = item.get("Clinical Item") or item.get("clinical_item")
        category = item.get("Category") or item.get("category")
        description = item.get("Description") or item.get("description", "")

        if not clinical_item or not category:
            # Skip incomplete records
            updated_items.append(item)
            continue

        match = llm_validated_concept_match(clinical_item, category, description)

        if match:
            item["Clinical Item"] = match["concept_name"]
            item["Code"] = match["concept_id"]
            item["Coding System"] = "OMOPCDM"
            item["Category"] = match["domain_id"]

        updated_items.append(item)

    return updated_items

def process_json_file(raw_json):
    try:
        items = json.loads(raw_json)
        updated_items = validate_and_update_items(items)
        return json.dumps(updated_items, indent=2), updated_items
    except Exception as e:
        st.error(f"JSON Processing error: {e}")
        return raw_json, None

def process_csv_file(raw_csv):
    try:
        df = pd.read_csv(raw_csv)
        # Convert dataframe to list of dicts
        items = df.to_dict(orient="records")
        updated_items = validate_and_update_items(items)
        # Convert back to dataframe for display
        updated_df = pd.DataFrame(updated_items)
        # Also get JSON string for download
        updated_json = json.dumps(updated_items, indent=2)
        return updated_df, updated_json
    except Exception as e:
        st.error(f"CSV Processing error: {e}")
        return None, None

def process_excel_file(raw_excel):
    try:
        df = pd.read_excel(raw_excel)
        items = df.to_dict(orient="records")
        updated_items = validate_and_update_items(items)
        updated_df = pd.DataFrame(updated_items)
        updated_json = json.dumps(updated_items, indent=2)
        return updated_df, updated_json
    except Exception as e:
        st.error(f"Excel Processing error: {e}")
        return None, None

# --- Streamlit UI ---
st.title("File Upload and Processing")

uploaded_file = st.file_uploader(
    "Upload a file (CSV, JSON, PDF, Excel)", 
    type=["csv", "json", "pdf", "xlsx", "xls"],
    accept_multiple_files=False
)

if uploaded_file:
    file_type = uploaded_file.type
    file_name = uploaded_file.name

    if file_name.endswith((".xlsx", ".xls")):
        try:
            # Read Excel file, first sheet by default
            df = pd.read_excel(uploaded_file, sheet_name=0)

            # Columns to drop (check if they exist first)
            drop_cols = ['HuggingFace - Llama3-OpenBioLLM-70B', 
                         'Anthropic- Claude 3.7 Sonnet', 
                         'OpenAI -GPT-4o (Clinical Note)', 
                         'OpenAI -GPT-4o (Json)', 
                         'Google - Med-PaLM']
            existing_cols_to_drop = [col for col in drop_cols if col in df.columns]
            df = df.drop(columns=existing_cols_to_drop)

            st.subheader("Excel Data after dropping specific columns")
            st.dataframe(df.head())
        except Exception as e:
            st.error(f"Excel Processing error: {e}")

    elif file_name.endswith(".csv"):
        df = pd.read_csv(uploaded_file)
        st.subheader("CSV Data")
        st.dataframe(df.head())

    elif file_name.endswith(".json"):
        raw_json = uploaded_file.read().decode("utf-8")
        st.subheader("JSON Data")
        st.json(raw_json)

    elif file_name.endswith(".pdf"):
        st.info("PDF upload detected — implement your PDF processing here")
        # Implement PDF parsing if needed

    else:
        st.warning("Unsupported file type")
