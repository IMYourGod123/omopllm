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
st.title("🧠 OMOP LLM Concept Validator")

uploaded_files = st.file_uploader(
    "Upload one or more files (CSV, JSON, PDF, Excel)", 
    type=["csv", "json", "pdf", "xls", "xlsx"], 
    accept_multiple_files=True
)

if uploaded_files:
    for uploaded_file in uploaded_files:
        st.write(f"### File: {uploaded_file.name}")
        file_type = uploaded_file.type

        if uploaded_file.name.lower().endswith(".json") or file_type == "application/json":
            raw_json = uploaded_file.read().decode("utf-8")
            st.subheader("Original JSON")
            st.code(raw_json, language="json")

            if st.button(f"Validate {uploaded_file.name}"):
                st.info("Running validation...")
                validated_json_str, updated_items = process_json_file(raw_json)
                if updated_items is not None:
                    st.subheader("Validated JSON")
                    st.code(validated_json_str, language="json")

                    df = pd.DataFrame(updated_items)
                    st.dataframe(df)

                    st.download_button(
                        label=f"Download Validated JSON - {uploaded_file.name}",
                        data=validated_json_str,
                        file_name=f"validated_{uploaded_file.name}",
                        mime="application/json"
                    )

        elif uploaded_file.name.lower().endswith(".csv") or file_type == "text/csv":
            df, validated_json_str = process_csv_file(uploaded_file)
            if df is not None:
                st.subheader("Validated CSV Data")
                st.dataframe(df)

                st.download_button(
                    label=f"Download Validated JSON - {uploaded_file.name}",
                    data=validated_json_str,
                    file_name=f"validated_{uploaded_file.name}.json",
                    mime="application/json"
                )

        elif uploaded_file.name.lower().endswith((".xls", ".xlsx")):
            df, validated_json_str = process_excel_file(uploaded_file)
            if df is not None:
                st.subheader("Validated Excel Data")
                st.dataframe(df)

                st.download_button(
                    label=f"Download Validated JSON - {uploaded_file.name}",
                    data=validated_json_str,
                    file_name=f"validated_{uploaded_file.name}.json",
                    mime="application/json"
                )

        elif uploaded_file.name.lower().endswith(".pdf") or file_type == "application/pdf":
            st.write("PDF file uploaded. Displaying download option.")
            st.download_button(
                label=f"Download {uploaded_file.name}",
                data=uploaded_file.getvalue(),
                file_name=uploaded_file.name,
                mime="application/pdf"
            )

        else:
            st.warning(f"Unsupported file type for {uploaded_file.name}")
