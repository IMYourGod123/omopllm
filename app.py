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

# Load embedding model
device = "cuda" if torch.cuda.is_available() else "cpu"
st.info(f"Using device: {device}")
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


def process_dataframe(df):
    updated_rows = []
    correct_count = 0
    total = 0

    for idx, row in df.iterrows():
        clinical_item = row.get("Clinical Item") or row.get("clinical_item")
        category = row.get("Category") or row.get("category")
        description = row.get("Description") or row.get("description", "")
        gold_code = row.get("Code") or row.get("code", None)

        if pd.isna(clinical_item) or pd.isna(category):
            # Skip rows without these
            updated_rows.append(row)
            continue

        match = llm_validated_concept_match(str(clinical_item), str(category), str(description))

        updated_row = row.copy()
        if match:
            updated_row["Predicted Concept Name"] = match.get("concept_name")
            updated_row["Predicted Code"] = match.get("concept_id")
            updated_row["Predicted Domain"] = match.get("domain_id")
            updated_row["Coding System"] = "OMOPCDM"

            if gold_code and not pd.isna(gold_code):
                total += 1
                if str(gold_code).strip() == str(match.get("concept_id")).strip():
                    correct_count += 1
        else:
            updated_row["Predicted Concept Name"] = "No match"
            updated_row["Predicted Code"] = None

        updated_rows.append(updated_row)

    result_df = pd.DataFrame(updated_rows)
    accuracy = (correct_count / total * 100) if total > 0 else None
    return result_df, accuracy


# --- Streamlit UI ---
st.title("🧠 OMOP LLM Concept Validator")

uploaded_file = st.file_uploader("Upload file", type=["json", "csv", "pdf"])

if uploaded_file:
    file_type = uploaded_file.type
    filename = uploaded_file.name

    if filename.endswith(".json"):
        raw_json = uploaded_file.read().decode("utf-8")
        st.subheader("Original JSON")
        st.code(raw_json, language="json")

        if st.button("Validate with LLM"):
            st.info("Running validation on JSON...")
            result_json = validate_and_update_json(raw_json)
            st.subheader("Validated JSON")
            st.code(result_json, language="json")

            df = pd.read_json(result_json)
            st.dataframe(df)

            st.download_button("Download Validated JSON", result_json, file_name="validated_output.json", mime="application/json")

    elif filename.endswith(".csv"):
        df = pd.read_csv(uploaded_file)
        st.subheader("Uploaded CSV Data")
        st.dataframe(df)

        if st.button("Validate with LLM"):
            st.info("Running validation on CSV...")
            result_df, accuracy = process_dataframe(df)

            st.subheader("Validated CSV Data")
            st.dataframe(result_df)

            if accuracy is not None:
                st.success(f"Accuracy (matching 'Code' column): {accuracy:.2f}%")

            csv_output = result_df.to_csv(index=False)
            st.download_button("Download Validated CSV", csv_output, file_name="validated_output.csv", mime="text/csv")

    elif filename.endswith(".pdf"):
        st.warning("PDF upload detected. PDF processing is not yet supported.")
        # Optional: you can integrate PDF text extraction libraries like PyMuPDF or pdfplumber here
    else:
        st.error("Unsupported file type. Please upload JSON, CSV, or PDF.")
