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

def validate_and_update_json(raw_json_str):
    try:
        items = json.loads(raw_json_str)
        updated_items = []

        for item in items:
            clinical_item = item["Clinical Item"]
            category = item["Category"]
            description = item.get("Description", "")

            match = llm_validated_concept_match(clinical_item, category, description)

            if match:
                item["Clinical Item"] = match["concept_name"]
                item["Code"] = match["concept_id"]
                item["Coding System"] = "OMOPCDM"
                item["Category"] = match["domain_id"]
            updated_items.append(item)

        return json.dumps(updated_items, indent=2)

    except Exception as e:
        st.error(f"Validation error: {e}")
        return raw_json_str

# --- Streamlit UI ---
st.title("🧠 OMOP LLM Concept Validator")

uploaded_file = st.file_uploader("Upload JSON file", type=["json"])
if uploaded_file:
    raw_json = uploaded_file.read().decode("utf-8")
    st.subheader("Original JSON")
    st.code(raw_json, language="json")

    if st.button("Validate with LLM"):
        st.info("Running validation...")
        result_json = validate_and_update_json(raw_json)

        st.subheader("Validated JSON")
        st.code(result_json, language="json")

        df = pd.read_json(result_json)
        st.dataframe(df)

        st.download_button("Download Validated JSON", result_json, file_name="validated_output.json", mime="application/json")
