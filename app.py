import streamlit as st
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

# -------------------------------
# Load Embedding Model
# -------------------------------
@st.cache_resource
def load_embed_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

embed_model = load_embed_model()

# -------------------------------
# Load Summarization Model (FIXED)
# -------------------------------
@st.cache_resource
def load_summarizer():
    from transformers import pipeline
    return pipeline("text2text-generation", model="google/flan-t5-base")

summarizer = load_summarizer()

# -------------------------------
# Load Data
# -------------------------------
def load_data():
    try:
        with open("data/meetings.txt", "r") as f:
            texts = f.readlines()
    except:
        texts = ["No previous meeting data available."]
    return texts

texts = load_data()

# -------------------------------
# Create FAISS Index
# -------------------------------
def create_index(texts):
    embeddings = embed_model.encode(texts)
    dimension = embeddings.shape[1]

    index = faiss.IndexFlatL2(dimension)
    index.add(np.array(embeddings))

    return index

index = create_index(texts)

# -------------------------------
# Retrieve Context
# -------------------------------
def retrieve(query, k=3):
    query_embedding = embed_model.encode([query])
    distances, indices = index.search(np.array(query_embedding), k)

    results = [texts[i] for i in indices[0]]
    return "\n".join(results)

# -------------------------------
# Generate Response (NO API)
# -------------------------------
def generate_response(transcript):
    context = retrieve(transcript)
    full_text = context + "\n" + transcript

    # Summarization (FIXED)
    summary = summarizer(
        f"summarize: {full_text}",
        max_length=150,
        do_sample=False
    )[0]['generated_text']

    # Rule-based Action Extraction
    action_items = []
    lines = transcript.split(".")
    for line in lines:
        if "will" in line.lower():
            action_items.append(line.strip())

    return f"""
### 📌 Summary:
{summary}

### ✅ Action Items:
{chr(10).join(action_items) if action_items else "No clear actions found"}

### ⚠️ Note:
Owners and deadlines may need manual review.
"""

# -------------------------------
# Streamlit UI
# -------------------------------
st.title("📊 Meeting Notes → Action Items (FREE RAG)")

transcript = st.text_area("Paste your meeting transcript here...")

if st.button("Generate Action Items"):
    if transcript.strip() != "":
        result = generate_response(transcript)
        st.write(result)
    else:
        st.warning("Please enter a meeting transcript.")
