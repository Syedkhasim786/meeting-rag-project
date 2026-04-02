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
# Simple Summary Function
# -------------------------------
def simple_summary(text):
    sentences = text.split(".")
    return ".".join(sentences[:3]) if len(sentences) >= 3 else text

# -------------------------------
# Load Data
# -------------------------------
def load_data():
    try:
        with open("data/meetings.txt", "r") as f:
            texts = f.readlines()
    except:
        texts = ["Project discussion with no prior data."]
    return texts

texts = load_data()

# -------------------------------
# Create FAISS Index
# -------------------------------
@st.cache_resource
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
# Extract Action Items
# -------------------------------
def extract_actions(transcript):
    actions = []
    lines = transcript.split(".")

    for line in lines:
        line = line.strip()
        if "will" in line.lower():
            actions.append(f"- {line}")

    return actions if actions else ["No clear action items found"]

# -------------------------------
# Generate Response
# -------------------------------
def generate_response(transcript):
    context = retrieve(transcript)
    full_text = context + "\n" + transcript

    summary = simple_summary(full_text)
    actions = extract_actions(transcript)

    return f"""
### 📌 Summary:
{summary}

### ✅ Action Items:
{chr(10).join(actions)}

### ⚠️ Note:
Owners and deadlines may need manual review.
"""

# -------------------------------
# Streamlit UI
# -------------------------------
st.title("📊 Meeting Notes → Action Items (FREE RAG)")

st.write("Paste your meeting transcript and generate summary & actions.")

transcript = st.text_area("Enter transcript here...")

if st.button("Generate Action Items"):
    if transcript.strip():
        result = generate_response(transcript)
        st.markdown(result)
    else:
        st.warning("Please enter a meeting transcript.")
