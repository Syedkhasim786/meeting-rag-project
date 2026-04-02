import streamlit as st
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from openai import OpenAI

# -------------------------------
# Load Model
# -------------------------------
@st.cache_resource
def load_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

model = load_model()

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
    embeddings = model.encode(texts)
    dimension = embeddings.shape[1]
    
    index = faiss.IndexFlatL2(dimension)
    index.add(np.array(embeddings))
    
    return index

index = create_index(texts)

# -------------------------------
# Retrieve Context
# -------------------------------
def retrieve(query, k=3):
    query_embedding = model.encode([query])
    distances, indices = index.search(np.array(query_embedding), k)
    
    results = [texts[i] for i in indices[0]]
    return "\n".join(results)

# -------------------------------
# Generate Response (RAG)
# -------------------------------
client = OpenAI()

def generate_response(transcript):
    context = retrieve(transcript)

    prompt = f"""
Summarise this meeting transcript into:

1. A 3-sentence overview
2. Key decisions made
3. Open questions
4. A numbered action list with:
   - Owner
   - Deadline
   - Reason

Rules:
- If no owner → mark as **Owner: TBD**
- Highlight conflicts under "Conflicts to resolve"

Context from previous meetings:
{context}

Transcript:
{transcript}
"""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "user", "content": prompt}
        ]
    )

    return response.choices[0].message.content

# -------------------------------
# Streamlit UI
# -------------------------------
st.title("📊 Meeting Notes → Action Items (RAG)")

transcript = st.text_area("Paste your meeting transcript here...")

if st.button("Generate Action Items"):
    if transcript.strip() != "":
        result = generate_response(transcript)
        st.write(result)
    else:
        st.warning("Please enter a meeting transcript.")
