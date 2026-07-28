import os

import numpy as np
from sentence_transformers import SentenceTransformer
import anthropic

# ---- 0. Your knowledge base (replace with real docs) ----
DOCUMENTS = [
    "The refund policy allows returns within 30 days of purchase with a receipt.",
    "Our headquarters is located in Austin, Texas, opened in 2019.",
    "Premium subscribers get 24/7 priority support and a dedicated account manager.",
    "The mobile app supports iOS 15+ and Android 11+ as of this release.",
    "Data is encrypted at rest using AES-256 and in transit using TLS 1.3.",
]

# ---- 1. CHUNKING ----
# Real docs need splitting. Here docs are short, so 1 doc = 1 chunk.
# For long text, split on paragraphs with overlap (see helper below).
def chunk_text(text, size=500, overlap=50):
    words = text.split()
    chunks, i = [], 0
    while i < len(words):
        chunks.append(" ".join(words[i:i + size]))
        i += size - overlap
    return chunks

chunks = DOCUMENTS  # already chunk-sized

# ---- 2. EMBEDDING ----
embedder = SentenceTransformer("all-MiniLM-L6-v2")   # 384-dim, fast, local
chunk_vectors = embedder.encode(chunks, normalize_embeddings=True)

# ---- 3. RETRIEVAL (cosine similarity = dot product on normalized vectors) ----
def retrieve(query, k=3):
    q = embedder.encode([query], normalize_embeddings=True)[0]
    scores = chunk_vectors @ q                 # cosine similarity to every chunk
    top = np.argsort(scores)[::-1][:k]         # indices of top-k
    return [(chunks[i], float(scores[i])) for i in top]

# ---- 4. GENERATION (grounded in retrieved context) ----
client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])

def answer(query, k=3):
    hits = retrieve(query, k)
    context = "\n".join(f"- {c}" for c, _ in hits)
    prompt = (
        "Answer the question using ONLY the context below. "
        "If the answer isn't in the context, say you don't know.\n\n"
        f"Context:\n{context}\n\nQuestion: {query}"
    )
    resp = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=300,
        temperature=0,                          # factual → deterministic
        messages=[{"role": "user", "content": prompt}],
    )
    return resp.content[0].text, hits

# ---- Try it ----
ans, sources = answer("How long do I have to return something?")
print(ans)
print("\nRetrieved:", [round(s, 3) for _, s in sources])