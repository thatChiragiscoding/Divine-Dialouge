from sentence_transformers import SentenceTransformer
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import ollama

# Load Gita data
gita_df = pd.read_csv("cleaned_dataset.csv")

# Load embedding model
model = SentenceTransformer("all-mpnet-base-v2")  # better model!

# Encode corpus
corpus_embeddings = model.encode(
    (gita_df["Cleaned_Question"] + " " + gita_df["Answer"]).tolist(),
    normalize_embeddings=True
)

def generate_response_from_gita(user_query):
    prompt = (
        f"You are a spiritual guide based on Bhagavad Gita.\n"
        f"User asks: {user_query}\n\n"
        f"Give a peaceful, simple answer in 3-5 lines based on Gita teachings."
    )
    try:
        response = ollama.chat(
            model='mistral',
            messages=[{"role": "user", "content": prompt}],
            options={"num_predict": 150}
        )
        return response['message']['content']
    except Exception as e:
        return f"⚠️ LLaMA2 Error: {str(e)}"

def get_gita_response(query, threshold=0.4):
    query_embedding = model.encode([query], normalize_embeddings=True)
    similarities = cosine_similarity(query_embedding, corpus_embeddings)[0]
    max_index = np.argmax(similarities)
    max_score = similarities[max_index]

    if max_score < threshold:
        # No strong match, fallback to LLaMA2
        return f"📖 {generate_response_from_gita(query)}"
    else:
        # Good match found
        matched_answer = gita_df.iloc[max_index]["Answer"]
        return f"📖 {matched_answer}"
