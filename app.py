import streamlit as st
import pandas as pd
import pickle
import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.metrics.pairwise import cosine_similarity

# Load necessary NLP resources
nltk.download("punkt")
nltk.download("stopwords")
nltk.download("wordnet")

# Load the cleaned dataset
df = pd.read_csv("cleaned_dataset.csv")

# Load the saved TF-IDF vectorizer and matrix
with open("tfidf_vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)

with open("tfidf_matrix.pkl", "rb") as f:
    X = pickle.load(f)

# Initialize NLP tools
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words("english"))

# Function to preprocess user input
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r"[^\w\s]", "", text)
    words = word_tokenize(text)
    words = [lemmatizer.lemmatize(word) for word in words if word not in stop_words]
    return " ".join(words)

# Function to find the best matching question and return the answer
def get_answer(user_query):
    processed_query = preprocess_text(user_query)
    query_vector = vectorizer.transform([processed_query])
    similarities = cosine_similarity(query_vector, X)
    best_match_idx = similarities.argmax()
    return df.iloc[best_match_idx]["Answer"]

# Streamlit UI
st.title("Welcome to Divine Dialouge")
st.write("Find Your Solutions in the Divine Words of Bhagwad Gita!")

# Store chat history in session state
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

# User input box (always visible)
user_input = st.chat_input("Type your question here...")

# Process user input
if user_input:
    # If user types "exit", end the conversation
    if user_input.lower() == "exit":
        goodbye_message = "Goodbye! Have a great day! 😊"
        st.session_state.messages.append({"role": "assistant", "content": goodbye_message})
        st.write(goodbye_message)  # Show the message
        st.stop()  # Stop execution (prevents further interaction)

    # Save user query to chat history
    st.session_state.messages.append({"role": "user", "content": user_input})
    
    # Get chatbot response
    response = get_answer(user_input)
    
    # Save chatbot response to chat history
    st.session_state.messages.append({"role": "assistant", "content": response})

    # Rerun app to update UI
    st.rerun()