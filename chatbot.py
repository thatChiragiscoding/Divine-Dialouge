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
    text = text.lower()  # Convert to lowercase
    text = re.sub(r"[^\w\s]", "", text)  # Remove punctuation
    words = word_tokenize(text)  # Tokenize
    words = [lemmatizer.lemmatize(word) for word in words if word not in stop_words]  # Remove stopwords & lemmatize
    return " ".join(words)

# Function to find the best matching question and return the answer
def get_answer(user_query):
    processed_query = preprocess_text(user_query)  # Preprocess user input
    query_vector = vectorizer.transform([processed_query])  # Convert to TF-IDF vector
    similarities = cosine_similarity(query_vector, X)  # Compute similarity
    best_match_idx = similarities.argmax()  # Get index of best match
    return df.iloc[best_match_idx]["Answer"]  # Return corresponding answer

# Start Chatbot
print("🔵 GitaChat: Ask me anything from the Bhagavad Gita! (Type 'exit' to stop)")
while True:
    user_input = input("You: ")
    if user_input.lower() == "exit":
        print("🔵 GitaChat: Goodbye! 🙏")
        break
    response = get_answer(user_input)
    print(f"🔵 GitaChat: {response}\n")
