import pandas as pd
import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Download necessary NLP resources (only needed once)
nltk.download("punkt")
nltk.download("stopwords")
nltk.download("wordnet")

# Load dataset with proper encoding
df = pd.read_csv("geeta.csv", encoding="ISO-8859-1")  # Try changing encoding if needed

# Initialize NLP tools
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words("english"))

# Function to preprocess text
def preprocess_text(text):
    if isinstance(text, str):  # Check if text is valid
        text = text.lower()  # Convert to lowercase
        text = re.sub(r"[^\w\s]", "", text)  # Remove punctuation
        words = word_tokenize(text)  # Tokenize
        words = [lemmatizer.lemmatize(word) for word in words if word not in stop_words]  # Remove stopwords & lemmatize
        return " ".join(words)
    return ""

# Apply preprocessing to the "Question" column
df["Processed_Question"] = df["Question"].apply(preprocess_text)

# Save the cleaned dataset
df.to_csv("cleaned_dataset.csv", index=False, encoding="utf-8")

print("Preprocessing complete! Saved as cleaned_dataset.csv.")
