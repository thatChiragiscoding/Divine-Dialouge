import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle  # To save the TF-IDF model for later use

# Load the preprocessed dataset
df = pd.read_csv("cleaned_dataset.csv")

# Initialize TF-IDF Vectorizer
vectorizer = TfidfVectorizer()

# Convert the "Processed_Question" column into TF-IDF vectors
X = vectorizer.fit_transform(df["Processed_Question"])

# Save the vectorizer and transformed data for later use
with open("tfidf_vectorizer.pkl", "wb") as f:
    pickle.dump(vectorizer, f)

with open("tfidf_matrix.pkl", "wb") as f:
    pickle.dump(X, f)

print("TF-IDF vectorization complete! Saved vectorizer and matrix.")
