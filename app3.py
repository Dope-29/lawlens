import os
import re
import string
import nltk
from flask import Flask, render_template, request, jsonify
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Ensure necessary NLTK components are available
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)

app = Flask(__name__)

# Path to text files
FOLDER_PATH = "C:/Users/dany2/OneDrive/Desktop/lawlens_ai/uploads/outputs"

# NLP Utilities
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def preprocess_text(text):
    """Clean, tokenize, remove stopwords, and lemmatize text."""
    text = text.lower()
    text = re.sub(r'\d+', '', text)  # Remove numbers
    text = text.translate(str.maketrans('', '', string.punctuation))  # Remove punctuation
    tokens = word_tokenize(text)  # Tokenize
    processed_tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]
    return processed_tokens if processed_tokens else tokens  # Ensure at least one word remains

def load_text_files():
    """Load all text files from the folder."""
    if not os.path.exists(FOLDER_PATH):
        print("DEBUG: Folder path does not exist:", FOLDER_PATH)
        return {}

    text_files = {}
    for filename in os.listdir(FOLDER_PATH):
        if filename.endswith(".txt"):
            with open(os.path.join(FOLDER_PATH, filename), "r", encoding="utf-8") as file:
                text_files[filename] = file.read()
    return text_files

@app.route('/')
def index():
    """Render the main search page."""
    return render_template("index.html")

@app.route('/search', methods=['POST'])
def search():
    """Handle search queries and return ranked results."""
    data = request.json
    user_input = data.get("query", "").strip()

    if not user_input:
        return jsonify({"error": "Please enter a valid search term."})

    text_files = load_text_files()
    if not text_files:
        return jsonify({"error": "No text files available."})

    input_tokens = preprocess_text(user_input)
    if not input_tokens:
        return jsonify({"error": "Query too short or contains only stopwords."})

    input_text = " ".join(input_tokens)

    # Prepare corpus for TF-IDF
    filenames, documents = zip(*text_files.items())
    vectorizer = TfidfVectorizer(stop_words="english")  # Now handling stopwords here
    tfidf_matrix = vectorizer.fit_transform([input_text] + list(documents))
    similarities = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:]).flatten()

    # Rank results
    ranked_results = sorted(
        [(filenames[i], documents[i][:300] + "...", similarities[i]) for i in range(len(filenames))],
        key=lambda x: x[2], reverse=True
    )

    # Take top matches (with a similarity threshold)
    top_matches = [{"filename": f, "content": c} for f, c, s in ranked_results if s > 0.02]

    return jsonify({"matches": top_matches if top_matches else None})

if __name__ == "__main__":
    app.run(debug=True)
