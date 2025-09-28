from flask import Flask, request, jsonify
import os
import json
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)

# Load spaCy NLP model (English for now, can add multilingual later)
nlp = spacy.load("en_core_web_sm")

# Function to preprocess text using spaCy (lemmatization, lowercasing, stopword removal)
def preprocess(text):
    doc = nlp(text.lower())
    return " ".join([token.lemma_ for token in doc if not token.is_stop and token.is_alpha])

# Load all knowledge base files dynamically
def load_knowledge_bases(folder="knowledge"):
    questions, answers = [], []
    for filename in os.listdir(folder):
        if filename.endswith(".json"):
            with open(os.path.join(folder, filename), "r", encoding="utf-8") as f:
                kb = json.load(f)
                for subject, data in kb.get("subjects", {}).items():
                    for q in data.get("questions", []):
                        for pattern in q.get("patterns", []):
                            questions.append(preprocess(pattern))
                            answers.append(q["answer"])
    return questions, answers

# Load data once at startup
questions, answers = load_knowledge_bases()

def find_answer(user_input):
    processed_input = preprocess(user_input)
    vectorizer = TfidfVectorizer().fit_transform([processed_input] + questions)
    similarity = cosine_similarity(vectorizer[0:1], vectorizer[1:]).flatten()
    best_match = similarity.argmax()
    return answers[best_match]

@app.route("/chat", methods=["POST"])
def chat():
    data = request.json
    user_message = data.get("message", "")
    if not user_message.strip():
        return jsonify({"answer": "Please ask a valid question."})
    return jsonify({"answer": find_answer(user_message)})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
