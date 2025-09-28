import os
import json
from flask import Flask, request, jsonify
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)

# Load all knowledge base JSON files
knowledge_dir = "knowledge"
questions, answers = [], []

for file in os.listdir(knowledge_dir):
    if file.endswith(".json"):
        with open(os.path.join(knowledge_dir, file), "r") as f:
            kb = json.load(f)
            for q in kb["questions"]:
                for pattern in q["patterns"]:
                    questions.append(pattern)
                    answers.append(q["answer"])

# TF-IDF setup
vectorizer = TfidfVectorizer().fit(questions)
question_vectors = vectorizer.transform(questions)

def find_answer(user_input):
    user_vec = vectorizer.transform([user_input])
    similarity = cosine_similarity(user_vec, question_vectors).flatten()
    best_match = similarity.argmax()
    score = similarity[best_match]

    if score < 0.2:  # Threshold to avoid random bad matches
        return "I'm not sure about that. Please ask something related to NCST."
    return answers[best_match]

@app.route("/chat", methods=["POST"])
def chat():
    data = request.json
    user_message = data.get("message", "")
    return jsonify({"answer": find_answer(user_message)})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
