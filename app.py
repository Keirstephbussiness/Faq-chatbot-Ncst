import os
import json
from flask import Flask, request, jsonify
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)

# Folder where JSON knowledge files are stored
knowledge_folder = "knowledge"

questions_list = []
answers_list = []

# Loop through all JSON files in the folder
for file in os.listdir(knowledge_folder):
    if file.endswith(".json"):
        path = os.path.join(knowledge_folder, file)
        with open(path, "r", encoding="utf-8") as f:
            kb = json.load(f)
        # Flatten all questions
        for subject in kb.get("subjects", {}).values():
            for item in subject.get("questions", []):
                questions_list.append(" ".join(item.get("patterns", [])))
                answers_list.append(item.get("answer", ""))

# Build TF-IDF vectorizer
vectorizer = TfidfVectorizer().fit(questions_list)
question_vectors = vectorizer.transform(questions_list)

@app.route("/chat", methods=["POST"])
def chat():
    user_input = request.json.get("message", "")
    if not user_input:
        return jsonify({"answer": "Please ask a question."})
    
    user_vec = vectorizer.transform([user_input])
    similarities = cosine_similarity(user_vec, question_vectors).flatten()
    best_idx = similarities.argmax()
    
    if similarities[best_idx] < 0.3:
        return jsonify({"answer": "Sorry, I don't know the answer to that."})
    
    return jsonify({"answer": answers_list[best_idx]})

if __name__ == "__main__":
    app.run(debug=True)
