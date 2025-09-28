from flask import Flask, request, jsonify
import os
import json
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import logging
from pathlib import Path
from typing import List, Tuple
import sys

# Initialize Flask app
app = Flask(__name__)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("chatbot.log")
    ]
)
logger = logging.getLogger(__name__)

# Load spaCy NLP model with error handling
try:
    nlp = spacy.load("en_core_web_sm")
    logger.info("Successfully loaded spaCy model: en_core_web_sm")
except Exception as e:
    logger.error(f"Failed to load spaCy model: {e}")
    raise SystemExit("Cannot start application without spaCy model")

# Environment configuration
KNOWLEDGE_BASE_DIR = os.getenv("KNOWLEDGE_BASE_DIR", "knowledge")
PORT = int(os.getenv("PORT", 5000))

# Function to preprocess text using spaCy
def preprocess(text: str) -> str:
    """Preprocess text: lowercase, lemmatize, remove stopwords and non-alpha tokens."""
    try:
        doc = nlp(text.lower())
        return " ".join([token.lemma_ for token in doc if not token.is_stop and token.is_alpha])
    except Exception as e:
        logger.error(f"Error preprocessing text: {e}")
        return ""

# Load all knowledge base files dynamically
def load_knowledge_bases(folder: str = KNOWLEDGE_BASE_DIR) -> Tuple[List[str], List[str]]:
    """Load questions and answers from JSON files in the knowledge base folder."""
    questions, answers = [], []
    folder_path = Path(folder)
    
    if not folder_path.exists():
        logger.error(f"Knowledge base directory {folder} does not exist")
        return questions, answers

    for filename in folder_path.glob("*.json"):
        try:
            with open(filename, "r", encoding="utf-8") as f:
                kb = json.load(f)
                for subject, data in kb.get("subjects", {}).items():
                    for q in data.get("questions", []):
                        for pattern in q.get("patterns", []):
                            processed_pattern = preprocess(pattern)
                            if processed_pattern:  # Only add non-empty patterns
                                questions.append(processed_pattern)
                                answers.append(q["answer"])
            logger.info(f"Loaded knowledge base file: {filename}")
        except Exception as e:
            logger.error(f"Error loading {filename}: {e}")
            continue

    if not questions:
        logger.warning("No questions loaded from knowledge base")
    return questions, answers

# Load data at startup
try:
    questions, answers = load_knowledge_bases()
    logger.info(f"Loaded {len(questions)} questions from knowledge base")
except Exception as e:
    logger.error(f"Failed to load knowledge bases: {e}")
    raise SystemExit("Cannot start application without knowledge base")

# Find the best answer for user input
def find_answer(user_input: str) -> str:
    """Find the most relevant answer based on cosine similarity."""
    try:
        processed_input = preprocess(user_input)
        if not processed_input:
            logger.warning("Empty processed input received")
            return "Sorry, I couldn't process your question."

        vectorizer = TfidfVectorizer().fit_transform([processed_input] + questions)
        similarity = cosine_similarity(vectorizer[0:1], vectorizer[1:]).flatten()
        best_match_idx = similarity.argmax()
        
        if similarity[best_match_idx] < 0.1:  # Threshold for low similarity
            logger.info(f"Low similarity score: {similarity[best_match_idx]}")
            return "I don't have a good match for your question. Please try rephrasing."
        
        return answers[best_match_idx]
    except Exception as e:
        logger.error(f"Error finding answer: {e}")
        return "An error occurred while processing your request."

@app.route("/chat", methods=["POST"])
def chat():
    """Handle chat requests and return the best answer."""
    try:
        data = request.get_json()
        if not data or "message" not in data:
            logger.warning("Invalid or missing JSON data in request")
            return jsonify({"answer": "Please provide a valid JSON with a 'message' field"}), 400
        
        user_message = data.get("message", "").strip()
        if not user_message:
            logger.warning("Empty message received")
            return jsonify({"answer": "Please ask a valid question."}), 400
        
        answer = find_answer(user_message)
        return jsonify({"answer": answer}), 200
    except Exception as e:
        logger.error(f"Error in /chat endpoint: {e}")
        return jsonify({"answer": "An error occurred. Please try again later."}), 500

@app.route("/health", methods=["GET"])
def health():
    """Health check endpoint for monitoring."""
    return jsonify({"status": "healthy", "questions_loaded": len(questions)}), 200

if __name__ == "__main__":
    logger.info(f"Starting Flask app on port {PORT}")
    app.run(host="0.0.0.0", port=PORT, debug=False)
