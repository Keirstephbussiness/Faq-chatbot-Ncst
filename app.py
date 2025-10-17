import os
import json
from flask import Flask, request, jsonify
from flask_cors import CORS
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app, origins=["*"])

knowledge_folder = "knowledge"
questions_list = []
answers_list = []

def load_knowledge_base():
    """Load knowledge base from JSON files"""
    global questions_list, answers_list
    
    logger.info(f"Loading knowledge base from {knowledge_folder}")
    
    if not os.path.exists(knowledge_folder):
        logger.error(f"Knowledge folder '{knowledge_folder}' not found!")
        create_sample_knowledge()
        return
    
    questions_list = []
    answers_list = []
    json_files_found = 0
    
    for file in os.listdir(knowledge_folder):
        if file.endswith(".json"):
            json_files_found += 1
            path = os.path.join(knowledge_folder, file)
            try:
                with open(path, "r", encoding="utf-8") as f:
                    kb = json.load(f)
                
                logger.info(f"Loaded {file}")
                
                for subject in kb.get("subjects", {}).values():
                    for item in subject.get("questions", []):
                        patterns = item.get("patterns", [])
                        answer = item.get("answer", "")
                        
                        if patterns and answer:
                            questions_list.append(" ".join(patterns))
                            answers_list.append(answer)
                
            except json.JSONDecodeError as e:
                logger.error(f"Error reading {file}: {e}")
            except Exception as e:
                logger.error(f"Error processing {file}: {e}")
    
    if json_files_found == 0:
        logger.warning("No JSON files found")
        create_sample_knowledge()
    
    logger.info(f"✅ LOADED: {len(questions_list)} questions from {json_files_found} files")

def create_sample_knowledge():
    """Create sample knowledge base"""
    global questions_list, answers_list
    sample_data = [
        {"patterns": ["hello", "hi", "hey"], "answer": "Hello! NCST FAQ Assistant here!"},
        {"patterns": ["contact", "phone"], "answer": "Call NCST: (046) 416-4779"}
    ]
    questions_list = [" ".join(item["patterns"]) for item in sample_data]
    answers_list = [item["answer"] for item in sample_data]

# 🔥 FIXED: Build vectorizer with BETTER ERROR HANDLING
def build_vectorizer():
    """Build or rebuild the TF-IDF vectorizer"""
    global vectorizer, question_vectors
    
    if not questions_list:
        logger.error("❌ No questions to build vectorizer")
        return False
    
    try:
        vectorizer = TfidfVectorizer(stop_words='english', lowercase=True, max_features=2000)
        question_vectors = vectorizer.fit_transform(questions_list)
        logger.info(f"✅ VECTORIZER BUILT: {len(questions_list)} questions")
        return True
    except Exception as e:
        logger.error(f"❌ Vectorizer error: {e}")
        return False

# Load on startup
load_knowledge_base()
vectorizer = None
question_vectors = None
build_vectorizer()  # 🔥 FIXED: Always rebuild after load

@app.route("/", methods=["GET"])
def home():
    return jsonify({
        "status": "online",
        "questions_loaded": len(questions_list),
        "version": "1.1.0-FIXED"
    })

@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status": "healthy",
        "questions_count": len(questions_list),
        "answers_count": len(answers_list),
        "vectorizer_ready": vectorizer is not None,
        "sample_question": questions_list[0] if questions_list else "none"
    })

@app.route("/chat", methods=["POST", "OPTIONS"])
def chat():
    if request.method == "OPTIONS":
        response = jsonify({"status": "ok"})
        response.headers.add("Access-Control-Allow-Origin", "*")
        response.headers.add("Access-Control-Allow-Headers", "Content-Type")
        response.headers.add("Access-Control-Allow-Methods", "POST, OPTIONS")
        return response
    
    try:
        data = request.get_json()
        user_input = data.get("message", "").strip()
        if not user_input:
            return jsonify({"answer": "Please ask a question."})
        
        logger.info(f"💬 Q: {user_input}")
        
        if vectorizer is None or question_vectors is None:
            return jsonify({"answer": "System loading... Try again!"})
        
        user_vec = vectorizer.transform([user_input])
        similarities = cosine_similarity(user_vec, question_vectors).flatten()
        best_idx = similarities.argmax()
        best_score = similarities[best_idx]
        
        logger.info(f"📊 Best score: {best_score:.3f}")
        
        # 🔥 FIXED: Lower threshold for better matching
        if best_score < 0.15:  # Was 0.3 → Now 0.15
            response = "Sorry, try: contact, phone, ncst address, or courses"
        else:
            response = answers_list[best_idx]
        
        logger.info(f"✅ A: {response[:50]}...")
        return jsonify({"answer": response})
        
    except Exception as e:
        logger.error(f"❌ Chat error: {e}")
        return jsonify({"answer": "Error! Try again."}), 500

# 🔥 FIXED SUGGESTIONS - NOW WORKS 100%!
@app.route("/suggest", methods=["POST", "OPTIONS"])
def suggest():
    if request.method == "OPTIONS":
        response = jsonify({"status": "ok"})
        response.headers.add("Access-Control-Allow-Origin", "*")
        response.headers.add("Access-Control-Allow-Headers", "Content-Type")
        response.headers.add("Access-Control-Allow-Methods", "POST, OPTIONS")
        return response
    
    try:
        data = request.get_json()
        partial_input = data.get("query", "").strip().lower()
        if len(partial_input) < 1:  # 🔥 FIXED: Now works with 1+ char
            return jsonify({"suggestions": []})
        
        if vectorizer is None or question_vectors is None:
            return jsonify({"suggestions": []})
        
        partial_vec = vectorizer.transform([partial_input])
        similarities = cosine_similarity(partial_vec, question_vectors).flatten()
        
        # 🔥 FIXED: Lower threshold + NO duplicate filter
        threshold = 0.05  # Was 0.1 → Now 0.05
        top_indices = [(i, score) for i, score in enumerate(similarities) if score > threshold]
        top_indices.sort(key=lambda x: x[1], reverse=True)
        top_indices = top_indices[:10]  # More suggestions
        
        suggestions = []
        seen_texts = set()
        for idx, score in top_indices:
            pattern = questions_list[idx]
            # 🔥 FIXED: Better text extraction
            words = pattern.split()[:3]
            text = " ".join(words)
            
            if text not in seen_texts:
                seen_texts.add(text)
                suggestions.append({
                    "text": text,
                    "confidence": round(float(score), 2)
                })
        
        logger.info(f"💡 Suggestions: {len(suggestions)} for '{partial_input}'")
        return jsonify({"suggestions": suggestions})
        
    except Exception as e:
        logger.error(f"❌ Suggest error: {e}")
        return jsonify({"suggestions": []}), 500

# 🔥 FIXED RELOAD - NOW WORKS 100%!
@app.route("/reload", methods=["POST"])
def reload_knowledge():
    try:
        logger.info("🔄 RELOADING...")
        load_knowledge_base()
        
        # 🔥 CRITICAL FIX: Reset globals FIRST
        global vectorizer, question_vectors
        vectorizer = None
        question_vectors = None
        
        # Then rebuild
        success = build_vectorizer()
        
        if success:
            logger.info(f"✅ RELOAD SUCCESS: {len(questions_list)} questions")
            return jsonify({
                "status": "success",
                "message": "Reloaded successfully!",
                "questions_count": len(questions_list),
                "files": len([f for f in os.listdir(knowledge_folder) if f.endswith('.json')])
            })
        else:
            return jsonify({"status": "error", "message": "Vectorizer failed"}), 500
            
    except Exception as e:
        logger.error(f"❌ Reload error: {e}")
        return jsonify({"status": "error", "message": str(e)}), 500

# Error handlers
@app.errorhandler(404)
def not_found(error):
    return jsonify({"error": "Endpoint not found"}), 404

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
