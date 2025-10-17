import os
import json
from flask import Flask, request, jsonify, session
from flask_cors import CORS
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app, origins=["*"])

# 🔥 NEW: Secret key for sessions (context)
app.secret_key = os.urandom(24)  # Or set to a fixed string in prod

knowledge_folder = "knowledge"
kb_items = []  # 🔥 IMPROVED: Structured storage for better suggestions

def load_knowledge_base():
    """Load knowledge base from JSON files"""
    global kb_items
    
    logger.info(f"Loading knowledge base from {knowledge_folder}")
    
    if not os.path.exists(knowledge_folder):
        logger.error(f"Knowledge folder '{knowledge_folder}' not found!")
        create_sample_knowledge()
        return
    
    kb_items = []
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
                            # 🔥 IMPROVED: Store representative (first pattern) for suggestions
                            kb_items.append({
                                "joined_patterns": " ".join(patterns),
                                "representative": patterns[0],
                                "answer": answer
                            })
                
            except json.JSONDecodeError as e:
                logger.error(f"Error reading {file}: {e}")
            except Exception as e:
                logger.error(f"Error processing {file}: {e}")
    
    if json_files_found == 0:
        logger.warning("No JSON files found")
        create_sample_knowledge()
    
    logger.info(f"✅ LOADED: {len(kb_items)} questions from {json_files_found} files")

def create_sample_knowledge():
    """Create sample knowledge base"""
    global kb_items
    sample_data = [
        {"patterns": ["hello", "hi", "hey"], "answer": "Hello! NCST FAQ Assistant here!"},
        {"patterns": ["contact", "phone"], "answer": "Call NCST: (046) 416-4779"}
    ]
    kb_items = [{
        "joined_patterns": " ".join(item["patterns"]),
        "representative": item["patterns"][0],
        "answer": item["answer"]
    } for item in sample_data]

# 🔥 IMPROVED: Vectorizer with n-grams for better phrase matching
def build_vectorizer():
    """Build or rebuild the TF-IDF vectorizer"""
    global vectorizer, question_vectors
    
    if not kb_items:
        logger.error("❌ No questions to build vectorizer")
        return False
    
    try:
        vectorizer = TfidfVectorizer(
            stop_words='english', 
            lowercase=True, 
            max_features=5000,  # Increased
            ngram_range=(1, 3)  # 🔥 NEW: Captures phrases like "fee structure"
        )
        question_vectors = vectorizer.fit_transform([item["joined_patterns"] for item in kb_items])
        logger.info(f"✅ VECTORIZER BUILT: {len(kb_items)} questions")
        return True
    except Exception as e:
        logger.error(f"❌ Vectorizer error: {e}")
        return False

# Load on startup
load_knowledge_base()
vectorizer = None
question_vectors = None
build_vectorizer()

@app.route("/", methods=["GET"])
def home():
    return jsonify({
        "status": "online",
        "questions_loaded": len(kb_items),
        "version": "1.2.0-IMPROVED"
    })

@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status": "healthy",
        "questions_count": len(kb_items),
        "vectorizer_ready": vectorizer is not None,
        "sample_rep": kb_items[0]["representative"] if kb_items else "none"
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
        
        # 🔥 NEW: Add context from session history (last 3 exchanges)
        if 'history' not in session:
            session['history'] = []
        
        # Build contextual input: append last 3 messages (user + bot)
        context = " ".join(session['history'][-6:])  # Last 3 exchanges (6 msgs)
        full_input = f"{context} {user_input}".strip()
        
        user_vec = vectorizer.transform([full_input])
        similarities = cosine_similarity(user_vec, question_vectors).flatten()
        best_idx = similarities.argmax()
        best_score = similarities[best_idx]
        
        logger.info(f"📊 Best score: {best_score:.3f} (with context: '{context[:50]}...')")
        
        if best_score < 0.1:  # Lowered for better recall
            response = "Sorry, I couldn't find a match. Try rephrasing or ask about admissions, fees, courses, etc. What was your previous question about?"
        else:
            response = kb_items[best_idx]["answer"]
        
        # 🔥 NEW: Update history
        session['history'].append(user_input)
        session['history'].append(response)
        if len(session['history']) > 20:  # Limit history
            session['history'] = session['history'][-20:]
        
        logger.info(f"✅ A: {response[:50]}...")
        return jsonify({"answer": response})
        
    except Exception as e:
        logger.error(f"❌ Chat error: {e}")
        return jsonify({"answer": "Error! Try again."}), 500

# 🔥 IMPROVED SUGGESTIONS: Use representative phrases!
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
        if len(partial_input) < 1:
            return jsonify({"suggestions": []})
        
        if vectorizer is None or question_vectors is None:
            return jsonify({"suggestions": []})
        
        partial_vec = vectorizer.transform([partial_input])
        similarities = cosine_similarity(partial_vec, question_vectors).flatten()
        
        threshold = 0.03  # 🔥 LOWERED: More sensitive for partials
        top_indices = [(i, score) for i, score in enumerate(similarities) if score > threshold]
        top_indices.sort(key=lambda x: x[1], reverse=True)
        top_indices = top_indices[:10]
        
        suggestions = []
        seen_texts = set()
        for idx, score in top_indices:
            rep = kb_items[idx]["representative"]
            if rep not in seen_texts:
                seen_texts.add(rep)
                suggestions.append({
                    "text": rep,  # 🔥 IMPROVED: Full natural phrase!
                    "confidence": round(float(score), 2)
                })
        
        logger.info(f"💡 Suggestions: {len(suggestions)} for '{partial_input}'")
        return jsonify({"suggestions": suggestions})
        
    except Exception as e:
        logger.error(f"❌ Suggest error: {e}")
        return jsonify({"suggestions": []}), 500

@app.route("/reload", methods=["POST"])
def reload_knowledge():
    try:
        logger.info("🔄 RELOADING...")
        load_knowledge_base()
        
        global vectorizer, question_vectors
        vectorizer = None
        question_vectors = None
        
        success = build_vectorizer()
        
        if success:
            logger.info(f"✅ RELOAD SUCCESS: {len(kb_items)} questions")
            return jsonify({
                "status": "success",
                "message": "Reloaded successfully!",
                "questions_count": len(kb_items),
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
