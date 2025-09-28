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

# Enable CORS for all routes - this fixes the browser fetch issues
CORS(app, origins=["*"])

# Folder where JSON knowledge files are stored
knowledge_folder = "knowledge"
questions_list = []
answers_list = []

def load_knowledge_base():
    """Load knowledge base from JSON files"""
    global questions_list, answers_list
    
    logger.info(f"Loading knowledge base from {knowledge_folder}")
    
    # Check if knowledge folder exists
    if not os.path.exists(knowledge_folder):
        logger.error(f"Knowledge folder '{knowledge_folder}' not found!")
        # Create sample data if folder doesn't exist
        create_sample_knowledge()
        return
    
    # Reset lists
    questions_list = []
    answers_list = []
    
    # Loop through all JSON files in the folder
    json_files_found = 0
    for file in os.listdir(knowledge_folder):
        if file.endswith(".json"):
            json_files_found += 1
            path = os.path.join(knowledge_folder, file)
            try:
                with open(path, "r", encoding="utf-8") as f:
                    kb = json.load(f)
                
                logger.info(f"Loaded {file}")
                
                # Flatten all questions
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
        logger.warning("No JSON files found in knowledge folder")
        create_sample_knowledge()
    
    logger.info(f"Loaded {len(questions_list)} questions from {json_files_found} files")

def create_sample_knowledge():
    """Create sample knowledge base if none exists"""
    logger.info("Creating sample knowledge base")
    
    global questions_list, answers_list
    
    # Sample NCST FAQ data
    sample_data = [
        {
            "patterns": ["admission", "apply", "application", "entry", "how to apply"],
            "answer": "For admission to NCST, you need to meet the eligibility criteria for your desired program. Please visit our admissions office or check the official website for detailed application procedures, required documents, and deadlines."
        },
        {
            "patterns": ["fees", "fee structure", "cost", "tuition", "payment"],
            "answer": "Fee structures vary by program. Please contact the accounts office or visit the official website for current fee information. Payment can typically be made online or at the college accounts office."
        },
        {
            "patterns": ["courses", "programs", "degrees", "subjects", "curriculum"],
            "answer": "NCST offers various engineering programs, diploma courses, and degree programs. For detailed information about specific courses, curriculum, and duration, please contact the academic office or visit our website."
        },
        {
            "patterns": ["contact", "phone", "email", "address", "location"],
            "answer": "You can contact NCST through our main office during business hours. Please visit our official website for complete contact information including phone numbers, email addresses, and campus location."
        },
        {
            "patterns": ["placement", "jobs", "recruitment", "career", "companies"],
            "answer": "NCST has an active placement cell that organizes campus recruitment drives with various companies. For current placement statistics and career guidance, please contact the placement office."
        },
        {
            "patterns": ["hostel", "accommodation", "boarding", "residence"],
            "answer": "NCST provides hostel facilities for both male and female students with basic amenities. For availability and booking, please contact the hostel administration or warden."
        },
        {
            "patterns": ["hello", "hi", "hey", "greetings"],
            "answer": "Hello! I'm the NCST FAQ assistant. I can help you with information about admissions, fees, courses, placements, and other college-related queries. How can I help you today?"
        }
    ]
    
    questions_list = [" ".join(item["patterns"]) for item in sample_data]
    answers_list = [item["answer"] for item in sample_data]

# Load knowledge base on startup
load_knowledge_base()

# Build TF-IDF vectorizer only if we have data
vectorizer = None
question_vectors = None

def build_vectorizer():
    """Build or rebuild the TF-IDF vectorizer"""
    global vectorizer, question_vectors
    
    if not questions_list:
        logger.error("No questions available to build vectorizer")
        return False
    
    try:
        vectorizer = TfidfVectorizer(stop_words='english', lowercase=True, max_features=1000)
        question_vectors = vectorizer.fit_transform(questions_list)
        logger.info("TF-IDF vectorizer built successfully")
        return True
    except Exception as e:
        logger.error(f"Error building vectorizer: {e}")
        return False

# Build vectorizer on startup
build_vectorizer()

@app.route("/", methods=["GET"])
def home():
    """Health check endpoint"""
    return jsonify({
        "status": "online",
        "message": "NCST FAQ Chatbot API is running",
        "questions_loaded": len(questions_list),
        "version": "1.0.0"
    })

@app.route("/health", methods=["GET"])
def health():
    """Detailed health check"""
    return jsonify({
        "status": "healthy",
        "questions_count": len(questions_list),
        "answers_count": len(answers_list),
        "vectorizer_ready": vectorizer is not None
    })

@app.route("/chat", methods=["POST", "OPTIONS"])
def chat():
    """Main chat endpoint"""
    
    # Handle preflight OPTIONS request for CORS
    if request.method == "OPTIONS":
        response = jsonify({"status": "ok"})
        response.headers.add("Access-Control-Allow-Origin", "*")
        response.headers.add("Access-Control-Allow-Headers", "Content-Type")
        response.headers.add("Access-Control-Allow-Methods", "POST, OPTIONS")
        return response
    
    try:
        # Get user input
        data = request.get_json()
        if not data:
            return jsonify({"error": "No JSON data received"}), 400
            
        user_input = data.get("message", "").strip()
        if not user_input:
            return jsonify({"answer": "Please ask a question."})
        
        logger.info(f"Received question: {user_input}")
        
        # Check if vectorizer is ready
        if vectorizer is None or question_vectors is None:
            logger.error("Vectorizer not initialized")
            return jsonify({"answer": "Sorry, the system is not ready yet. Please try again in a moment."})
        
        # Transform user input
        user_vec = vectorizer.transform([user_input])
        
        # Calculate similarities
        similarities = cosine_similarity(user_vec, question_vectors).flatten()
        best_idx = similarities.argmax()
        best_score = similarities[best_idx]
        
        logger.info(f"Best match score: {best_score}")
        
        # Set threshold for minimum similarity
        threshold = 0.2
        
        if best_score < threshold:
            response = "Sorry, I don't have information about that specific topic. Could you please rephrase your question or ask about admissions, fees, courses, placements, hostel facilities, or contact information?"
        else:
            response = answers_list[best_idx]
        
        logger.info(f"Sending response: {response[:100]}...")
        
        return jsonify({"answer": response})
        
    except Exception as e:
        logger.error(f"Error in chat endpoint: {str(e)}")
        return jsonify({
            "error": "Internal server error",
            "answer": "Sorry, I encountered an error while processing your request. Please try again."
        }), 500

@app.route("/reload", methods=["POST"])
def reload_knowledge():
    """Reload knowledge base (useful for updates)"""
    try:
        load_knowledge_base()
        success = build_vectorizer()
        
        if success:
            return jsonify({
                "status": "success",
                "message": "Knowledge base reloaded successfully",
                "questions_count": len(questions_list)
            })
        else:
            return jsonify({
                "status": "error",
                "message": "Failed to rebuild vectorizer"
            }), 500
            
    except Exception as e:
        logger.error(f"Error reloading knowledge base: {e}")
        return jsonify({
            "status": "error",
            "message": f"Failed to reload: {str(e)}"
        }), 500

# Error handlers
@app.errorhandler(404)
def not_found(error):
    return jsonify({"error": "Endpoint not found"}), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({"error": "Internal server error"}), 500

if __name__ == "__main__":
    # For development
    app.run(debug=True, host="0.0.0.0", port=5000)
else:
    # For production (like on Render)
    logger.info("Starting in production mode")
    
    # Additional setup for production if needed
    if not questions_list:
        logger.warning("No questions loaded, using sample data")
        create_sample_knowledge()
        build_vectorizer()
