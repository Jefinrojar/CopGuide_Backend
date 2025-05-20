from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import psycopg2
from psycopg2 import pool
import numpy as np
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
import google.generativeai as genai
import logging
from auth import register_auth_routes  # Import authentication routes

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

# Load environment variables
GOOGLE_API_KEY = "AIzaSyCCTW2ZY8D9hrYrAqrDeVCZzcSrsjYanN8"
if not GOOGLE_API_KEY:
    raise ValueError("GOOGLE_API_KEY not found in environment variables.")

genai.configure(api_key=GOOGLE_API_KEY)

# Database connection string
DB_CREDENTIALS = os.getenv(
    "DATABASE_URL",
    "postgresql://copguidedb_user:T3taIXpN797Gp3JQKoblnN0LlpYjeeDa@dpg-d0m9nnfdiees73dprrv0-a.oregon-postgres.render.com/copguidedb"
)

# PostgreSQL connection pool
try:
    connection_pool = psycopg2.pool.SimpleConnectionPool(
        minconn=1,
        maxconn=20,
        dsn=DB_CREDENTIALS
    )
    logger.info("PostgreSQL connection pool initialized successfully.")
except Exception as e:
    logger.error(f"Error connecting to PostgreSQL: {e}")
    raise

# Initialize feedback and Q-table (no user table here)
def init_db():
    conn = connection_pool.getconn()
    try:
        with conn.cursor() as cursor:
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS feedback (
                    id SERIAL PRIMARY KEY,
                    question TEXT NOT NULL,
                    response TEXT NOT NULL,
                    feedback INTEGER DEFAULT 0,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS q_table (
                    question_hash TEXT NOT NULL,
                    doc_index INTEGER NOT NULL,
                    q_value REAL DEFAULT 0.0,
                    PRIMARY KEY (question_hash, doc_index)
                )
            """)
            conn.commit()
            logger.info("Database tables initialized successfully.")
    except Exception as e:
        logger.error(f"Error initializing database: {e}")
        raise
    finally:
        connection_pool.putconn(conn)

init_db()

# Register authentication routes
register_auth_routes(app, connection_pool)

# Load the vector store
VECTOR_STORE_DIR = './vector_store/'
try:
    if not os.path.exists(VECTOR_STORE_DIR):
        raise FileNotFoundError(f"Vector store directory not found at {VECTOR_STORE_DIR}")
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=GOOGLE_API_KEY)
    vector_store = FAISS.load_local(VECTOR_STORE_DIR, embeddings, allow_dangerous_deserialization=True)
    logger.info("Vector store loaded successfully.")
except Exception as e:
    logger.error(f"Error loading vector store: {e}")
    raise

# RL parameters
ALPHA = 0.2
GAMMA = 0.9
EPSILON = 0.05

# Function to hash questions for Q-table
def hash_question(question):
    import hashlib
    return hashlib.md5(question.encode()).hexdigest()

# Function to get or initialize Q-value
def get_q_value(question, doc_index):
    question_hash = hash_question(question)
    conn = connection_pool.getconn()
    try:
        with conn.cursor() as cursor:
            cursor.execute(
                "SELECT q_value FROM q_table WHERE question_hash = %s AND doc_index = %s",
                (question_hash, doc_index)
            )
            result = cursor.fetchone()
            if result:
                logger.info(f"Retrieved Q-value: question_hash={question_hash}, doc_index={doc_index}, q_value={result[0]}")
                return result[0]
            else:
                cursor.execute(
                    "INSERT INTO q_table (question_hash, doc_index, q_value) VALUES (%s, %s, %s) ON CONFLICT DO NOTHING",
                    (question_hash, doc_index, 0.0)
                )
                conn.commit()
                logger.info(f"Initialized Q-value: question_hash={question_hash}, doc_index={doc_index}, q_value=0.0")
                return 0.0
    except Exception as e:
        logger.error(f"Error getting Q-value: {e}")
        return 0.0
    finally:
        connection_pool.putconn(conn)

# Function to update Q-value based on feedback
def update_q_value(question, doc_index, reward):
    question_hash = hash_question(question)
    current_q = get_q_value(question, doc_index)
    new_q = current_q + ALPHA * (reward - current_q)
    conn = connection_pool.getconn()
    try:
        with conn.cursor() as cursor:
            cursor.execute(
                "UPDATE q_table SET q_value = %s WHERE question_hash = %s AND doc_index = %s",
                (new_q, question_hash, doc_index)
            )
            if cursor.rowcount == 0:
                logger.warning(f"No rows updated in q_table for question_hash={question_hash}, doc_index={doc_index}")
            conn.commit()
            logger.info(f"Updated Q-value: question_hash={question_hash}, doc_index={doc_index}, new_q={new_q}")
    except Exception as e:
        logger.error(f"Error updating Q-value for question_hash={question_hash}, doc_index={doc_index}: {e}")
        raise
    finally:
        connection_pool.putconn(conn)

# Function to create conversational chain
def get_conversational_chain():
    prompt_template = """
    You are a highly knowledgeable and precise assistant specializing in legal and governmental queries. Your goal is to provide accurate, concise, and professional answers based solely on the provided context. Follow these steps:
    1. Analyze the context to identify relevant information. Do not reference or assume any information, sections, or laws not explicitly mentioned in the context.
    2. Formulate a clear and direct answer, citing specific details (e.g., rule numbers, dates, or sections) from the context when available.
    3. If the answer is not in the context, clearly state that the information is unavailable, summarize any related information from the context, and suggest consulting relevant authorities or legal references without assuming specific laws or sections.

    **Examples**:
    - Question: What is the capital of France?
      Context: France is a country in Europe. Its capital is Paris.
      Answer: The capital of France is Paris.

    - Question: What is the penalty for theft?
      Context: Government servants must not engage in unauthorized fund collections.
      Answer: The provided context does not address the penalty for theft, as it focuses on rules for government servants regarding fund collections. For information on theft penalties, consult the relevant criminal law of the jurisdiction.

    Context: {context}

    Question: {question}

    Answer:
    """
    try:
        model = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.1, google_api_key=GOOGLE_API_KEY)
        prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
        chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
        return chain
    except Exception as e:
        logger.error(f"Error creating conversational chain: {e}")
        return None

# Function to process user input with RL
def process_query(user_question):
    try:
        docs = vector_store.similarity_search(user_question, k=2)
        logger.info(f"Retrieved documents for '{user_question}': {[doc.page_content[:100] for doc in docs]}")
        if np.random.rand() < EPSILON:
            selected_doc_index = int(np.random.randint(len(docs)))
            logger.info(f"Exploration: Selected doc_index={selected_doc_index}")
        else:
            q_values = [get_q_value(user_question, i) for i in range(len(docs))]
            selected_doc_index = int(np.argmax(q_values))
            logger.info(f"Exploitation: Selected doc_index={selected_doc_index}, q_values={q_values}")
        
        selected_docs = [docs[selected_doc_index]]
        chain = get_conversational_chain()
        if not chain:
            return "Failed to create conversational chain.", None, None
        
        response = chain({"input_documents": selected_docs, "question": user_question}, return_only_outputs=True)
        response_text = response["output_text"]
        
        conn = connection_pool.getconn()
        try:
            with conn.cursor() as cursor:
                cursor.execute(
                    "INSERT INTO feedback (question, response, feedback) VALUES (%s, %s, %s) RETURNING id",
                    (user_question, response_text, 0)
                )
                feedback_id = cursor.fetchone()[0]
                conn.commit()
        except Exception as e:
            logger.error(f"Error storing feedback: {e}")
            return "Error processing question.", None, None
        finally:
            connection_pool.putconn(conn)
        
        return response_text, int(feedback_id), int(selected_doc_index)
    except Exception as e:
        logger.error(f"Error processing question: {e}")
        return "Error processing question.", None, None

# API endpoint for searching
@app.route('/search', methods=['POST'])
def search():
    try:
        data = request.get_json()
        if not data or 'question' not in data:
            return jsonify({'error': 'Missing question in request body'}), 400
        
        question = data['question'].strip()
        if not question:
            return jsonify({'error': 'Question cannot be empty'}), 400

        response, feedback_id, doc_index = process_query(question)
        return jsonify({
            'answer': response,
            'feedback_id': int(feedback_id) if feedback_id is not None else None,
            'doc_index': int(doc_index) if doc_index is not None else None
        }), 200
    except Exception as e:
        logger.error(f"Error in search endpoint: {e}")
        return jsonify({'error': 'Internal server error'}), 500

# API endpoint for feedback
@app.route('/feedback', methods=['POST'])
def feedback():
    try:
        data = request.get_json()
        if not data or 'feedback_id' not in data or 'feedback' not in data or 'doc_index' not in data:
            return jsonify({'error': 'Missing feedback_id, feedback, or doc_index in request body'}), 400
        
        feedback_id = data['feedback_id']
        feedback_score = data['feedback']
        doc_index = data['doc_index']
        
        conn = connection_pool.getconn()
        try:
            with conn.cursor() as cursor:
                cursor.execute(
                    "UPDATE feedback SET feedback = %s WHERE id = %s",
                    (feedback_score, feedback_id)
                )
                if cursor.rowcount == 0:
                    logger.warning(f"No feedback record found for id={feedback_id}")
                    return jsonify({'error': 'Invalid feedback_id'}), 400
                cursor.execute(
                    "SELECT question FROM feedback WHERE id = %s",
                    (feedback_id,)
                )
                question = cursor.fetchone()
                if not question:
                    return jsonify({'error': 'Invalid feedback_id'}), 400
                question = question[0]
                conn.commit()
        except Exception as e:
            logger.error(f"Error updating feedback: {e}")
            return jsonify({'error': 'Internal server error'}), 500
        finally:
            connection_pool.putconn(conn)
        
        update_q_value(question, doc_index, feedback_score)
        return jsonify({'status': 'Feedback recorded'}), 200
    except Exception as e:
        logger.error(f"Error in feedback endpoint: {e}")
        return jsonify({'error': 'Internal server error'}), 500

# API endpoint for retrieving Q-values
@app.route('/qvalues', methods=['GET'])
def get_qvalues():
    question = request.args.get('question', '').strip()
    if not question:
        return jsonify({'error': 'Missing or empty question parameter'}), 400
    
    logger.info(f"Fetching Q-values for question: {question}")
    question_hash = hash_question(question)
    conn = connection_pool.getconn()
    try:
        with conn.cursor() as cursor:
            cursor.execute(
                "SELECT doc_index, q_value FROM q_table WHERE question_hash = %s",
                (question_hash,)
            )
            q_values = cursor.fetchall()
            response = {
                'question': question,
                'q_values': sorted(
                    [{'doc_index': int(row[0]), 'q_value': float(row[1])} for row in q_values],
                    key=lambda x: x['q_value'],
                    reverse=True
                )
            }
            if not q_values:
                response['message'] = 'No Q-values found for this question. Try processing the question with /search and providing feedback.'
            else:
                response['message'] = 'Q-values retrieved successfully.'
            return jsonify(response), 200
    except Exception as e:
        logger.error(f"Error fetching Q-values: {e}")
        return jsonify({'error': 'Internal server error'}), 500
    finally:
        connection_pool.putconn(conn)

# Health check endpoint
@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'healthy'}), 200

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)