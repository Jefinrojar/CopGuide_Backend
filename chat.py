from flask import Flask, request, jsonify
import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
import google.generativeai as genai

app = Flask(__name__)

# Load environment variables (assuming GOOGLE_API_KEY is set in environment)
GOOGLE_API_KEY = "AIzaSyCCTW2ZY8D9hrYrAqrDeVCZzcSrsjYanN8"
if not GOOGLE_API_KEY:
    raise ValueError("GOOGLE_API_KEY not found in environment variables.")

genai.configure(api_key=GOOGLE_API_KEY)

# Load the vector store from local directory
VECTOR_STORE_DIR = './vector_store/'  # Update path if needed
try:
    if not os.path.exists(VECTOR_STORE_DIR):
        raise FileNotFoundError(f"Vector store directory not found at {VECTOR_STORE_DIR}")
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=GOOGLE_API_KEY)
    vector_store = FAISS.load_local(VECTOR_STORE_DIR, embeddings, allow_dangerous_deserialization=True)
    print("Vector store loaded successfully.")
except Exception as e:
    print(f"Error loading vector store: {e}")
    raise

# Function to create conversational chain
def get_conversational_chain():
    prompt_template = """
    Answer the question as precisely as possible based on the provided context. If the answer is not in the context, say "I couldn't find that information in the provided documents." Provide concise and accurate responses.

    Context: {context}

    Question: {question}

    Answer:
    """
    try:
        model = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.3, google_api_key=GOOGLE_API_KEY)
        prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
        chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
        return chain
    except Exception as e:
        print(f"Error creating conversational chain: {e}")
        return None

# Function to process user input and get response
def process_query(user_question):
    try:
        docs = vector_store.similarity_search(user_question, k=4)  # Retrieve top 4 relevant chunks
        chain = get_conversational_chain()
        if not chain:
            return "Failed to create conversational chain."
        response = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)
        return response["output_text"]
    except Exception as e:
        print(f"Error processing question: {e}")
        return "Error processing question."

# API endpoint for searching
@app.route('/search', methods=['POST'])
def search():
    try:
        data = request.get_json()
        if not data or 'question' not in data:
            return jsonify({'error': 'Missing question in request body'}), 400
        
        question = data['question']
        if not question:
            return jsonify({'error': 'Question cannot be empty'}), 400

        response = process_query(question)
        return jsonify({'answer': response}), 200
    except Exception as e:
        print(f"Error in search endpoint: {e}")
        return jsonify({'error': 'Internal server error'}), 500

# Health check endpoint
@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'healthy'}), 200

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)