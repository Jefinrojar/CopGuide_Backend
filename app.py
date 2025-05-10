from flask import Flask, request, jsonify
from flask_cors import CORS
from sklearn.preprocessing import normalize
from pymongo import MongoClient
from werkzeug.security import generate_password_hash, check_password_hash

# Initialize Flask app
app = Flask(__name__)
CORS(app)



# MongoDB configuration for user authentication
MONGO_URI = 'mongodb+srv://chatbot:chatbot123@cluster0.i1lqs.mongodb.net/'  # Update with your MongoDB URI
DATABASE_NAME = 'chatbot'  # Name of your database
USERS_COLLECTION_NAME = 'users'  # Name of the users collection

# Initialize MongoDB client and access collection for user authentication
client = MongoClient(MONGO_URI)
db = client[DATABASE_NAME]
users_collection = db[USERS_COLLECTION_NAME]


# Sign Up endpoint
@app.route('/signup', methods=['POST'])
def signup():
    data = request.json
    user_name = data.get('user_name')
    email = data.get('email')
    password = data.get('password')
    confirm_password = data.get('confirm_password')

    # Check for required fields
    if not all([user_name, email, password, confirm_password]):
        return jsonify({"error": "All fields are required"}), 400

    # Check if the passwords match
    if password != confirm_password:
        return jsonify({"error": "Passwords do not match"}), 400

    # Check if the email is already in use
    if users_collection.find_one({"email": email}):
        return jsonify({"error": "Email already exists"}), 400

    # Hash the password for security
    hashed_password = generate_password_hash(password)

    # Insert the user into the MongoDB collection
    user_id = users_collection.insert_one({
        "user_name": user_name,
        "email": email,
        "password": hashed_password
    }).inserted_id

    return jsonify({"message": "User created", "user_id": str(user_id)}), 201

# Login
@app.route('/login', methods=['POST'])
def login():
    data = request.json
    email = data.get('email')  # Use .get() to avoid KeyError if key is missing
    password = data.get('password')

    # Check for required fields
    if not email or not password:
        return jsonify({"error": "Email and password are required"}), 400

    # Find the user in the MongoDB collection
    user = users_collection.find_one({"email": email})

    if not user:
        return jsonify({"error": "Invalid email or password"}), 400

    # Check if the password matches the stored hashed password
    if check_password_hash(user['password'], password):
        return jsonify({
            "message": "Login successful",
            "user_id": str(user['_id']),
            "username": user['user_name'],  # Add username to the response
            "email": user['email']
        }), 200
    else:
        return jsonify({"error": "Invalid email or password"}), 400


# Run the Flask app
if __name__ == "__main__":
    app.run(debug=True)