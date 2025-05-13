from flask import Flask, request, jsonify
from flask_cors import CORS
from werkzeug.security import generate_password_hash, check_password_hash
import os
import psycopg2
from psycopg2 import pool
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

# PostgreSQL connection string (using Supabase)
DB_CREDENTIALS = os.getenv(
    "DATABASE_URL",
    "postgresql://postgres:CopGuide%40123@db.exmjgsixqxuvejkgquco.supabase.co:5432/postgres"
)

# PostgreSQL connection pool using connection string
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

# Initialize the users table if not exists
# Initialize the users table if not exists
def init_user_table():
    conn = connection_pool.getconn()
    try:
        with conn.cursor() as cursor:
            # Enable pgcrypto extension for UUID generation
            cursor.execute("CREATE EXTENSION IF NOT EXISTS pgcrypto;")
            
            # Create users table with UUID as primary key
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS users (
                    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                    user_name VARCHAR(50) NOT NULL,
                    email VARCHAR(100) UNIQUE NOT NULL,
                    password TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            conn.commit()
            logger.info("Users table with UUID initialized successfully.")
    except Exception as e:
        logger.error(f"Error initializing users table: {e}")
    finally:
        connection_pool.putconn(conn)

init_user_table()

# 📝 Signup endpoint
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

    # Hash the password for security
    hashed_password = generate_password_hash(password)

    conn = connection_pool.getconn()
    try:
        with conn.cursor() as cursor:
            # Check if the email is already in use
            cursor.execute("SELECT * FROM users WHERE email = %s", (email,))
            existing_user = cursor.fetchone()

            if existing_user:
                return jsonify({"error": "Email already exists"}), 400

            # Insert the new user into the database
            cursor.execute("""
                INSERT INTO users (user_name, email, password) 
                VALUES (%s, %s, %s) RETURNING id
            """, (user_name, email, hashed_password))
            user_id = cursor.fetchone()[0]
            conn.commit()

            return jsonify({"message": "User created", "user_id": str(user_id)}), 201
    except Exception as e:
        logger.error(f"Error during signup: {e}")
        return jsonify({"error": "Internal server error"}), 500
    finally:
        connection_pool.putconn(conn)


# 🔑 Login endpoint
@app.route('/login', methods=['POST'])
def login():
    data = request.json
    email = data.get('email')
    password = data.get('password')

    # Check for required fields
    if not email or not password:
        return jsonify({"error": "Email and password are required"}), 400

    conn = connection_pool.getconn()
    try:
        with conn.cursor() as cursor:
            # Find the user in the PostgreSQL database
            cursor.execute("SELECT * FROM users WHERE email = %s", (email,))
            user = cursor.fetchone()

            if not user:
                return jsonify({"error": "Invalid email or password"}), 400

            # Check if the password matches the stored hashed password
            if check_password_hash(user[3], password):
                return jsonify({
                    "message": "Login successful",
                    "user_id": str(user[0]),  # UUID as user_id
                    "username": user[1],
                    "email": user[2]
                }), 200
            else:
                return jsonify({"error": "Invalid email or password"}), 400
    except Exception as e:
        logger.error(f"Error during login: {e}")
        return jsonify({"error": "Internal server error"}), 500
    finally:
        connection_pool.putconn(conn)


# Health check endpoint
@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'healthy'}), 200

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
