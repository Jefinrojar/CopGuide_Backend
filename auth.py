from flask import request, jsonify
from werkzeug.security import generate_password_hash, check_password_hash
import psycopg2
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize the users table
def init_user_table(connection_pool):
    conn = connection_pool.getconn()
    try:
        with conn.cursor() as cursor:
            cursor.execute("CREATE EXTENSION IF NOT EXISTS pgcrypto;")
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

# Signup endpoint
def create_signup(app, connection_pool):
    @app.route('/signup', methods=['POST'])
    def signup():
        data = request.json
        user_name = data.get('user_name')
        email = data.get('email')
        password = data.get('password')
        confirm_password = data.get('confirm_password')

        if not all([user_name, email, password, confirm_password]):
            return jsonify({"error": "All fields are required"}), 400

        if password != confirm_password:
            return jsonify({"error": "Passwords do not match"}), 400

        hashed_password = generate_password_hash(password)
        conn = connection_pool.getconn()
        try:
            with conn.cursor() as cursor:
                cursor.execute("SELECT * FROM users WHERE email = %s", (email,))
                existing_user = cursor.fetchone()

                if existing_user:
                    return jsonify({"error": "Email already exists"}), 400

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
    return signup

# Login endpoint
def create_login(app, connection_pool):
    @app.route('/login', methods=['POST'])
    def login():
        data = request.json
        email = data.get('email')
        password = data.get('password')

        if not email or not password:
            return jsonify({"error": "Email and password are required"}), 400

        conn = connection_pool.getconn()
        try:
            with conn.cursor() as cursor:
                cursor.execute("SELECT * FROM users WHERE email = %s", (email,))
                user = cursor.fetchone()

                if not user:
                    return jsonify({"error": "Invalid email or password"}), 400

                if check_password_hash(user[3], password):
                    return jsonify({
                        "message": "Login successful",
                        "user_id": str(user[0]),
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
    return login

# Function to register routes with the Flask app
def register_auth_routes(app, connection_pool):
    init_user_table(connection_pool)  # Initialize user table
    create_signup(app, connection_pool)  # Register signup route
    create_login(app, connection_pool)   # Register login route