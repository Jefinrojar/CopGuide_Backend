from app import app
from flask_cors import CORS

# Allow CORS for all domains
CORS(app)


if __name__ == '__main__':
    app.run(debug=True)
