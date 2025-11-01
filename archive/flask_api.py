from flask import Flask, request, jsonify
import joblib
import os
from flask_cors import CORS

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Get the current directory
current_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(current_dir, 'models')

# Load models at startup
try:
    vectorizer = joblib.load(os.path.join(model_path, 'vectorizer.pkl'))
    model = joblib.load(os.path.join(model_path, 'esg_regression.pkl'))
except Exception as e:
    print(f"Error loading models: {str(e)}")
    print("Make sure to run train_model.py first to generate the models.")

@app.route('/', methods=['GET'])
def home():
    return jsonify({
        "message": "Welcome to ESG Score Predictor API",
        "usage": {
            "endpoint": "/predict",
            "method": "POST",
            "body": {
                "description": "Your project description here"
            },
            "example_curl": """
                curl -X POST http://localhost:5000/predict 
                     -H "Content-Type: application/json" 
                     -d '{"description": "This project installs solar microgrids"}'
            """
        }
    })

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get JSON data from request
        data = request.get_json()
        
        if not data or 'description' not in data:
            return jsonify({"error": "Missing 'description' field"}), 400
            
        description = data['description']
        
        # Make prediction
        pred = model.predict(vectorizer.transform([description]))[0]
        
        # Format response
        response = {
            "description": description,
            "scores": {
                "Environment": float(round(pred[0], 2)),
                "Social": float(round(pred[1], 2)),
                "Governance": float(round(pred[2], 2))
            }
        }
        
        return jsonify(response)
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)