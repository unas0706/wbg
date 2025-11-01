from flask import Flask, request, jsonify
import joblib
import os
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.multioutput import MultiOutputRegressor
from sklearn.linear_model import LinearRegression

app = Flask(__name__)

# Initialize models
model = None
vectorizer = None

def initialize_models():
    """Create and train models if they don't exist"""
    global model, vectorizer
    
    # Sample data
    data = {
        'Description': [
            'Solar power plant installation with community training program',
            'Sustainable forestry management and biodiversity conservation project',
            'Rural education and healthcare infrastructure development initiative',
            'Urban water treatment and waste management system implementation',
            'Women entrepreneurship and microfinance program development'
        ],
        'E': [0.8, 0.6, 0.2, 0.4, 0.1],
        'S': [0.4, 0.3, 0.9, 0.5, 0.8],
        'G': [0.3, 0.4, 0.5, 0.7, 0.6]
    }
    df = pd.DataFrame(data)
    
    # Train models
    vectorizer = TfidfVectorizer(max_features=4000)
    X = vectorizer.fit_transform(df['Description'])
    y = df[['E', 'S', 'G']]
    model = MultiOutputRegressor(LinearRegression()).fit(X, y)
    return "Models initialized successfully"

@app.route('/')
def home():
    return jsonify({
        "message": "Welcome to ESG Score Predictor API",
        "usage": {
            "endpoint": "/predict",
            "method": "POST",
            "body": {"description": "Your project description"},
            "example": {
                "curl": "curl -X POST http://localhost:5000/predict -H 'Content-Type: application/json' -d '{\"description\":\"solar power project\"}'"
            }
        }
    })

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Initialize models if not already done
        if model is None or vectorizer is None:
            initialize_models()
        
        # Get input
        data = request.get_json()
        if not data or 'description' not in data:
            return jsonify({"error": "Missing 'description' field"}), 400
        
        # Make prediction
        text = data['description']
        pred = model.predict(vectorizer.transform([text]))[0]
        
        return jsonify({
            "description": text,
            "scores": {
                "Environmental": float(round(pred[0], 2)),
                "Social": float(round(pred[1], 2)),
                "Governance": float(round(pred[2], 2))
            }
        })
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    # Initialize models at startup
    initialize_models()
    print("✅ Models initialized")
    # Run the app
    app.run(host='0.0.0.0', port=5000, debug=True)