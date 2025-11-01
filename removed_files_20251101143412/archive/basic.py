from flask import Flask, request, jsonify

app = Flask(__name__)

def score_esg(text):
    """Simple keyword-based scoring"""
    text = text.lower()
    
    # Keywords for each category
    E_keywords = ['renewable', 'solar', 'wind', 'climate', 'water', 'carbon', 'forest', 'pollution', 'sustainability']
    S_keywords = ['education', 'health', 'community', 'women', 'youth', 'poverty', 'training', 'employment', 'social']
    G_keywords = ['governance', 'transparency', 'policy', 'regulation', 'anti-corruption', 'institution', 'audit']
    
    def calc_score(keywords):
        return round(min(1.0, sum(word in text for word in keywords) / 5.0), 2)
    
    return {
        "Environmental": calc_score(E_keywords),
        "Social": calc_score(S_keywords),
        "Governance": calc_score(G_keywords)
    }

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
        data = request.get_json()
        if not data or 'description' not in data:
            return jsonify({"error": "Missing 'description' field"}), 400
        
        text = data['description']
        scores = score_esg(text)
        
        return jsonify({
            "description": text,
            "scores": scores,
            "interpretation": {
                "scale": "Scores range from 0 to 1, where 1 indicates strongest alignment",
                "categories": {
                    "Environmental": "Environmental impact and sustainability",
                    "Social": "Social development and community impact",
                    "Governance": "Institutional strength and accountability"
                }
            }
        })
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    print("✅ Starting ESG Score Predictor API")
    app.run(host='0.0.0.0', port=5000)