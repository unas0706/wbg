from flask import Flask, request, jsonify
import re
import os
import numpy as np

from typing import Optional

# Optional S3 helper: if you prefer not to store model pickles in the repo,
# set these env vars in Render and the app will attempt to download missing
# model files from the S3 bucket on startup or first prediction request.
S3_BUCKET = os.environ.get('MODEL_S3_BUCKET')
S3_PREFIX = os.environ.get('MODEL_S3_PREFIX', '').rstrip('/')

def fetch_from_s3(bucket: str, key: str, dest_path: str) -> bool:
    """Attempt to download `key` from `bucket` into `dest_path` using boto3.
    Returns True on success, False otherwise.
    This function imports boto3 lazily so deployment won't fail if boto3 is not installed
    and S3 is not used.
    """
    try:
        import boto3
        os.makedirs(os.path.dirname(dest_path), exist_ok=True)
        s3 = boto3.client('s3')
        s3.download_file(bucket, key, dest_path)
        return True
    except Exception:
        return False

app = Flask(__name__)

# Attempt to locate models in common paths
ROOT = os.path.abspath(os.path.dirname(__file__))
MODEL_PATHS = [
    os.path.join(ROOT, 'models'),            # project root /models
    ROOT                                     # fallback to root
]

# Hold loaded objects (or None)
VECTORIZER = None
ESG_MODEL = None
SDG_MODEL = None

def try_load_models():
    global VECTORIZER, ESG_MODEL, SDG_MODEL
    for base in MODEL_PATHS:
        vfile = os.path.join(base, 'vectorizer.pkl')
        efile = os.path.join(base, 'esg_regression.pkl')
        sfile = os.path.join(base, 'sdg_regression.pkl')
        # If an S3 bucket is configured and a file is missing locally, try downloading it
        if S3_BUCKET:
            # Download vectorizer if missing
            if (not os.path.exists(vfile)) and S3_PREFIX:
                key = f"{S3_PREFIX}/vectorizer.pkl" if S3_PREFIX else 'vectorizer.pkl'
                fetch_from_s3(S3_BUCKET, key, vfile)
            if (not os.path.exists(efile)) and S3_PREFIX:
                key = f"{S3_PREFIX}/esg_regression.pkl" if S3_PREFIX else 'esg_regression.pkl'
                fetch_from_s3(S3_BUCKET, key, efile)
            if (not os.path.exists(sfile)) and S3_PREFIX:
                key = f"{S3_PREFIX}/sdg_regression.pkl" if S3_PREFIX else 'sdg_regression.pkl'
                fetch_from_s3(S3_BUCKET, key, sfile)

        try:
            # import heavy optional dependency lazily
            import joblib
            if VECTORIZER is None and os.path.exists(vfile):
                VECTORIZER = joblib.load(vfile)
                # Verify vectorizer is fitted
                if hasattr(VECTORIZER, 'idf_') and VECTORIZER.idf_ is None:
                    print(f"Warning: Vectorizer at {vfile} is not fitted")
                    VECTORIZER = None
            if ESG_MODEL is None and os.path.exists(efile):
                ESG_MODEL = joblib.load(efile)
            if SDG_MODEL is None and os.path.exists(sfile):
                SDG_MODEL = joblib.load(sfile)
        except Exception as e:
            # Log the error for debugging
            print(f"Error loading models from {base}: {str(e)}")
            import traceback
            traceback.print_exc()
            # don't crash on load/import errors; leave as None
            VECTORIZER = VECTORIZER or None
            ESG_MODEL = ESG_MODEL or None
            SDG_MODEL = SDG_MODEL or None

# Do not eagerly load heavy models at import time; load lazily on first predict call

def calculate_esg_scores(text):
    """Calculate ESG scores based on keyword presence and context"""
    text = text.lower()
    
    # Enhanced keywords with weights
    keywords = {
        'Environmental': {
            'high_impact': ['renewable energy', 'carbon reduction', 'climate action', 'environmental protection'],
            'medium_impact': ['solar', 'wind', 'water conservation', 'recycling', 'biodiversity'],
            'low_impact': ['green', 'sustainable', 'eco-friendly', 'natural resources']
        },
        'Social': {
            'high_impact': ['community development', 'poverty reduction', 'healthcare access', 'education equality'],
            'medium_impact': ['job creation', 'skill training', 'social welfare', 'gender equality'],
            'low_impact': ['community', 'training', 'social', 'welfare']
        },
        'Governance': {
            'high_impact': ['transparency initiative', 'anti-corruption', 'accountability framework'],
            'medium_impact': ['governance policy', 'compliance program', 'stakeholder engagement'],
            'low_impact': ['reporting', 'monitoring', 'policy', 'regulation']
        }
    }
    
    scores = {}
    details = {}
    
    for category, impact_levels in keywords.items():
        score = 0
        matched_terms = []
        
        # Check high impact terms (weight: 0.5)
        for term in impact_levels['high_impact']:
            if term in text:
                score += 0.5
                matched_terms.append(f"{term} (high impact)")
        
        # Check medium impact terms (weight: 0.3)
        for term in impact_levels['medium_impact']:
            if term in text:
                score += 0.3
                matched_terms.append(f"{term} (medium impact)")
        
        # Check low impact terms (weight: 0.2)
        for term in impact_levels['low_impact']:
            if term in text:
                score += 0.2
                matched_terms.append(f"{term} (low impact)")
        
        # Normalize score to 0-1 range and round to 2 decimal places
        scores[category] = round(min(1.0, score), 2)
        details[category] = matched_terms
    
    return scores, details


def predict_with_models(text):
    """If trained models are available, produce ESG and SDG predictions.
    Returns a tuple (esg_scores_dict, sdg_scores_dict) or (None, None) if models missing.
    """
    # Lazy-load models (may be heavy); do not do this at import time
    if VECTORIZER is None or ESG_MODEL is None:
        try_load_models()

    if VECTORIZER is None or ESG_MODEL is None:
        return None, None
    
    # Verify vectorizer is fitted before using
    if not hasattr(VECTORIZER, 'idf_') or VECTORIZER.idf_ is None:
        print("Warning: Vectorizer is not fitted")
        return None, None

    try:
        vec = VECTORIZER.transform([text])
    except Exception as e:
        print(f"Error transforming text with vectorizer: {str(e)}")
        return None, None

    try:
        esg_pred = ESG_MODEL.predict(vec)
        # ensure shape (n_outputs,)
        esg_arr = np.asarray(esg_pred).reshape(-1)
        esg_dict = {
            'Environment': float(round(esg_arr[0], 2)),
            'Social': float(round(esg_arr[1], 2)),
            'Governance': float(round(esg_arr[2], 2))
        }
    except Exception:
        esg_dict = None

    sdg_dict = None
    if SDG_MODEL is not None:
        try:
            sdg_pred = SDG_MODEL.predict(vec)
            sdg_arr = np.asarray(sdg_pred).reshape(-1)
            sdg_dict = {f'SDG{i+1}': float(round(v, 2)) for i, v in enumerate(sdg_arr)}
        except Exception:
            sdg_dict = None

    return esg_dict, sdg_dict

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        "status": "healthy",
        "message": "ESG Score Predictor API is running",
        "version": "2.0"
    }), 200


@app.route('/models/status', methods=['GET'])
def models_status():
    """Return which models are currently loaded and where artifacts exist."""
    # Ensure we attempted to load models
    try_load_models()
    roots = MODEL_PATHS
    status = {
        'vectorizer_loaded': bool(VECTORIZER),
        'esg_model_loaded': bool(ESG_MODEL),
        'sdg_model_loaded': bool(SDG_MODEL),
        'search_paths': roots,
        'found_files': {}
    }
    for base in roots:
        status['found_files'][base] = {
            'vectorizer': os.path.exists(os.path.join(base, 'vectorizer.pkl')),
            'esg_model': os.path.exists(os.path.join(base, 'esg_regression.pkl')),
            'sdg_model': os.path.exists(os.path.join(base, 'sdg_regression.pkl'))
        }
    return jsonify(status)


@app.route('/', methods=['GET'])
def home():
    """Home endpoint with API documentation"""
    return jsonify({
        "message": "Welcome to ESG Score Predictor API",
        "version": "2.0",
        "endpoints": {
            "/predict": {
                "method": "POST",
                "description": "Get ESG scores for a project description",
                "request_format": {
                    "description": "string (project description)"
                },
                "response_format": {
                    "input": {"description": "string"},
                    "scores": {
                        "Environmental": "float (0-1)",
                        "Social": "float (0-1)",
                        "Governance": "float (0-1)"
                    },
                    "model_scores": "(optional) model-based ESG predictions if models available",
                    "sdgs": "(optional) model-based SDG predictions if sdg model available",
                    "details": {
                        "Environmental": ["matched terms"],
                        "Social": ["matched terms"],
                        "Governance": ["matched terms"]
                    }
                },
                "example_request": {
                    "description": "Solar power installation with community training program and transparent governance"
                }
            },
            "/health": {
                "method": "GET",
                "description": "Check API health status"
            }
        }
    })

@app.route('/predict', methods=['POST'])
def predict():
    """Predict ESG scores for a given project description"""
    try:
        data = request.get_json(force=True)
        
        # Validate input
        if not data or 'description' not in data:
            return jsonify({
                "error": "Missing required field: description",
                "usage": {
                    "required_format": {
                        "description": "Your project description here"
                    }
                }
            }), 400
        
        description = data['description']
        
        # Calculate ESG scores and get details (keyword-based)
        scores, details = calculate_esg_scores(description)
        overall_score = round(sum(scores.values()) / 3, 2)

        # Try model-based predictions (if models exist)
        try:
            model_esg, model_sdgs = predict_with_models(description)
        except Exception as model_error:
            # Log model error but don't fail the request
            print(f"Model prediction error (using keyword-based scores only): {str(model_error)}")
            model_esg = None
            model_sdgs = None

        # Prepare response
        response = {
            "input": {"description": description},
            "scores": scores,
            "overall_score": overall_score,
            "details": details,
            "model_scores": model_esg,
            "sdgs": model_sdgs,
            "interpretation": {
                "scale": "Scores range from 0 to 1, where 1 indicates strongest alignment",
                "score_levels": {"high": "0.7 - 1.0", "medium": "0.4 - 0.69", "low": "0 - 0.39"}
            }
        }
        
        # Add note if model predictions are unavailable
        if model_esg is None:
            response["note"] = "Model-based predictions are currently unavailable. Showing keyword-based scores only."
        
        return jsonify(response)
    
    except Exception as e:
        # Log full error for debugging
        import traceback
        error_trace = traceback.format_exc()
        print(f"Error in predict endpoint: {error_trace}")
        return jsonify({
            "error": "Internal server error",
            "message": str(e)
        }), 500

if __name__ == '__main__':
    print("✅ Starting ESG Score Predictor API v2.0...")
    print("📡 API will be available at http://localhost:5000")
    print("📚 View documentation at http://localhost:5000")
    print("🔍 Try the health check at http://localhost:5000/health")
    app.run(host='0.0.0.0', port=5000, debug=True)

