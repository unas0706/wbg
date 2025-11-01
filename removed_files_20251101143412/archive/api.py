from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict
import joblib
import uvicorn
import os
import json

app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ProjectRequest(BaseModel):
    description: str

class ProjectResponse(BaseModel):
    description: str
    scores: Dict[str, float]

@app.post("/predict")
async def predict_esg(request: Request):
    try:
        # Debug: Print raw request body
        raw_body = await request.body()
        body_str = raw_body.decode()
        print(f"Raw request body: {body_str}")
        
        # Parse JSON manually first
        try:
            data = json.loads(body_str)
            print(f"Parsed JSON: {data}")
        except json.JSONDecodeError as e:
            raise HTTPException(status_code=400, detail=f"Invalid JSON: {str(e)}")
        
        # Validate request structure
        if not isinstance(data, dict):
            raise HTTPException(status_code=400, detail="Request body must be a JSON object")
        
        if "description" not in data:
            raise HTTPException(status_code=400, detail="Missing 'description' field")
            
        description = data["description"]
        if not isinstance(description, str):
            raise HTTPException(status_code=400, detail="'description' must be a string")

        # Get the current directory and model path
        current_dir = os.path.dirname(os.path.abspath(__file__))
        model_path = os.path.join(current_dir, 'models')
        
        # Load models
        vectorizer = joblib.load(os.path.join(model_path, 'vectorizer.pkl'))
        model = joblib.load(os.path.join(model_path, 'esg_regression.pkl'))
        
        # Make prediction
        pred = model.predict(vectorizer.transform([request.description]))[0]
        
        # Format response
        scores = {
            'Environment': float(round(pred[0], 2)),
            'Social': float(round(pred[1], 2)),
            'Governance': float(round(pred[2], 2))
        }
        
        return ProjectResponse(
            description=request.description,
            scores=scores
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
def read_root():
    return {
        "message": "Welcome to ESG Score Predictor API",
        "endpoints": {
            "/predict": {
                "method": "POST",
                "description": "Predict ESG scores for a project description",
                "request_format": {
                    "description": "string"
                },
                "example": {
                    "description": "This project installs solar microgrids for rural villages and trains women workers."
                }
            }
        }
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)