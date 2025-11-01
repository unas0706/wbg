from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import uvicorn
from typing import Dict, Optional

app = FastAPI(
    title="ESG Score Predictor API",
    description="API for predicting Environmental, Social, and Governance (ESG) scores for project descriptions",
    version="1.0.0"
)

class ProjectRequest(BaseModel):
    description: str
    
class ProjectResponse(BaseModel):
    description: str
    scores: Dict[str, float]

@app.post("/predict", response_model=ProjectResponse)
async def predict_esg(request: ProjectRequest):
    try:
        # Load the saved model and vectorizer
        vectorizer = joblib.load('vectorizer.pkl')
        model = joblib.load('esg_regression.pkl')
        
        # Make prediction
        pred = model.predict(vectorizer.transform([request.description]))[0]
        
        # Format response
        scores = {
            'Environment': round(pred[0], 2),
            'Social': round(pred[1], 2),
            'Governance': round(pred[2], 2)
        }
        
        return ProjectResponse(
            description=request.description,
            scores=scores
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
async def root():
    return {
        "message": "ESG Score Predictor API",
        "usage": {
            "endpoint": "/predict",
            "method": "POST",
            "request_body": {
                "description": "Your project description here"
            }
        }
    }

if __name__ == "__main__":
    import os
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)