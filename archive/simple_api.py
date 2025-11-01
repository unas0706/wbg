from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import os

class ProjectInput(BaseModel):
    description: str

app = FastAPI()

@app.post("/predict")
def predict(input: ProjectInput):
    try:
        # Load models
        current_dir = os.path.dirname(os.path.abspath(__file__))
        model_path = os.path.join(current_dir, 'models')
        vectorizer = joblib.load(os.path.join(model_path, 'vectorizer.pkl'))
        model = joblib.load(os.path.join(model_path, 'esg_regression.pkl'))
        
        # Predict
        pred = model.predict(vectorizer.transform([input.description]))[0]
        
        return {
            "description": input.description,
            "scores": {
                "Environment": float(round(pred[0], 2)),
                "Social": float(round(pred[1], 2)),
                "Governance": float(round(pred[2], 2))
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
def read_root():
    return {"message": "Welcome to ESG Predictor API"}