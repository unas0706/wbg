# ESG Score Predictor API

This project provides an API that predicts Environmental, Social, and Governance (ESG) scores for project descriptions using machine learning.

## Project Structure

```
app/
├── api.py              # FastAPI application
├── train_model.py      # Script to train the ML model
├── models/            # Directory for saved ML models
│   ├── vectorizer.pkl
│   └── esg_regression.pkl
└── data/             # Directory for dataset files
    ├── projects.csv
    └── projects_scored.csv
```

## Prerequisites

- Python 3.8 or higher
- pip (Python package installer)

## Local Development

1. Clone this repository:

   ```bash
   git clone <your-repository-url>
   cd <repository-name>
   ```

2. Install required packages:

   ```bash
   pip install -r requirements.txt
   ```

3. Train the model:

   ```bash
   python app/train_model.py
   ```

4. Start the API server:

   ```bash
   uvicorn app.api:app --reload
   ```

5. The API will be available at `http://localhost:8000`
   - API documentation: `http://localhost:8000/docs`
   - OpenAPI specification: `http://localhost:8000/openapi.json`

## API Usage

### Predict ESG Scores

**Endpoint:** `POST /predict`

**Request Body:**

```json
{
  "description": "Your project description here"
}
```

**Example Request:**

```bash
curl -X POST "http://localhost:8000/predict" \
     -H "Content-Type: application/json" \
     -d "{\"description\": \"This project installs solar microgrids for rural villages and trains women workers.\"}"
```

**Example Response:**

```json
{
  "description": "This project installs solar microgrids for rural villages and trains women workers.",
  "scores": {
    "Environment": 0.06,
    "Social": 0.26,
    "Governance": 0.04
  }
}
```

## Deployment to Render

1. Create a [Render](https://render.com) account if you haven't already.

2. Fork or push this repository to GitHub.

3. On Render dashboard:

   - Click "New +"
   - Select "Web Service"
   - Connect your GitHub repository
   - Configure the deployment:
     - **Name**: `esg-score-predictor` (or your preferred name)
     - **Environment**: `Python 3`
     - **Build Command**: `pip install -r requirements.txt && python app/train_model.py`
     - **Start Command**: `uvicorn app.api:app --host 0.0.0.0 --port $PORT`
   - Click "Create Web Service"

4. Render will automatically deploy your application and provide you with a URL.

## Environment Variables

- `PORT`: The port number for the API server (default: 8000)

## Model Training

The model is trained on a sample dataset of project descriptions with the following features:

- Environmental keywords (renewable, solar, climate, etc.)
- Social keywords (education, health, community, etc.)
- Governance keywords (transparency, policy, regulation, etc.)

To retrain the model with your own data, modify the `train_model.py` script.

## License

[Your chosen license]

## Contributing

[Your contribution guidelines]
