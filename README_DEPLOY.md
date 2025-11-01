Deployment to Render

This repository contains a Flask API (`app/app.py`) for ESG and SDG scoring.

Quick steps to deploy to Render:

1. Choose model hosting strategy

   - Option A: Commit `app/models/*.pkl` to repo (use Git LFS for .pkl files)
   - Option B: Upload models to S3 and set `MODEL_S3_BUCKET` and optional `MODEL_S3_PREFIX` environment variables in Render

2. Ensure `requirements.txt` includes dependencies. This repo includes:

   - Flask, gunicorn, scikit-learn, joblib, numpy, pandas, boto3, requests

3. Start command (Render):

   - `gunicorn app.app:app -b 0.0.0.0:$PORT --workers 2`
   - Or use the provided `Procfile`.

4. Environment variables (if using S3):

   - MODEL_S3_BUCKET: your-bucket-name
   - MODEL_S3_PREFIX: optional/path/to/models (without trailing slash)
   - AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, AWS_REGION (if using AWS)

5. Set health check path to `/health`.

6. After deploy, check logs and test endpoints:
   - GET /health
   - POST /predict {"description": "..."}

Notes

- If you prefer Git LFS for model files: `git lfs install; git lfs track "*.pkl"; git add .gitattributes` then commit and push model files.
- Keep credentials secret — use Render environment settings.
