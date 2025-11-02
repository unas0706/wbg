# Render Deployment Guide

This guide will help you deploy the ESG Score Predictor API to Render.

## Prerequisites

1. A GitHub account
2. Your code pushed to a GitHub repository
3. A Render account (sign up at https://render.com)

## Step 1: Push Your Code to GitHub

1. Initialize git if you haven't already:
   ```bash
   git init
   git add .
   git commit -m "Initial commit"
   ```

2. Create a new repository on GitHub and push your code:
   ```bash
   git remote add origin YOUR_GITHUB_REPO_URL
   git branch -M main
   git push -u origin main
   ```

## Step 2: Deploy on Render

### Option A: Using render.yaml (Recommended)

1. Update `render.yaml` with your GitHub repository URL:
   - Replace `REPLACE_WITH_YOUR_REPO_URL` with your actual GitHub repository URL

2. Go to https://dashboard.render.com
3. Click "New" → "Blueprint"
4. Connect your GitHub account if not already connected
5. Select your repository
6. Render will automatically detect `render.yaml` and deploy your service

### Option B: Manual Setup

1. Go to https://dashboard.render.com
2. Click "New" → "Web Service"
3. Connect your GitHub account and select your repository
4. Configure the service:
   - **Name**: `esg-sdg-api` (or any name you prefer)
   - **Environment**: `Python 3`
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `gunicorn app:app -b 0.0.0.0:$PORT --workers 2`
   - **Plan**: Choose your plan (Starter is free for testing)

5. Click "Create Web Service"

## Step 3: Environment Variables (Optional)

If you want to use S3 for model storage instead of including them in the repository:

1. In your Render dashboard, go to your service
2. Navigate to "Environment" tab
3. Add these environment variables:
   - `MODEL_S3_BUCKET`: Your S3 bucket name
   - `MODEL_S3_PREFIX`: Your S3 prefix/path (optional)

## Step 4: Verify Deployment

1. Once deployed, Render will provide you with a URL like: `https://your-app-name.onrender.com`
2. Test the health endpoint: `https://your-app-name.onrender.com/health`
3. Test the API documentation: `https://your-app-name.onrender.com/`

## Project Structure

After cleanup, your root folder should contain:
```
WBG/
├── app.py              # Main Flask application
├── requirements.txt    # Python dependencies
├── models/             # Model files
│   ├── esg_regression.pkl
│   ├── sdg_regression.pkl
│   └── vectorizer.pkl
├── Procfile            # Process file for deployment
└── render.yaml         # Render configuration file
```

## Important Files for Deployment

- `app.py` - Main Flask application
- `requirements.txt` - Python dependencies
- `models/` - Model files directory
- `Procfile` - Process file for deployment (optional, if using render.yaml)
- `render.yaml` - Render configuration file

## API Endpoints

Once deployed, your API will have these endpoints:

- `GET /` - API documentation
- `GET /health` - Health check
- `GET /models/status` - Check model loading status
- `POST /predict` - Predict ESG scores

### Example POST Request

```bash
curl -X POST https://your-app-name.onrender.com/predict \
  -H "Content-Type: application/json" \
  -d '{"description": "Solar power installation with community training program"}'
```

## Troubleshooting

1. **Build fails**: Check that all dependencies in `app/requirements.txt` are correct
2. **Models not loading**: Ensure `app/models/` directory contains all `.pkl` files
3. **Port errors**: Make sure your app uses `$PORT` environment variable (already configured)
4. **Import errors**: Verify all required Python packages are in `requirements.txt`

## Notes

- The free tier on Render spins down after 15 minutes of inactivity
- First request after spin-down may take longer (cold start)
- Consider upgrading to a paid plan for production use

