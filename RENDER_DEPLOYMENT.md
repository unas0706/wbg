# 🚀 Deploy to Render - Step by Step Guide

## Quick Start (5 Steps)

### Step 1: Prepare Your Code for Git

First, make sure you have a `.gitignore` file (already created). Then initialize git:

```bash
# If you haven't initialized git yet
git init
git add .
git commit -m "Ready for Render deployment"
```

### Step 2: Push to GitHub

1. **Create a new repository on GitHub:**
   - Go to https://github.com/new
   - Name it (e.g., `esg-score-predictor`)
   - Click "Create repository"

2. **Push your code:**
   ```bash
   git remote add origin https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git
   git branch -M main
   git push -u origin main
   ```

### Step 3: Create Render Account

1. Go to https://render.com
2. Sign up (you can use your GitHub account for easy connection)
3. Verify your email if required

### Step 4: Deploy on Render (Choose ONE method)

#### Method A: Using Blueprint (Easiest - Recommended) ✅

1. **Go to Render Dashboard:** https://dashboard.render.com
2. Click **"New +"** → **"Blueprint"**
3. **Connect GitHub** (if not already connected):
   - Click "Connect GitHub"
   - Authorize Render to access your repositories
4. **Select your repository** from the list
5. **Render will auto-detect `render.yaml`** - click **"Apply"**
6. Wait for deployment to complete (5-10 minutes)

#### Method B: Manual Web Service Setup

1. **Go to Render Dashboard:** https://dashboard.render.com
2. Click **"New +"** → **"Web Service"**
3. **Connect your GitHub repository:**
   - If not connected, click "Connect GitHub" and authorize
   - Select your repository
   - Click "Connect"
4. **Configure your service:**
   - **Name:** `esg-sdg-api` (or your preferred name)
   - **Region:** Choose closest to you (e.g., `Frankfurt`, `Oregon`)
   - **Branch:** `main`
   - **Root Directory:** (leave empty - uses root)
   - **Environment:** `Python 3`
   - **Build Command:** `pip install -r requirements.txt`
   - **Start Command:** `gunicorn app:app -b 0.0.0.0:$PORT --workers 2`
   - **Plan:** `Free` (for testing) or `Starter` ($7/month for always-on)
5. Click **"Create Web Service"**

### Step 5: Wait for Deployment

- Build will take 5-10 minutes the first time
- Watch the logs in real-time
- Once complete, you'll get a URL like: `https://esg-sdg-api.onrender.com`

## 🧪 Test Your Deployment

Once deployed, test your API:

### 1. Health Check
```bash
curl https://your-app-name.onrender.com/health
```

Or visit in browser: `https://your-app-name.onrender.com/health`

### 2. API Documentation
Visit: `https://your-app-name.onrender.com/`

### 3. Test Prediction
```bash
curl -X POST https://your-app-name.onrender.com/predict \
  -H "Content-Type: application/json" \
  -d '{"description": "Solar power installation with community training program"}'
```

Or use PowerShell (Windows):
```powershell
$body = @{description = "Solar power installation with community training program"} | ConvertTo-Json
Invoke-RestMethod -Uri "https://your-app-name.onrender.com/predict" -Method Post -Body $body -ContentType "application/json"
```

## 📋 Important Notes

### Free Tier Limitations:
- ⚠️ **Spins down after 15 minutes of inactivity**
- First request after spin-down takes ~30-60 seconds (cold start)
- Perfect for testing and development

### Recommended Settings for Production:

1. **Upgrade to Starter Plan ($7/month)**:
   - Service stays on 24/7
   - No cold starts
   - Better performance

2. **Or Keep Free and Handle Cold Starts**:
   - Use a service like UptimeRobot to ping `/health` every 10 minutes
   - This keeps the service awake

## 🔧 Troubleshooting

### Build Fails?

**Check:**
1. All files are in the root directory (not in `app/` subfolder)
2. `requirements.txt` is in root
3. `app.py` is in root
4. `models/` folder exists with all `.pkl` files

**Common errors:**
- ❌ `Module not found` → Check `requirements.txt` has all dependencies
- ❌ `No module named 'app'` → Make sure `app.py` is in root
- ❌ `Models not found` → Ensure `models/` folder is in git and contains `.pkl` files

### Models Not Loading?

1. Check file sizes - Render has limits on free tier
2. Verify all `.pkl` files are committed to git:
   ```bash
   git ls-files models/
   ```

3. Check model status after deployment:
   ```bash
   curl https://your-app-name.onrender.com/models/status
   ```

### App Crashes on Startup?

Check logs in Render dashboard:
1. Go to your service
2. Click "Logs" tab
3. Look for error messages

**Common fixes:**
- Ensure `gunicorn` is in `requirements.txt`
- Check Python version compatibility
- Verify all imports work

### Update Your Deployment

After making changes:
```bash
git add .
git commit -m "Your changes"
git push
```

Render will automatically detect changes and redeploy!

## 📁 Current Project Structure

```
WBG/
├── app.py              # ✅ Main Flask application
├── requirements.txt    # ✅ Python dependencies
├── models/             # ✅ Model files
│   ├── esg_regression.pkl
│   ├── sdg_regression.pkl
│   └── vectorizer.pkl
├── Procfile            # ✅ Process file
├── render.yaml         # ✅ Render config (for Blueprint method)
└── .gitignore          # ✅ Git ignore file
```

## 🎯 Quick Checklist Before Deploying

- [ ] Code is pushed to GitHub
- [ ] All files are in root (not in subfolders)
- [ ] `requirements.txt` exists with all dependencies
- [ ] `app.py` is in root directory
- [ ] `models/` folder exists with all `.pkl` files
- [ ] `Procfile` or `render.yaml` is configured correctly
- [ ] Tested locally - app runs without errors

## 🆘 Need Help?

- Render Docs: https://render.com/docs
- Render Community: https://community.render.com
- Check Render dashboard logs for specific error messages

---

**Ready to deploy?** Follow Steps 1-4 above! 🚀

