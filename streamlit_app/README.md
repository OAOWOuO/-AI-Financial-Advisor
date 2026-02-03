# AI Hedge Fund - Streamlit Dashboard

A web-based dashboard for running AI hedge fund analysis with multiple AI agents.

## Features

- **Multi-Agent Analysis**: Choose from 12+ AI analysts including Warren Buffett, Charlie Munger, and technical/fundamental agents
- **Portfolio Recommendations**: Get actionable BUY/SHORT/HOLD signals with confidence scores
- **Signal Visualization**: Interactive charts showing bullish/bearish/neutral distributions
- **Agent Leaderboard**: See which AI agents have the highest confidence in their predictions

## Quick Start (Local)

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Set up your API keys:
```bash
cp .streamlit/secrets.toml.example .streamlit/secrets.toml
# Edit secrets.toml with your OpenAI API key
```

3. Run the app:
```bash
streamlit run app.py
```

## Deploy to Streamlit Cloud (share.streamlit.io)

### Step 1: Push to GitHub

Make sure your code is pushed to GitHub:
```bash
git add streamlit_app/
git commit -m "Add Streamlit dashboard"
git push origin main
```

### Step 2: Deploy on Streamlit Cloud

1. Go to [share.streamlit.io](https://share.streamlit.io)
2. Click **"New app"**
3. Connect your GitHub account
4. Select your repository: `ai-hedge-fund-repo`
5. Set the **Main file path**: `streamlit_app/app.py`
6. Click **"Deploy"**

### Step 3: Add Secrets

After deploying:
1. Click **"Settings"** (gear icon) on your deployed app
2. Go to **"Secrets"**
3. Add your API keys:

```toml
OPENAI_API_KEY = "sk-your-actual-openai-api-key"
FINANCIAL_DATASETS_API_KEY = "your-financial-datasets-key"
```

4. Click **"Save"**

Your app will restart and be ready to use!

## Share Your App

Once deployed, you'll get a URL like:
```
https://your-username-ai-hedge-fund-repo-streamlit-appapp-xxxxx.streamlit.app
```

Share this link with anyone to let them use your AI Hedge Fund dashboard!

## Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `OPENAI_API_KEY` | Yes | Your OpenAI API key for GPT models |
| `FINANCIAL_DATASETS_API_KEY` | No | For fetching financial data |

## Troubleshooting

### "Missing OPENAI_API_KEY" error
Make sure you've added your API key in the Streamlit Cloud secrets settings.

### App is slow
The first run may take longer as models initialize. Subsequent runs will be faster.

### Import errors
Ensure all dependencies in `requirements.txt` are installed. The app needs access to the parent `src/` directory for the hedge fund analysis code.
