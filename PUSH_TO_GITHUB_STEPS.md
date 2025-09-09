# 🚀 Steps to Push Your Pfizer Demo to GitHub

## Step 1: Create GitHub Repository

1. Go to [github.com](https://github.com) and sign in
2. Click the "+" icon in the top right → "New repository"
3. Fill in the details:
   - **Repository name:** `pfizer-demo-agentic-ai`
   - **Description:** `DXC Agentic AI Demo for Pharmaceutical Supply Chain Data Quality Management`
   - **Visibility:** Public ✅
   - **DO NOT** check "Add a README file" (we already have one)
   - **DO NOT** check "Add .gitignore" (we already have one)
   - **DO NOT** check "Choose a license" (optional)
4. Click "Create repository"

## Step 2: Copy Repository URL

After creating the repository, GitHub will show you a page with setup instructions. 
Copy the HTTPS URL that looks like:
`https://github.com/YOUR_USERNAME/pfizer-demo-agentic-ai.git`

## Step 3: Run These Commands

Once you have the repository URL, run these commands in your terminal:

```bash
cd /Users/peterbentley/CascadeProjects/Pfizer-Demo

# Add your GitHub repository as remote
git remote add origin https://github.com/YOUR_USERNAME/pfizer-demo-agentic-ai.git

# Push your code to GitHub
git push -u origin main
```

## What Will Be Uploaded

Your repository will contain:
- ✅ 23 files including all source code
- ✅ 6 specialized AI agents
- ✅ Interactive Streamlit dashboard
- ✅ Onboarding system with tooltips
- ✅ Production-ready configuration
- ✅ Comprehensive documentation

## After Successful Push

Once pushed, your repository will be available at:
`https://github.com/YOUR_USERNAME/pfizer-demo-agentic-ai`

Then you can deploy to Streamlit Community Cloud:
1. Visit [share.streamlit.io](https://share.streamlit.io)
2. Connect your GitHub repository
3. Set main file: `src/dashboard/app.py`
4. Deploy!

---

**Ready to push! Just create the GitHub repository first, then run the commands above.**
