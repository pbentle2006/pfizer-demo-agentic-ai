# 🚀 Pfizer Demo - Public Deployment Guide

This guide will help you deploy the Pfizer Demo to make it publicly accessible.

## 📋 Prerequisites

1. **GitHub Account** - Required for Streamlit Community Cloud
2. **Streamlit Community Cloud Account** - Free at [share.streamlit.io](https://share.streamlit.io)

## 🎯 Deployment Options

### Option 1: Streamlit Community Cloud (Recommended)

**Why Streamlit Community Cloud?**
- ✅ Free hosting for Streamlit apps
- ✅ Automatic deployments from GitHub
- ✅ Built-in Python environment
- ✅ Easy sharing with custom URLs

**Steps:**

1. **Push to GitHub:**
   ```bash
   cd /Users/peterbentley/CascadeProjects/Pfizer-Demo
   git init
   git add .
   git commit -m "Initial commit - Pfizer Demo with onboarding"
   git branch -M main
   git remote add origin https://github.com/YOUR_USERNAME/pfizer-demo.git
   git push -u origin main
   ```

2. **Deploy on Streamlit Community Cloud:**
   - Visit [share.streamlit.io](https://share.streamlit.io)
   - Click "New app"
   - Connect your GitHub repository
   - Set main file path: `src/dashboard/app.py`
   - Click "Deploy!"

3. **Custom URL:**
   - Your app will be available at: `https://YOUR_USERNAME-pfizer-demo-src-dashboard-app-HASH.streamlit.app/`
   - You can customize the URL in the Streamlit Cloud settings

### Option 2: Heroku (Alternative)

**Steps:**
1. Create `Procfile`:
   ```
   web: streamlit run src/dashboard/app.py --server.port=$PORT --server.address=0.0.0.0
   ```

2. Deploy to Heroku:
   ```bash
   heroku create pfizer-demo-agentic-ai
   git push heroku main
   ```

### Option 3: Railway (Alternative)

**Steps:**
1. Connect GitHub repository to Railway
2. Set start command: `streamlit run src/dashboard/app.py --server.port=$PORT --server.address=0.0.0.0`
3. Deploy automatically

## 🔧 Configuration Files Created

- ✅ `.streamlit/config.toml` - Streamlit configuration
- ✅ `requirements.txt` - Python dependencies
- ✅ `packages.txt` - System packages (if needed)
- ✅ `.gitignore` - Git ignore patterns
- ✅ `netlify.toml` - Netlify configuration (redirects to Streamlit)

## 🎨 Features Added for Public Demo

### 🎯 Onboarding System
- **Welcome modal** for new users
- **Interactive tutorial** explaining the demo
- **User tracking** with unique session IDs
- **Agent visit tracking** to monitor user engagement

### 💡 Tooltip System
- **Contextual help** throughout the application
- **Toggle-able tips** (users can disable if desired)
- **Guided experience** for first-time users

### 📊 Enhanced UX
- **Professional styling** with custom CSS
- **Responsive design** for different screen sizes
- **Clear navigation** with agent descriptions
- **Progress tracking** for demo completion

## 🌐 Sharing Your Demo

Once deployed, you can share your demo with:

1. **Direct URL:** `https://your-app-name.streamlit.app/`
2. **QR Code:** Generate using the URL for mobile access
3. **Embed:** Use iframe to embed in presentations
4. **Social Media:** Share the link with descriptions

## 🔒 Security Considerations

- ✅ No sensitive data hardcoded
- ✅ Synthetic data only (no real Pfizer data)
- ✅ No API keys required for basic functionality
- ✅ Public-safe configuration

## 📈 Analytics & Tracking

The app includes:
- **User session tracking** (anonymous UUIDs)
- **Agent usage analytics** 
- **Demo completion metrics**
- **Performance monitoring**

## 🛠️ Maintenance

To update your deployed app:
1. Make changes locally
2. Commit and push to GitHub
3. Streamlit Community Cloud auto-deploys from main branch

## 📞 Support

If you encounter issues:
1. Check Streamlit Community Cloud logs
2. Verify all dependencies in `requirements.txt`
3. Test locally first: `streamlit run src/dashboard/app.py`

---

**🎉 Your Pfizer Demo is now ready for public sharing!**
