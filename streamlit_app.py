"""
Pfizer Demo - Simplified Streamlit App for Cloud Deployment
Streamlit-based interface for SAP/Kinaxis anomaly detection demo
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import json
import uuid
from faker import Faker
import random
from typing import Dict, List, Any, Optional

# Page configuration
st.set_page_config(
    page_title="Pfizer Demo - Agentic AI for Supply Chain",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .anomaly-critical {
        background-color: #ffebee;
        border-left: 4px solid #f44336;
    }
    .anomaly-high {
        background-color: #fff3e0;
        border-left: 4px solid #ff9800;
    }
    .tooltip {
        background-color: #e3f2fd;
        padding: 10px;
        border-radius: 5px;
        border-left: 4px solid #2196f3;
        margin: 10px 0;
    }
    .onboarding-modal {
        background-color: #f0f8ff;
        padding: 20px;
        border-radius: 10px;
        border: 2px solid #4CAF50;
        margin-bottom: 20px;
    }
</style>
""", unsafe_allow_html=True)

def initialize_user_tracking():
    """Initialize user tracking and onboarding state"""
    if 'user_id' not in st.session_state:
        st.session_state.user_id = str(uuid.uuid4())
    
    if 'onboarding_completed' not in st.session_state:
        st.session_state.onboarding_completed = False
    
    if 'visited_agents' not in st.session_state:
        st.session_state.visited_agents = set()
    
    if 'demo_start_time' not in st.session_state:
        st.session_state.demo_start_time = datetime.now()
    
    if 'show_tooltips' not in st.session_state:
        st.session_state.show_tooltips = True

def show_tooltip(text: str):
    """Display a tooltip with help information"""
    if st.session_state.show_tooltips:
        st.markdown(f"""
        <div class="tooltip">
            💡 <strong>Tip:</strong> {text}
        </div>
        """, unsafe_allow_html=True)

def show_onboarding_modal():
    """Show onboarding modal for new users"""
    if not st.session_state.onboarding_completed:
        st.markdown("""
        <div class="onboarding-modal">
            <h2 style="color: #2E8B57;">🏥 Welcome to the Pfizer Demo!</h2>
            <p style="font-size: 16px;">This interactive demo showcases DXC's Agentic AI capabilities for pharmaceutical supply chain data quality management.</p>
        </div>
        """, unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            **🎯 What you'll explore:**
            - 6 specialized AI agents for data validation
            - Real-time anomaly detection
            - SAP-Kinaxis integration scenarios
            - Risk assessment and execution readiness
            """)
        
        with col2:
            st.markdown("""
            **🚀 Getting started:**
            1. Select an agent from the sidebar
            2. Upload data or use synthetic samples
            3. Run analysis and explore results
            4. View detailed metrics and recommendations
            """)
        
        col1, col2, col3 = st.columns([1, 1, 1])
        with col2:
            if st.button("🎉 Start Demo", type="primary", use_container_width=True):
                st.session_state.onboarding_completed = True
                st.rerun()
        
        if st.checkbox("Don't show tips during demo"):
            st.session_state.show_tooltips = False
        
        return True
    return False

def generate_sample_data():
    """Generate sample pharmaceutical data"""
    fake = Faker()
    
    # Generate SAP-like data
    sap_data = []
    for i in range(100):
        record = {
            'material_number': f"MAT{i+1:06d}",
            'material_description': fake.catch_phrase(),
            'quantity': random.randint(100, 10000),
            'unit_price': round(random.uniform(1.0, 500.0), 2),
            'plant_code': random.choice(['PF01', 'PF02', 'PF03']),
            'batch_number': f"BATCH{random.randint(1, 1000):06d}",
            'expiry_date': fake.date_between(start_date='today', end_date='+2y'),
            'created_date': fake.date_between(start_date='-1y', end_date='today'),
            'anomaly_score': random.uniform(0, 1)
        }
        sap_data.append(record)
    
    return pd.DataFrame(sap_data)

def create_anomaly_chart(df):
    """Create anomaly visualization"""
    fig = px.scatter(df, 
                     x='quantity', 
                     y='unit_price',
                     color='anomaly_score',
                     size='anomaly_score',
                     hover_data=['material_number', 'batch_number'],
                     title="Anomaly Detection - Quantity vs Unit Price",
                     color_continuous_scale='Reds')
    
    fig.update_layout(height=500)
    return fig

def render_agent_metrics(agent_name: str, data: pd.DataFrame):
    """Render metrics for selected agent"""
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Records", len(data))
    
    with col2:
        anomalies = len(data[data['anomaly_score'] > 0.7])
        st.metric("Anomalies Detected", anomalies, delta=f"{anomalies/len(data)*100:.1f}%")
    
    with col3:
        avg_score = data['anomaly_score'].mean()
        st.metric("Avg Risk Score", f"{avg_score:.3f}")
    
    with col4:
        critical = len(data[data['anomaly_score'] > 0.9])
        st.metric("Critical Issues", critical, delta="High Priority" if critical > 0 else "All Clear")

def main():
    """Main application function"""
    initialize_user_tracking()
    
    # Show onboarding modal for new users
    if show_onboarding_modal():
        return
    
    # Header
    st.markdown('<h1 class="main-header">🏥 Pfizer Demo - Agentic AI for Supply Chain</h1>', unsafe_allow_html=True)
    
    # Sidebar
    st.sidebar.title("🔧 Control Panel")
    
    # Agent selection
    agents = {
        "Anomaly Detection Agent": "🔍 Detects anomalies in SAP-Kinaxis data synchronization",
        "Pre-Batch Validation Agent": "✅ Validates master data before Kinaxis transfer",
        "Transactional Validation Agent": "🔄 Ensures transaction consistency between systems",
        "Return Flow Validation Agent": "📋 Validates planning results for execution readiness",
        "Proactive Rule Agent": "⚡ Real-time SAP monitoring and rule enforcement",
        "Three-Way Comparison Agent": "📊 Advanced batch/streaming/reconciled data comparison"
    }
    
    selected_agent = st.sidebar.selectbox(
        "Select AI Agent",
        list(agents.keys()),
        help="Choose an AI agent to demonstrate"
    )
    
    st.sidebar.markdown(f"**Description:** {agents[selected_agent]}")
    
    # Track agent visit
    st.session_state.visited_agents.add(selected_agent)
    
    # Data controls
    st.sidebar.markdown("---")
    st.sidebar.subheader("📊 Data Controls")
    
    data_source = st.sidebar.radio(
        "Data Source",
        ["Generate Synthetic Data", "Upload CSV File"]
    )
    
    # Main content
    st.subheader(f"📈 {selected_agent} - Analysis Overview")
    show_tooltip("This overview shows key metrics from your selected agent. The data is generated synthetically for demonstration purposes.")
    
    # Generate or load data
    if data_source == "Generate Synthetic Data":
        if st.sidebar.button("🎲 Generate New Data"):
            st.session_state.sample_data = generate_sample_data()
        
        if 'sample_data' not in st.session_state:
            st.session_state.sample_data = generate_sample_data()
        
        data = st.session_state.sample_data
    else:
        uploaded_file = st.sidebar.file_uploader("Choose a CSV file", type="csv")
        if uploaded_file is not None:
            data = pd.read_csv(uploaded_file)
        else:
            st.info("Please upload a CSV file to proceed.")
            return
    
    # Display metrics
    render_agent_metrics(selected_agent, data)
    
    # Show data preview
    st.markdown("---")
    st.subheader("📋 Data Preview")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.dataframe(data.head(10), use_container_width=True)
    
    with col2:
        st.markdown("**Key Statistics:**")
        st.write(f"- Total Records: {len(data)}")
        st.write(f"- Anomalies (>0.7): {len(data[data['anomaly_score'] > 0.7])}")
        st.write(f"- Critical (>0.9): {len(data[data['anomaly_score'] > 0.9])}")
        st.write(f"- Plants: {data['plant_code'].nunique()}")
    
    # Visualization
    st.markdown("---")
    st.subheader("📊 Anomaly Visualization")
    
    fig = create_anomaly_chart(data)
    st.plotly_chart(fig, use_container_width=True)
    
    # Agent-specific insights
    st.markdown("---")
    st.subheader(f"🎯 {selected_agent} - Key Insights")
    
    insights = {
        "Anomaly Detection Agent": [
            "🔍 Detected 15 quantity anomalies in pharmaceutical batches",
            "💊 Life-saving drug shipments flagged for priority handling",
            "📈 Inventory variance exceeds 5% threshold in 3 locations"
        ],
        "Pre-Batch Validation Agent": [
            "✅ Master data validation completed for 1,247 items",
            "🚨 12 items require regulatory approval updates",
            "🏭 Production version mismatches detected in 3 BOMs"
        ],
        "Transactional Validation Agent": [
            "🔄 Transaction consistency verified across SAP-Kinaxis",
            "⚠️ 8 purchase orders show quantity discrepancies",
            "💰 Cost center allocations require manual review"
        ],
        "Return Flow Validation Agent": [
            "📋 Planning results validated for execution readiness",
            "🎯 95% of recommendations are execution-ready",
            "🚀 Critical path items prioritized for immediate action"
        ],
        "Proactive Rule Agent": [
            "⚡ Real-time monitoring active on 2,341 transactions",
            "🛡️ 23 rule violations prevented from reaching Kinaxis",
            "🔧 Auto-remediation applied to 18 data quality issues"
        ],
        "Three-Way Comparison Agent": [
            "📊 Batch, streaming, and reconciled data synchronized",
            "🎯 99.2% data consistency across all three sources",
            "🔍 Minor discrepancies identified in 12 material records"
        ]
    }
    
    for insight in insights[selected_agent]:
        st.markdown(f"- {insight}")
    
    # Footer
    st.markdown("---")
    st.markdown("**🏥 DXC Agentic AI Demo** | Pharmaceutical Supply Chain Data Quality Management")
    
    # Progress tracking
    progress = len(st.session_state.visited_agents) / len(agents)
    st.sidebar.markdown("---")
    st.sidebar.subheader("📈 Demo Progress")
    st.sidebar.progress(progress)
    st.sidebar.write(f"Explored {len(st.session_state.visited_agents)}/{len(agents)} agents")

if __name__ == "__main__":
    main()
