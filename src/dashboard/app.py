"""
Pfizer Demo - Main Dashboard Application
Streamlit-based interface for SAP/Kinaxis anomaly detection
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import sys
import os
from datetime import datetime, timedelta
import json
import uuid
import os
from typing import Dict, List, Any, Optional

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from src.agents.anomaly_detection_agent import AnomalyDetectionAgent, AnomalyResult, DetectionSummary
from src.agents.pre_batch_validation_agent import PreBatchValidationAgent
from src.agents.transactional_validation_agent import TransactionalValidationAgent
from src.agents.return_flow_validation_agent import ReturnFlowValidationAgent
from src.agents.proactive_rule_agent import ProactiveRuleAgent
from src.agents.three_way_comparison_agent import ThreeWayComparisonAgent
from src.data.sap_data_generator import SAPDataGenerator
from src.data.kinaxis_data_generator import KinaxisDataGenerator

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
    .anomaly-medium {
        background-color: #f3e5f5;
        border-left: 4px solid #9c27b0;
    }
    .anomaly-low {
        background-color: #e8f5e8;
        border-left: 4px solid #4caf50;
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

def show_tooltip(text: str, key: str = None):
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

def track_agent_visit(agent_name: str):
    """Track which agents the user has visited"""
    st.session_state.visited_agents.add(agent_name)

def initialize_session_state():
    """Initialize session state variables"""
    if 'anomaly_agent' not in st.session_state:
        st.session_state.anomaly_agent = AnomalyDetectionAgent()
    
    if 'pre_batch_agent' not in st.session_state:
        st.session_state.pre_batch_agent = PreBatchValidationAgent()
    
    if 'transactional_agent' not in st.session_state:
        st.session_state.transactional_agent = TransactionalValidationAgent()
    
    if 'return_flow_agent' not in st.session_state:
        st.session_state.return_flow_agent = ReturnFlowValidationAgent()
    
    if 'proactive_agent' not in st.session_state:
        st.session_state.proactive_agent = ProactiveRuleAgent()
    
    if 'three_way_agent' not in st.session_state:
        st.session_state.three_way_agent = ThreeWayComparisonAgent()
    
    if 'sap_data' not in st.session_state:
        st.session_state.sap_data = None
    
    if 'kinaxis_data' not in st.session_state:
        st.session_state.kinaxis_data = None
    
    if 'anomaly_results' not in st.session_state:
        st.session_state.anomaly_results = []
    
    if 'detection_summary' not in st.session_state:
        st.session_state.detection_summary = None
    
    if 'processed_data' not in st.session_state:
        st.session_state.processed_data = None
    
    if 'selected_agent' not in st.session_state:
        st.session_state.selected_agent = 'Anomaly Detection Agent'
    
    if 'agent_results' not in st.session_state:
        st.session_state.agent_results = {}

def render_header():
    """Render the main header"""
    st.markdown('<h1 class="main-header">🏥 Pfizer Demo - Agentic AI Anomaly Detection</h1>', unsafe_allow_html=True)
    st.markdown("---")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.info("**SAP S/4 HANA** Staging Data")
    with col2:
        st.info("**Kinaxis** Master Data")
    with col3:
        st.info("**AI Agent** Analysis")

def render_sidebar():
    """Render the sidebar with controls"""
    st.sidebar.title("🔧 Control Panel")
    
    # Agent Selection
    st.sidebar.subheader("🤖 Agent Selection")
    
    agent_options = {
        'Anomaly Detection Agent': '🔍 General anomaly detection',
        'Pre-Batch Validation Agent': '📋 Master data validation',
        'Transactional Validation Agent': '💼 Transaction consistency',
        'Return Flow Validation Agent': '🔄 Planning result validation (CRITICAL)',
        'Proactive Rule Agent': '⚡ Real-time rule enforcement',
        'Three-Way Comparison Agent': '📊 Advanced data comparison'
    }
    
    selected_agent = st.sidebar.selectbox(
        "Select Agent to Run",
        options=list(agent_options.keys()),
        index=0,
        format_func=lambda x: agent_options[x]
    )
    
    st.session_state.selected_agent = selected_agent
    
    # Agent description
    agent_descriptions = {
        'Anomaly Detection Agent': 'Detects general anomalies between SAP and Kinaxis data using rule-based, statistical, and ML methods.',
        'Pre-Batch Validation Agent': 'Validates SAP master data before batch transfer to Kinaxis. Critical for preventing downstream errors.',
        'Transactional Validation Agent': 'Ensures transactional data consistency between SAP and Kinaxis systems.',
        'Return Flow Validation Agent': 'MOST CRITICAL - Validates planning results from Kinaxis back to SAP for execution readiness.',
        'Proactive Rule Agent': 'Real-time monitoring within SAP to prevent bad data from reaching Kinaxis.',
        'Three-Way Comparison Agent': 'Advanced comparison of batch, streaming, and reconciled data sources.'
    }
    
    st.sidebar.info(agent_descriptions[selected_agent])
    
    # Demo data generation
    st.sidebar.subheader("📊 Demo Data")
    
    if st.sidebar.button("Generate Sample Data", type="primary"):
        with st.spinner("Generating synthetic data..."):
            # Generate SAP data
            sap_generator = SAPDataGenerator()
            sap_datasets = sap_generator.generate_full_dataset(
                material_records=500,
                production_records=200,
                inventory_records=800,
                bom_records=300,
                anomaly_rate=0.15
            )
            
            # Generate Kinaxis data
            kinaxis_generator = KinaxisDataGenerator()
            kinaxis_datasets = kinaxis_generator.generate_full_dataset(
                item_records=500,
                network_records=300,
                forecast_records=600,
                schedule_records=250,
                sap_data=sap_datasets,
                mismatch_rate=0.20
            )
            
            st.session_state.sap_data = sap_datasets
            st.session_state.kinaxis_data = kinaxis_datasets
            
        st.sidebar.success("✅ Sample data generated!")
    
    # File upload section
    st.sidebar.subheader("📁 Upload Data")
    
    sap_file = st.sidebar.file_uploader(
        "Upload SAP Data",
        type=['csv', 'xlsx'],
        help="Upload SAP S/4 HANA staging data"
    )
    
    kinaxis_file = st.sidebar.file_uploader(
        "Upload Kinaxis Data", 
        type=['csv', 'xlsx'],
        help="Upload Kinaxis master data"
    )
    
    if sap_file and kinaxis_file:
        # Process uploaded files
        try:
            if sap_file.name.endswith('.csv'):
                sap_df = pd.read_csv(sap_file)
            else:
                sap_df = pd.read_excel(sap_file)
            
            if kinaxis_file.name.endswith('.csv'):
                kinaxis_df = pd.read_csv(kinaxis_file)
            else:
                kinaxis_df = pd.read_excel(kinaxis_file)
            
            st.session_state.sap_data = {'uploaded_data': sap_df}
            st.session_state.kinaxis_data = {'uploaded_data': kinaxis_df}
            
            st.sidebar.success("✅ Files uploaded successfully!")
            
        except Exception as e:
            st.sidebar.error(f"❌ Error processing files: {str(e)}")
    
    # Analysis controls
    st.sidebar.subheader("🔍 Analysis")
    
    if st.session_state.sap_data and st.session_state.kinaxis_data:
        if st.sidebar.button(f"Run {selected_agent}", type="primary"):
            run_selected_agent(selected_agent)
    
    # Settings
    st.sidebar.subheader("⚙️ Settings")
    
    st.sidebar.slider(
        "Anomaly Sensitivity",
        min_value=0.1,
        max_value=1.0,
        value=0.5,
        step=0.1,
        help="Adjust sensitivity of anomaly detection"
    )
    
    auto_fix = st.sidebar.checkbox(
        "Enable Auto-Remediation",
        value=True,
        help="Automatically fix simple anomalies"
    )

def run_selected_agent(agent_name: str):
    """Run the selected agent"""
    with st.spinner(f"🤖 {agent_name} analyzing data..."):
        try:
            if agent_name == 'Anomaly Detection Agent':
                run_anomaly_detection()
            elif agent_name == 'Pre-Batch Validation Agent':
                run_pre_batch_validation()
            elif agent_name == 'Transactional Validation Agent':
                run_transactional_validation()
            elif agent_name == 'Return Flow Validation Agent':
                run_return_flow_validation()
            elif agent_name == 'Proactive Rule Agent':
                run_proactive_rules()
            elif agent_name == 'Three-Way Comparison Agent':
                run_three_way_comparison()
            
        except Exception as e:
            st.error(f"❌ {agent_name} failed: {str(e)}")

def run_anomaly_detection():
    """Run the anomaly detection process"""
    # Get first dataset from each source for comparison
    sap_df = list(st.session_state.sap_data.values())[0]
    kinaxis_df = list(st.session_state.kinaxis_data.values())[0]
    
    # Run anomaly detection
    anomalies, summary = st.session_state.anomaly_agent.detect_anomalies(sap_df, kinaxis_df)
    
    st.session_state.anomaly_results = anomalies
    st.session_state.detection_summary = summary
    st.session_state.agent_results['anomaly'] = {'anomalies': anomalies, 'summary': summary}
    
    # Auto-remediation if enabled
    if st.sidebar.checkbox("Enable Auto-Remediation", value=True):
        merged_data = st.session_state.anomaly_agent._merge_datasets(sap_df, kinaxis_df)
        corrected_data, remediation_log = st.session_state.anomaly_agent.auto_remediate(
            anomalies, merged_data
        )
        st.session_state.processed_data = corrected_data
    
    st.success(f"✅ Anomaly Detection complete! Found {len(anomalies)} anomalies in {summary.processing_time}s")

def run_pre_batch_validation():
    """Run pre-batch validation"""
    sap_data = st.session_state.sap_data
    
    # Run pre-batch validation
    violations, summary = st.session_state.pre_batch_agent.validate_pre_batch_data(sap_data)
    
    st.session_state.agent_results['pre_batch'] = {'violations': violations, 'summary': summary}
    
    st.success(f"✅ Pre-Batch Validation complete! Found {len(violations)} violations. Data Quality Score: {summary['data_quality_score']}%")

def run_transactional_validation():
    """Run transactional validation"""
    sap_data = st.session_state.sap_data
    kinaxis_data = st.session_state.kinaxis_data
    
    # Run transactional validation
    violations, summary = st.session_state.transactional_agent.validate_transactional_data(sap_data, kinaxis_data)
    
    st.session_state.agent_results['transactional'] = {'violations': violations, 'summary': summary}
    
    st.success(f"✅ Transactional Validation complete! Found {len(violations)} violations. Consistency Score: {summary['consistency_score']}%")

def run_return_flow_validation():
    """Run return flow validation (MOST CRITICAL)"""
    kinaxis_data = st.session_state.kinaxis_data
    sap_data = st.session_state.sap_data
    
    # Simulate capacity data
    capacity_data = {'work_centers': pd.DataFrame()}
    
    # Run return flow validation
    violations, summary = st.session_state.return_flow_agent.validate_return_flow(
        kinaxis_data, sap_data, capacity_data
    )
    
    st.session_state.agent_results['return_flow'] = {'violations': violations, 'summary': summary}
    
    execution_readiness = summary['execution_readiness_score']
    risk_level = summary['execution_risk_level']
    
    if risk_level == 'CRITICAL':
        st.error(f"🚨 CRITICAL: Execution Readiness: {execution_readiness}% - {summary['recommendation']}")
    elif risk_level == 'HIGH':
        st.warning(f"⚠️ HIGH RISK: Execution Readiness: {execution_readiness}% - {summary['recommendation']}")
    else:
        st.success(f"✅ Return Flow Validation complete! Execution Readiness: {execution_readiness}% - {summary['recommendation']}")

def run_proactive_rules():
    """Run proactive rule scanning"""
    sap_data = st.session_state.sap_data
    
    # Run proactive scanning
    violations, summary = st.session_state.proactive_agent.proactive_scan(sap_data)
    
    st.session_state.agent_results['proactive'] = {'violations': violations, 'summary': summary}
    
    prevention_score = summary['prevention_effectiveness_score']
    action_required = summary['action_required']
    
    if 'BLOCK' in action_required:
        st.error(f"🚫 {action_required} - Prevention Score: {prevention_score}%")
    elif 'REVIEW' in action_required:
        st.warning(f"⚠️ {action_required} - Prevention Score: {prevention_score}%")
    else:
        st.success(f"✅ Proactive Rules complete! {action_required} - Prevention Score: {prevention_score}%")

def run_three_way_comparison():
    """Run three-way comparison"""
    # Simulate batch, streaming, and reconciled data
    batch_data = st.session_state.sap_data
    streaming_data = st.session_state.kinaxis_data
    reconciled_data = st.session_state.sap_data  # Simulate reconciled as SAP data
    
    # Run three-way comparison
    results, summary = st.session_state.three_way_agent.perform_three_way_comparison(
        batch_data, streaming_data, reconciled_data
    )
    
    st.session_state.agent_results['three_way'] = {'results': results, 'summary': summary}
    
    consistency_score = summary['average_consistency_score']
    data_quality = summary['data_quality_assessment']
    
    st.success(f"✅ Three-Way Comparison complete! Data Quality: {data_quality} - Consistency: {consistency_score:.1%}")

def render_overview_metrics():
    """Render overview metrics"""
    selected_agent = st.session_state.selected_agent
    
    if selected_agent not in st.session_state.agent_results:
        st.info(f"👆 Generate or upload data and run {selected_agent} to see results")
        return
    
    # Display metrics based on selected agent
    if selected_agent == 'Anomaly Detection Agent':
        render_anomaly_metrics()
    elif selected_agent == 'Pre-Batch Validation Agent':
        render_pre_batch_metrics()
    elif selected_agent == 'Transactional Validation Agent':
        render_transactional_metrics()
    elif selected_agent == 'Return Flow Validation Agent':
        render_return_flow_metrics()
    elif selected_agent == 'Proactive Rule Agent':
        render_proactive_metrics()
    elif selected_agent == 'Three-Way Comparison Agent':
        render_three_way_metrics()

def render_anomaly_metrics():
    """Render anomaly detection metrics"""
    if 'anomaly' not in st.session_state.agent_results:
        return
    
    summary = st.session_state.agent_results['anomaly']['summary']
    
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.metric(
            label="Total Records",
            value=f"{summary.total_records:,}",
            help="Total number of records analyzed"
        )
    
    with col2:
        st.metric(
            label="Anomalies Found", 
            value=f"{summary.anomalies_found:,}",
            delta=f"{summary.anomaly_rate}%",
            help="Number and percentage of anomalous records"
        )
    
    with col3:
        st.metric(
            label="Auto-Fixable",
            value=f"{summary.auto_fixable_count:,}",
            delta=f"{(summary.auto_fixable_count/summary.anomalies_found*100):.1f}%" if summary.anomalies_found > 0 else "0%",
            help="Anomalies that can be automatically fixed"
        )
    
    with col4:
        st.metric(
            label="Manual Review",
            value=f"{summary.manual_review_count:,}",
            delta=f"{(summary.manual_review_count/summary.anomalies_found*100):.1f}%" if summary.anomalies_found > 0 else "0%",
            help="Anomalies requiring manual review"
        )
    
    with col5:
        st.metric(
            label="Processing Time",
            value=f"{summary.processing_time}s",
            help="Time taken to analyze the data"
        )

def render_pre_batch_metrics():
    """Render pre-batch validation metrics"""
    if 'pre_batch' not in st.session_state.agent_results:
        return
    
    summary = st.session_state.agent_results['pre_batch']['summary']
    violations = st.session_state.agent_results['pre_batch']['violations']
    
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.metric(
            label="Total Validations",
            value=f"{summary['total_validations']:,}",
            help="Total number of validation checks performed"
        )
    
    with col2:
        st.metric(
            label="Violations Found",
            value=f"{len(violations):,}",
            help="Number of validation violations detected"
        )
    
    with col3:
        st.metric(
            label="Data Quality Score",
            value=f"{summary['data_quality_score']}%",
            help="Overall data quality assessment"
        )
    
    with col4:
        st.metric(
            label="Auto-Fixable",
            value=f"{summary['auto_fixable_count']:,}",
            help="Violations that can be automatically fixed"
        )
    
    with col5:
        st.metric(
            label="Transfer Readiness",
            value=summary['transfer_readiness'],
            help="Readiness for batch transfer to Kinaxis"
        )

def render_transactional_metrics():
    """Render transactional validation metrics"""
    if 'transactional' not in st.session_state.agent_results:
        return
    
    summary = st.session_state.agent_results['transactional']['summary']
    violations = st.session_state.agent_results['transactional']['violations']
    
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.metric(
            label="Records Compared",
            value=f"{summary['total_comparisons']:,}",
            help="Total records compared between systems"
        )
    
    with col2:
        st.metric(
            label="Inconsistencies",
            value=f"{len(violations):,}",
            help="Data inconsistencies found"
        )
    
    with col3:
        st.metric(
            label="Consistency Score",
            value=f"{summary['consistency_score']}%",
            help="Overall data consistency between systems"
        )
    
    with col4:
        st.metric(
            label="Critical Issues",
            value=f"{summary['critical_inconsistencies']:,}",
            help="Critical inconsistencies requiring immediate attention"
        )
    
    with col5:
        st.metric(
            label="Processing Time",
            value=f"{summary['processing_time']}s",
            help="Time taken for validation"
        )

def render_return_flow_metrics():
    """Render return flow validation metrics"""
    if 'return_flow' not in st.session_state.agent_results:
        return
    
    summary = st.session_state.agent_results['return_flow']['summary']
    violations = st.session_state.agent_results['return_flow']['violations']
    
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.metric(
            label="Planning Records",
            value=f"{summary['total_validations']:,}",
            help="Total planning records validated"
        )
    
    with col2:
        st.metric(
            label="Execution Issues",
            value=f"{len(violations):,}",
            help="Issues that would prevent execution"
        )
    
    with col3:
        st.metric(
            label="Execution Readiness",
            value=f"{summary['execution_readiness_score']}%",
            help="Readiness score for SAP execution"
        )
    
    with col4:
        st.metric(
            label="Critical Issues",
            value=f"{summary['critical_issues']:,}",
            help="Critical issues blocking execution"
        )
    
    with col5:
        st.metric(
            label="Risk Level",
            value=summary['execution_risk_level'],
            help="Overall execution risk assessment"
        )

def render_proactive_metrics():
    """Render proactive rule metrics"""
    if 'proactive' not in st.session_state.agent_results:
        return
    
    summary = st.session_state.agent_results['proactive']['summary']
    violations = st.session_state.agent_results['proactive']['violations']
    
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.metric(
            label="Records Scanned",
            value=f"{summary['total_violations']:,}",
            help="Total records scanned for rule violations"
        )
    
    with col2:
        st.metric(
            label="Rule Violations",
            value=f"{len(violations):,}",
            help="Rule violations detected"
        )
    
    with col3:
        st.metric(
            label="Prevention Score",
            value=f"{summary['prevention_effectiveness_score']}%",
            help="Effectiveness of prevention measures"
        )
    
    with col4:
        st.metric(
            label="Blocked Records",
            value=f"{summary['blocked_records']:,}",
            help="Records blocked from transfer"
        )
    
    with col5:
        st.metric(
            label="Auto-Fixed",
            value=f"{summary['auto_fixable_count']:,}",
            help="Issues automatically fixed"
        )

def render_three_way_metrics():
    """Render three-way comparison metrics"""
    if 'three_way' not in st.session_state.agent_results:
        return
    
    summary = st.session_state.agent_results['three_way']['summary']
    results = st.session_state.agent_results['three_way']['results']
    
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.metric(
            label="Comparisons",
            value=f"{summary['total_comparisons']:,}",
            help="Total three-way comparisons performed"
        )
    
    with col2:
        st.metric(
            label="Inconsistencies",
            value=f"{len(results):,}",
            help="Inconsistencies found across data sources"
        )
    
    with col3:
        st.metric(
            label="Consistency Score",
            value=f"{summary['average_consistency_score']:.1%}",
            help="Average consistency across all comparisons"
        )
    
    with col4:
        st.metric(
            label="Data Quality",
            value=summary['data_quality_assessment'],
            help="Overall data quality assessment"
        )
    
    with col5:
        st.metric(
            label="Sync Health",
            value=summary['synchronization_health'],
            help="Data synchronization health status"
        )

def render_anomaly_breakdown():
    """Render anomaly breakdown charts"""
    if not st.session_state.detection_summary or not st.session_state.anomaly_results:
        return
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Anomaly types pie chart
        breakdown = st.session_state.detection_summary.anomaly_breakdown
        if breakdown:
            fig_pie = px.pie(
                values=list(breakdown.values()),
                names=list(breakdown.keys()),
                title="Anomaly Types Distribution",
                color_discrete_sequence=px.colors.qualitative.Set3
            )
            fig_pie.update_traces(textposition='inside', textinfo='percent+label')
            st.plotly_chart(fig_pie, use_container_width=True)
    
    with col2:
        # Severity distribution
        anomalies = st.session_state.anomaly_results
        severity_counts = {}
        for anomaly in anomalies:
            severity_counts[anomaly.severity] = severity_counts.get(anomaly.severity, 0) + 1
        
        if severity_counts:
            fig_bar = px.bar(
                x=list(severity_counts.keys()),
                y=list(severity_counts.values()),
                title="Anomaly Severity Distribution",
                color=list(severity_counts.keys()),
                color_discrete_map={
                    'critical': '#f44336',
                    'high': '#ff9800', 
                    'medium': '#9c27b0',
                    'low': '#4caf50'
                }
            )
            fig_bar.update_layout(showlegend=False)
            st.plotly_chart(fig_bar, use_container_width=True)

def render_anomaly_details():
    """Render detailed anomaly information"""
    if not st.session_state.anomaly_results:
        return
    
    st.subheader("🔍 Anomaly Details")
    
    # Filter controls
    col1, col2, col3 = st.columns(3)
    
    with col1:
        severity_filter = st.selectbox(
            "Filter by Severity",
            options=['All'] + list(set(a.severity for a in st.session_state.anomaly_results)),
            index=0
        )
    
    with col2:
        type_filter = st.selectbox(
            "Filter by Type",
            options=['All'] + list(set(a.anomaly_type for a in st.session_state.anomaly_results)),
            index=0
        )
    
    with col3:
        fixable_filter = st.selectbox(
            "Filter by Fixability",
            options=['All', 'Auto-Fixable', 'Manual Review'],
            index=0
        )
    
    # Apply filters
    filtered_anomalies = st.session_state.anomaly_results
    
    if severity_filter != 'All':
        filtered_anomalies = [a for a in filtered_anomalies if a.severity == severity_filter]
    
    if type_filter != 'All':
        filtered_anomalies = [a for a in filtered_anomalies if a.anomaly_type == type_filter]
    
    if fixable_filter == 'Auto-Fixable':
        filtered_anomalies = [a for a in filtered_anomalies if a.auto_fixable]
    elif fixable_filter == 'Manual Review':
        filtered_anomalies = [a for a in filtered_anomalies if not a.auto_fixable]
    
    # Display anomalies
    if filtered_anomalies:
        for i, anomaly in enumerate(filtered_anomalies[:50]):  # Limit to 50 for performance
            severity_class = f"anomaly-{anomaly.severity}"
            
            with st.expander(f"🚨 {anomaly.anomaly_type.replace('_', ' ').title()} - Record {anomaly.record_id}"):
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write(f"**Severity:** {anomaly.severity.upper()}")
                    st.write(f"**Confidence:** {anomaly.confidence:.1%}")
                    st.write(f"**Auto-Fixable:** {'✅ Yes' if anomaly.auto_fixable else '❌ No'}")
                
                with col2:
                    st.write(f"**SAP Value:** {anomaly.sap_value}")
                    st.write(f"**Kinaxis Value:** {anomaly.kinaxis_value}")
                
                st.write(f"**Description:** {anomaly.description}")
                st.write(f"**Recommended Action:** {anomaly.recommended_action}")
                
                if anomaly.auto_fixable:
                    if st.button(f"🔧 Auto-Fix", key=f"fix_{i}"):
                        st.success("✅ Anomaly automatically resolved!")
    else:
        st.info("No anomalies match the selected filters.")

def render_data_preview():
    """Render data preview section"""
    if not st.session_state.sap_data and not st.session_state.kinaxis_data:
        return
    
    st.subheader("📊 Data Preview")
    
    tab1, tab2 = st.tabs(["SAP Data", "Kinaxis Data"])
    
    with tab1:
        if st.session_state.sap_data:
            sap_table = st.selectbox(
                "Select SAP Table",
                options=list(st.session_state.sap_data.keys()),
                key="sap_table_select"
            )
            
            if sap_table:
                df = st.session_state.sap_data[sap_table]
                st.write(f"**{sap_table}** - {len(df)} records")
                st.dataframe(df.head(100), use_container_width=True)
    
    with tab2:
        if st.session_state.kinaxis_data:
            kinaxis_table = st.selectbox(
                "Select Kinaxis Table",
                options=list(st.session_state.kinaxis_data.keys()),
                key="kinaxis_table_select"
            )
            
            if kinaxis_table:
                df = st.session_state.kinaxis_data[kinaxis_table]
                st.write(f"**{kinaxis_table}** - {len(df)} records")
                st.dataframe(df.head(100), use_container_width=True)

def render_export_section():
    """Render export and reporting section"""
    if not st.session_state.anomaly_results:
        return
    
    st.subheader("📄 Export & Reports")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("📊 Export Anomalies (CSV)"):
            # Create CSV export
            anomaly_data = []
            for anomaly in st.session_state.anomaly_results:
                anomaly_data.append({
                    'Record ID': anomaly.record_id,
                    'Anomaly Type': anomaly.anomaly_type,
                    'Severity': anomaly.severity,
                    'Description': anomaly.description,
                    'SAP Value': anomaly.sap_value,
                    'Kinaxis Value': anomaly.kinaxis_value,
                    'Confidence': anomaly.confidence,
                    'Auto Fixable': anomaly.auto_fixable,
                    'Recommended Action': anomaly.recommended_action
                })
            
            df_export = pd.DataFrame(anomaly_data)
            csv = df_export.to_csv(index=False)
            
            st.download_button(
                label="Download CSV",
                data=csv,
                file_name=f"pfizer_anomalies_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )
    
    with col2:
        if st.button("📋 Generate Summary Report"):
            st.info("📝 Summary report generation feature coming soon!")
    
    with col3:
        if st.button("📧 Share Results"):
            st.info("📤 Results sharing feature coming soon!")

def main():
    """Main application function"""
    initialize_user_tracking()
    initialize_session_state()
    
    # Show onboarding modal for new users
    if show_onboarding_modal():
        return
    
    render_header()
    render_sidebar()
    
    # Main content area
    st.subheader("📈 Analysis Overview")
    show_tooltip("This overview shows key metrics from your selected agent. Choose an agent from the sidebar to get started!")
    render_overview_metrics()
    
    if st.session_state.detection_summary:
        st.markdown("---")
        st.subheader("📊 Anomaly Analysis")
        render_anomaly_breakdown()
        
        st.markdown("---")
        render_anomaly_details()
        
        st.markdown("---")
        render_export_section()
    
    st.markdown("---")
    render_data_preview()
    
    # Footer
    st.markdown("---")
    st.markdown(
        """
        <div style='text-align: center; color: #666; padding: 2rem;'>
            <p>🏥 <strong>Pfizer Demo</strong> - Agentic AI Anomaly Detection System</p>
            <p>Built by DXC Technology for pharmaceutical supply chain optimization</p>
        </div>
        """,
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()
