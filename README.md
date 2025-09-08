# Pfizer Demo - Agentic AI for Supply Chain

This project demonstrates DXC's agentic AI capabilities for pharmaceutical supply chain data quality management, specifically focusing on SAP S/4 HANA and Kinaxis integration scenarios.

## 🚀 Live Demo

**Public Demo URL:** [Coming Soon - Deploy to Streamlit Community Cloud]

## 📱 Quick Start

1. **Local Development:**
   ```bash
   pip install -r requirements.txt
   streamlit run src/dashboard/app.py
   ```

2. **Public Access:**
   Visit the live demo URL above to interact with the application without any setup.

## Key Features
- **Anomaly Detection Agent**: Compares SAP staging data with Kinaxis master data
- **Rule-Based Validation**: Validates data integrity before batch updates
- **Interactive Dashboard**: Visual anomaly detection and resolution interface
- **LLM Assistant**: Embedded chat for anomaly queries and resolution guidance
- **Report Generation**: Exportable reconciliation reports (PDF/CSV)

## Architecture
```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   SAP S/4 HANA  │    │    Kinaxis      │    │   Dashboard     │
│ Synthetic Data  │◄──►│  Master Data    │◄──►│      UI         │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         └───────────────────────┼───────────────────────┘
                                 │
                    ┌─────────────────┐
                    │ Anomaly Agent   │
                    │ + Rule Engine   │
                    └─────────────────┘
                                 │
         ┌───────────────────────┼───────────────────────┐
         │                       │                       │
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│ LLM Assistant   │    │ Report Generator│    │   Alerts &      │
│ Chat Interface  │    │   (PDF/CSV)     │    │ Auto-Remediate  │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

## Anomaly Detection Capabilities
- **Negative Quantities**: Detect invalid negative values in inventory/production data
- **Excessive Decimal Precision**: Flag inappropriate precision levels
- **Mismatched Production Versions**: Identify version inconsistencies
- **Missing/Invalid Values**: Detect null, empty, or malformed data
- **Data Type Mismatches**: Validate field types and formats

## Quick Start
1. Install dependencies: `pip install -r requirements.txt`
2. Generate synthetic data: `python scripts/generate_data.py`
3. Start the dashboard: `streamlit run src/dashboard/app.py`
4. Upload data files and run anomaly detection

## Demo Workflow
1. **Data Upload**: Upload synthetic SAP and Kinaxis files
2. **Anomaly Detection**: Agent automatically compares and flags issues
3. **Interactive Review**: Use dashboard to explore anomalies
4. **LLM Assistance**: Chat with AI for resolution guidance
5. **Report Generation**: Export reconciliation reports
6. **Auto-Remediation**: Apply automated fixes where possible

## Project Structure
```
pfizer-demo/
├── src/
│   ├── agents/           # Anomaly detection agent
│   ├── data/            # Data generators and validators
│   ├── dashboard/       # Streamlit UI
│   ├── llm/            # LLM assistant integration
│   ├── reports/        # Report generation
│   └── rules/          # Validation rule engine
├── data/
│   ├── synthetic/      # Generated test data
│   ├── uploads/        # User uploaded files
│   └── processed/      # Processed results
├── docs/               # Documentation
├── scripts/            # Utility scripts
└── tests/              # Test suite
```

## Technology Stack
- **Backend**: Python, FastAPI
- **Frontend**: Streamlit
- **AI/ML**: LangChain, OpenAI
- **Data**: Pandas, NumPy
- **Reports**: ReportLab (PDF), CSV
- **Visualization**: Plotly, Matplotlib

## Demo Scenarios
1. **High Anomaly Rate**: 15% of records have issues
2. **Mixed Anomaly Types**: Various error categories
3. **Auto-Resolution**: 70% of issues auto-fixed
4. **Manual Review**: 30% require human intervention

Built for DXC to demonstrate proactive anomaly detection capabilities for Pfizer.
