#!/usr/bin/env python3
"""
Demo Runner Script for Pfizer Demo
Complete demo workflow execution
"""

import sys
import os
import asyncio
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from agents.anomaly_detection_agent import AnomalyDetectionAgent
from data.sap_data_generator import SAPDataGenerator
from data.kinaxis_data_generator import KinaxisDataGenerator


async def run_demo():
    """Run the complete demo workflow"""
    print("🏥 Pfizer Demo - Complete Workflow")
    print("=" * 50)
    
    # Step 1: Generate synthetic data
    print("\n1️⃣ Generating synthetic data...")
    sap_generator = SAPDataGenerator()
    sap_datasets = sap_generator.generate_full_dataset(
        material_records=500,
        production_records=200,
        inventory_records=800,
        bom_records=300,
        anomaly_rate=0.15
    )
    
    kinaxis_generator = KinaxisDataGenerator()
    kinaxis_datasets = kinaxis_generator.generate_full_dataset(
        item_records=500,
        network_records=300,
        forecast_records=600,
        schedule_records=250,
        sap_data=sap_datasets,
        mismatch_rate=0.20
    )
    
    print(f"   ✅ Generated {sum(len(df) for df in sap_datasets.values()):,} SAP records")
    print(f"   ✅ Generated {sum(len(df) for df in kinaxis_datasets.values()):,} Kinaxis records")
    
    # Step 2: Initialize anomaly detection agent
    print("\n2️⃣ Initializing AI agent...")
    agent = AnomalyDetectionAgent()
    print("   ✅ Anomaly detection agent ready")
    
    # Step 3: Run anomaly detection
    print("\n3️⃣ Running anomaly detection...")
    
    # Use material master data for comparison
    sap_data = sap_datasets['material_master']
    kinaxis_data = kinaxis_datasets['item_master']
    
    anomalies, summary = agent.detect_anomalies(sap_data, kinaxis_data)
    
    print(f"   ✅ Analysis completed in {summary.processing_time}s")
    print(f"   📊 Analyzed {summary.total_records:,} records")
    print(f"   🚨 Found {summary.anomalies_found:,} anomalies ({summary.anomaly_rate}%)")
    
    # Step 4: Show anomaly breakdown
    print("\n4️⃣ Anomaly breakdown:")
    for anomaly_type, count in summary.anomaly_breakdown.items():
        print(f"   • {anomaly_type.replace('_', ' ').title()}: {count}")
    
    print(f"\n   🔧 Auto-fixable: {summary.auto_fixable_count}")
    print(f"   👥 Manual review: {summary.manual_review_count}")
    
    # Step 5: Auto-remediation
    print("\n5️⃣ Running auto-remediation...")
    merged_data = agent._merge_datasets(sap_data, kinaxis_data)
    corrected_data, remediation_log = agent.auto_remediate(anomalies, merged_data)
    
    print(f"   ✅ Applied {len(remediation_log)} automatic fixes")
    for log_entry in remediation_log[:5]:  # Show first 5
        print(f"   • {log_entry}")
    if len(remediation_log) > 5:
        print(f"   • ... and {len(remediation_log) - 5} more fixes")
    
    # Step 6: Demo summary
    print("\n" + "=" * 50)
    print("🎉 Demo Results Summary:")
    print(f"   📊 Total Records Processed: {summary.total_records:,}")
    print(f"   🚨 Anomalies Detected: {summary.anomalies_found:,} ({summary.anomaly_rate}%)")
    print(f"   🔧 Auto-Fixed: {summary.auto_fixable_count:,}")
    print(f"   👥 Requiring Review: {summary.manual_review_count:,}")
    print(f"   ⏱️ Processing Time: {summary.processing_time}s")
    print(f"   🎯 Success Rate: {((summary.total_records - summary.anomalies_found) / summary.total_records * 100):.1f}%")
    
    print("\n🚀 Demo completed successfully!")
    print("\nNext steps:")
    print("   • Start dashboard: streamlit run src/dashboard/app.py")
    print("   • Generate full dataset: python scripts/generate_data.py")
    print("   • View project documentation: cat README.md")


def main():
    """Main function"""
    try:
        asyncio.run(run_demo())
    except KeyboardInterrupt:
        print("\n⏹️ Demo interrupted by user")
    except Exception as e:
        print(f"\n❌ Demo failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
