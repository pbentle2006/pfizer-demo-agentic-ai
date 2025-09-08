#!/usr/bin/env python3
"""
Data Generation Script for Pfizer Demo
Generates synthetic SAP and Kinaxis data for demonstration
"""

import sys
import os
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from data.sap_data_generator import SAPDataGenerator
from data.kinaxis_data_generator import KinaxisDataGenerator


def main():
    """Generate synthetic data for the demo"""
    print("🏥 Pfizer Demo - Synthetic Data Generation")
    print("=" * 50)
    
    # Create output directories
    output_dir = Path(__file__).parent.parent / 'data' / 'synthetic'
    sap_dir = output_dir / 'sap'
    kinaxis_dir = output_dir / 'kinaxis'
    
    sap_dir.mkdir(parents=True, exist_ok=True)
    kinaxis_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate SAP data
    print("\n📊 Generating SAP S/4 HANA data...")
    sap_generator = SAPDataGenerator()
    sap_datasets = sap_generator.generate_full_dataset(
        material_records=1000,
        production_records=500,
        inventory_records=2000,
        bom_records=800,
        anomaly_rate=0.15
    )
    sap_generator.save_datasets(sap_datasets, str(sap_dir))
    
    # Generate Kinaxis data
    print("\n📈 Generating Kinaxis master data...")
    kinaxis_generator = KinaxisDataGenerator()
    kinaxis_datasets = kinaxis_generator.generate_full_dataset(
        item_records=1000,
        network_records=800,
        forecast_records=1500,
        schedule_records=600,
        sap_data=sap_datasets,
        mismatch_rate=0.20
    )
    kinaxis_generator.save_datasets(kinaxis_datasets, str(kinaxis_dir))
    
    print("\n✅ Data generation completed!")
    print(f"📁 SAP data saved to: {sap_dir}")
    print(f"📁 Kinaxis data saved to: {kinaxis_dir}")
    
    # Summary
    total_sap_records = sum(len(df) for df in sap_datasets.values())
    total_kinaxis_records = sum(len(df) for df in kinaxis_datasets.values())
    
    print(f"\n📈 Summary:")
    print(f"   SAP records: {total_sap_records:,}")
    print(f"   Kinaxis records: {total_kinaxis_records:,}")
    print(f"   Total records: {total_sap_records + total_kinaxis_records:,}")
    print(f"   Estimated anomalies: ~{int((total_sap_records + total_kinaxis_records) * 0.175):,}")


if __name__ == "__main__":
    main()
