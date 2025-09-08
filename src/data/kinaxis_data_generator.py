"""
Kinaxis Master Data Generator for Pfizer Demo
Generates realistic Kinaxis master data with controlled variations from SAP
"""

import pandas as pd
import numpy as np
from faker import Faker
from datetime import datetime, timedelta
import random
from typing import Dict, List, Any, Optional


class KinaxisDataGenerator:
    """
    Generates synthetic Kinaxis master data for pharmaceutical supply chain planning
    """
    
    def __init__(self, seed: int = 42):
        """Initialize the Kinaxis data generator"""
        self.fake = Faker()
        Faker.seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        
        self.pharmaceutical_items = [
            'ASPIRIN-100MG', 'IBUPROFEN-200MG', 'ACETAMINOPHEN-500MG',
            'AMOXICILLIN-250MG', 'METFORMIN-500MG', 'LISINOPRIL-10MG',
            'ATORVASTATIN-20MG', 'OMEPRAZOLE-20MG', 'LEVOTHYROXINE-50MCG',
            'AMLODIPINE-5MG', 'METOPROLOL-50MG', 'HYDROCHLOROTHIAZIDE-25MG'
        ]
        
        self.sites = ['SITE_PF01', 'SITE_PF02', 'SITE_PF03', 'SITE_PF04', 'SITE_PF05']
        self.item_types = ['MAKE', 'BUY', 'TRANSFER']
        self.planning_methods = ['MRP', 'MPS', 'KANBAN', 'MANUAL']
        
    def generate_item_master(self, num_records: int = 1000) -> pd.DataFrame:
        """Generate Kinaxis Item Master data"""
        data = []
        
        for i in range(num_records):
            item_id = f"ITEM{i+1:06d}"
            record = {
                'item_id': f"KIN{random.randint(10000, 99999)}",
                'item_description': self.fake.catch_phrase(),
                'item_type': random.choice(self.item_types),
                'uom': random.choice(['EA', 'KG', 'LB', 'L', 'GAL', 'M', 'FT']),
                'lot_size_minimum': random.randint(1, 100),
                'lot_size_multiple': random.choice([1, 5, 10, 25, 50, 100]),
                'lead_time_days': random.randint(1, 180),
                'safety_stock_days': random.randint(5, 60),
                'reorder_point': random.randint(100, 2000),
                'max_stock_level': random.randint(5000, 50000),
                'abc_class': random.choice(['A', 'B', 'C']),
                'planning_method': random.choice(self.planning_methods),
                'make_buy_indicator': random.choice(['MAKE', 'BUY']),
                'standard_cost': round(random.uniform(1.0, 500.0), 4),
                'currency_code': 'USD',
                'active_flag': random.random() < 0.95,  # 95% chance of True
                'created_date': self.fake.date_between(start_date='-2y', end_date='today'),
                'last_updated': self.fake.date_between(start_date='-1y', end_date='today'),
                'planner_code': f"PLANNER{random.randint(1, 20):02d}",
                'buyer_code': f"BUYER{random.randint(1, 15):02d}",
                'product_family': f"FAMILY_{random.randint(1, 50):02d}",
                'regulatory_status': random.choice(['APPROVED', 'PENDING', 'RESTRICTED']),
                'temperature_controlled': random.random() < 0.3  # 30% chance of True
            }
            data.append(record)
        
        df = pd.DataFrame(data)
        return self._clean_dataframe_for_pyarrow(df)
    
    def generate_supply_network(self, num_records: int = 800) -> pd.DataFrame:
        """Generate Kinaxis Supply Network data"""
        data = []
        
        for i in range(num_records):
            network_id = f"NET{i+1:06d}"
            
            record = {
                'network_id': network_id,
                'source_site': random.choice(self.sites),
                'destination_site': random.choice(self.sites),
                'item_id': f"ITEM{random.randint(1, 1000):06d}",
                'transport_mode': random.choice(['TRUCK', 'AIR', 'OCEAN', 'RAIL']),
                'transit_time_days': random.randint(1, 30),
                'transport_cost_per_unit': round(random.uniform(0.1, 50.0), 3),
                'minimum_shipment_qty': random.randint(1, 1000),
                'maximum_shipment_qty': random.randint(5000, 100000),
                'frequency_days': random.choice([1, 7, 14, 30]),
                'capacity_per_period': random.randint(1000, 50000),
                'supplier_id': f"SUP{random.randint(1, 200):04d}" if random.random() < 0.4 else None,
                'contract_id': f"CONTRACT{random.randint(1, 500):06d}" if random.random() < 0.6 else None,
                'effective_date': self.fake.date_between(start_date='-1y', end_date='today'),
                'expiry_date': self.fake.date_between(start_date='today', end_date='+2y'),
                'priority_rank': random.randint(1, 10),
                'reliability_percent': round(random.uniform(85.0, 99.9), 1),
                'quality_rating': random.choice(['A', 'B', 'C']),
                'carbon_footprint': round(random.uniform(0.1, 10.0), 2),
                'risk_category': random.choice(['LOW', 'MEDIUM', 'HIGH']),
                'backup_source_flag': random.random() < 0.2  # 20% chance of True
            }
            data.append(record)
        
        df = pd.DataFrame(data)
        
        # Convert date columns to datetime for PyArrow compatibility
        date_columns = ['effective_date', 'expiry_date']
        for col in date_columns:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col])
        
        # Ensure string columns are properly typed
        string_columns = ['network_id', 'source_location', 'destination_location', 'supplier_name', 'quality_rating', 'risk_category']
        for col in string_columns:
            if col in df.columns:
                df[col] = df[col].astype(str)
        
        return df
    
    def generate_demand_forecast(self, num_records: int = 1500) -> pd.DataFrame:
        """Generate Kinaxis Demand Forecast data"""
        data = []
        
        # Generate forecasts for next 12 months
        base_date = datetime.now().date()
        
        for i in range(num_records):
            item_id = f"ITEM{random.randint(1, 1000):06d}"
            site_id = random.choice(self.sites)
            
            # Generate 12 months of forecasts for this item/site combination
            for month_offset in range(12):
                forecast_date = base_date + timedelta(days=month_offset * 30)
                
                record = {
                    'forecast_id': f"FC{i+1:06d}_{month_offset+1:02d}",
                    'item_id': item_id,
                    'site_id': site_id,
                    'forecast_date': forecast_date,
                    'forecast_period': f"{forecast_date.year}-{forecast_date.month:02d}",
                    'demand_quantity': random.randint(100, 10000),
                    'forecast_method': random.choice(['STATISTICAL', 'MANUAL', 'COLLABORATIVE', 'ML']),
                    'confidence_level': round(random.uniform(60.0, 95.0), 1),
                    'forecast_bias': round(random.uniform(-20.0, 20.0), 2),
                    'forecast_accuracy': round(random.uniform(70.0, 98.0), 1),
                    'customer_segment': random.choice(['RETAIL', 'HOSPITAL', 'GOVERNMENT', 'EXPORT']),
                    'seasonal_factor': round(random.uniform(0.7, 1.3), 2),
                    'trend_factor': round(random.uniform(0.95, 1.05), 3),
                    'promotion_impact': round(random.uniform(0.0, 50.0), 1) if random.random() < 0.2 else 0.0,
                    'new_product_flag': random.random() < 0.1,  # 10% chance of True
                    'lifecycle_stage': random.choice(['INTRO', 'GROWTH', 'MATURE', 'DECLINE']),
                    'forecast_version': random.choice(['BASELINE', 'CONSENSUS', 'STATISTICAL']),
                    'created_by': f"PLANNER{random.randint(1, 20):02d}",
                    'created_date': self.fake.date_between(start_date='-30d', end_date='today'),
                    'approved_flag': random.random() < 0.8  # 80% chance of True
                }
                data.append(record)
        
        df = pd.DataFrame(data)
        
        # Convert date columns to datetime for PyArrow compatibility
        date_columns = ['created_date']
        for col in date_columns:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col])
        
        # Ensure string columns are properly typed
        string_columns = ['item_id', 'location_code', 'forecast_version', 'lifecycle_stage', 'created_by']
        for col in string_columns:
            if col in df.columns:
                df[col] = df[col].astype(str)
        
        return df
    
    def generate_production_schedule(self, num_records: int = 600) -> pd.DataFrame:
        """Generate Kinaxis Production Schedule data"""
        data = []
        
        for i in range(num_records):
            schedule_id = f"SCHED{i+1:08d}"
            
            record = {
                'schedule_id': schedule_id,
                'item_id': f"ITEM{random.randint(1, 1000):06d}",
                'site_id': random.choice(self.sites),
                'resource_id': f"RES{random.randint(1, 100):04d}",
                'production_date': self.fake.date_between(start_date='-3m', end_date='+6m'),
                'shift': random.choice(['DAY', 'NIGHT', 'WEEKEND']),
                'planned_quantity': random.randint(100, 20000),
                'actual_quantity': random.randint(80, 22000) if random.random() < 0.7 else None,
                'setup_time_hours': round(random.uniform(0.5, 8.0), 1),
                'run_time_hours': round(random.uniform(2.0, 24.0), 1),
                'efficiency_percent': round(random.uniform(75.0, 98.0), 1),
                'yield_percent': round(random.uniform(85.0, 99.5), 1),
                'batch_size': random.randint(500, 5000),
                'campaign_id': f"CAMP{random.randint(1, 200):04d}" if random.random() < 0.6 else None,
                'priority_level': random.randint(1, 5),
                'status': random.choice(['PLANNED', 'RELEASED', 'ACTIVE', 'COMPLETED', 'CANCELLED']),
                'changeover_required': random.random() < 0.3,  # 30% chance of True
                'quality_hold': random.random() < 0.05,  # 5% chance of True
                'maintenance_window': random.random() < 0.1,  # 10% chance of True
                'operator_id': f"OP{random.randint(1, 50):03d}",
                'supervisor_id': f"SUP{random.randint(1, 20):02d}",
                'equipment_utilization': round(random.uniform(60.0, 95.0), 1),
                'energy_consumption': round(random.uniform(100.0, 5000.0), 1),
                'waste_generated': round(random.uniform(0.0, 500.0), 1)
            }
            data.append(record)
        
        df = pd.DataFrame(data)
        
        # Convert date columns to datetime for PyArrow compatibility
        date_columns = ['planned_start_date', 'planned_end_date']
        for col in date_columns:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col])
        
        # Ensure string columns are properly typed
        string_columns = ['schedule_id', 'item_id', 'location_code', 'production_line', 'operator_id', 'supervisor_id']
        for col in string_columns:
            if col in df.columns:
                df[col] = df[col].astype(str)
        
        return df
    
    def create_sap_kinaxis_mismatches(self, kinaxis_data: Dict[str, pd.DataFrame], 
                                    sap_data: Dict[str, pd.DataFrame],
                                    mismatch_rate: float = 0.20) -> Dict[str, pd.DataFrame]:
        """
        Create intentional mismatches between SAP and Kinaxis data
        to simulate real-world synchronization issues
        """
        kinaxis_modified = {}
        
        for table_name, df in kinaxis_data.items():
            modified_df = df.copy()
            num_mismatches = int(len(df) * mismatch_rate)
            mismatch_indices = np.random.choice(df.index, size=num_mismatches, replace=False)
            
            for idx in mismatch_indices:
                mismatch_type = random.choice([
                    'version_difference',
                    'timing_lag',
                    'unit_conversion',
                    'rounding_difference',
                    'status_mismatch'
                ])
                
                if mismatch_type == 'version_difference' and table_name == 'item_master':
                    # Simulate version differences in item descriptions
                    if 'item_description' in modified_df.columns:
                        current_desc = modified_df.loc[idx, 'item_description']
                        modified_df.loc[idx, 'item_description'] = f"{current_desc}_V2"
                
                elif mismatch_type == 'timing_lag':
                    # Simulate timing differences in dates
                    date_columns = [col for col in modified_df.columns if 'date' in col.lower()]
                    if date_columns:
                        col = random.choice(date_columns)
                        if pd.notna(modified_df.loc[idx, col]):
                            # Add random lag of 1-30 days
                            lag_days = random.randint(1, 30)
                            try:
                                current_date = pd.to_datetime(modified_df.loc[idx, col])
                                modified_df.loc[idx, col] = current_date + timedelta(days=lag_days)
                            except:
                                pass
                
                elif mismatch_type == 'unit_conversion':
                    # Simulate unit conversion issues
                    if 'standard_cost' in modified_df.columns:
                        # Convert some costs to different currency or unit
                        current_cost = modified_df.loc[idx, 'standard_cost']
                        if pd.notna(current_cost):
                            modified_df.loc[idx, 'standard_cost'] = round(float(current_cost) * 1.2, 4)  # 20% difference
                
                elif mismatch_type == 'rounding_difference':
                    # Simulate rounding differences
                    numeric_columns = modified_df.select_dtypes(include=[np.number]).columns
                    if len(numeric_columns) > 0:
                        col = random.choice(numeric_columns)
                        current_value = modified_df.loc[idx, col]
                        if pd.notna(current_value) and current_value != 0:
                            # Add small rounding difference
                            modified_df.loc[idx, col] = round(float(current_value) * random.uniform(1.001, 1.005), 6)
                
                elif mismatch_type == 'status_mismatch':
                    # Simulate status mismatches
                    if 'active_flag' in modified_df.columns:
                        modified_df.loc[idx, 'active_flag'] = not modified_df.loc[idx, 'active_flag']
            
            kinaxis_modified[table_name] = modified_df
        
        return kinaxis_modified
    
    def _clean_dataframe_for_pyarrow(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean DataFrame to ensure PyArrow compatibility"""
        # Convert date columns to datetime
        date_columns = ['created_date', 'last_updated', 'effective_date', 'expiry_date',
                       'planned_start_date', 'planned_end_date']
        for col in date_columns:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], errors='coerce')
        
        # Ensure all object columns are strings
        for col in df.columns:
            if df[col].dtype == 'object':
                # Check if it's a datetime column that failed conversion
                if col not in date_columns:
                    df[col] = df[col].astype(str)
        
        return df
    
    def generate_full_dataset(self, 
                            item_records: int = 1000,
                            network_records: int = 800,
                            forecast_records: int = 1500,
                            schedule_records: int = 600,
                            sap_data: Optional[Dict[str, pd.DataFrame]] = None,
                            mismatch_rate: float = 0.20) -> Dict[str, pd.DataFrame]:
        """
        Generate complete Kinaxis dataset
        
        Args:
            sap_data: Optional SAP data to create realistic mismatches
            mismatch_rate: Rate of intentional mismatches with SAP data
        """
        datasets = {}
        
        # Generate base Kinaxis data
        datasets['item_master'] = self.generate_item_master(item_records)
        datasets['supply_network'] = self.generate_supply_network(network_records)
        datasets['demand_forecast'] = self.generate_demand_forecast(forecast_records)
        datasets['production_schedule'] = self.generate_production_schedule(schedule_records)
        
        # Create mismatches with SAP data if provided
        if sap_data:
            datasets = self.create_sap_kinaxis_mismatches(datasets, sap_data, mismatch_rate)
        
        return datasets
    
    def save_datasets(self, datasets: Dict[str, pd.DataFrame], output_dir: str = "data/synthetic/kinaxis"):
        """Save generated datasets to files"""
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        for table_name, df in datasets.items():
            # Save as CSV
            csv_path = os.path.join(output_dir, f"{table_name}.csv")
            df.to_csv(csv_path, index=False)
            
            # Save as Excel
            excel_path = os.path.join(output_dir, f"{table_name}.xlsx")
            df.to_excel(excel_path, index=False)
            
            print(f"Saved {table_name}: {len(df)} records to {output_dir}")


# Example usage
if __name__ == "__main__":
    generator = KinaxisDataGenerator()
    datasets = generator.generate_full_dataset()
    generator.save_datasets(datasets)
    
    print("Kinaxis synthetic data generation completed!")
    for table_name, df in datasets.items():
        print(f"{table_name}: {len(df)} records")
