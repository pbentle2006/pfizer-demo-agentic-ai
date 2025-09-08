"""
SAP S/4 HANA Synthetic Data Generator for Pfizer Demo
Generates realistic SAP staging data with controlled anomalies
"""

import pandas as pd
import numpy as np
from faker import Faker
from datetime import datetime, timedelta
import random
from typing import Dict, List, Any, Optional


class SAPDataGenerator:
    """
    Generates synthetic SAP S/4 HANA staging data for pharmaceutical manufacturing
    """
    
    def __init__(self, seed: int = 42):
        """Initialize the SAP data generator"""
        self.fake = Faker()
        Faker.seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        
        self.pharmaceutical_materials = [
            'ASPIRIN-100MG', 'IBUPROFEN-200MG', 'ACETAMINOPHEN-500MG',
            'AMOXICILLIN-250MG', 'METFORMIN-500MG', 'LISINOPRIL-10MG',
            'ATORVASTATIN-20MG', 'OMEPRAZOLE-20MG', 'LEVOTHYROXINE-50MCG',
            'AMLODIPINE-5MG', 'METOPROLOL-50MG', 'HYDROCHLOROTHIAZIDE-25MG'
        ]
        
        self.plant_codes = ['PF01', 'PF02', 'PF03', 'PF04', 'PF05']
        self.production_versions = ['001', '002', '003', '004']
        self.bom_versions = ['001', '002', '003', '004']
        
    def generate_material_master(self, num_records: int = 1000) -> pd.DataFrame:
        """Generate SAP Material Master (MARA/MARC) data"""
        data = []
        
        for i in range(num_records):
            material_number = f"MAT{i+1:06d}"
            material_desc = random.choice(self.pharmaceutical_materials)
            
            record = {
                'material_number': material_number,
                'material_description': material_desc,
                'material_type': random.choice(['FERT', 'HALB', 'ROH']),  # Finished, Semi-finished, Raw
                'plant_code': random.choice(self.plant_codes),
                'base_unit': random.choice(['EA', 'KG', 'L', 'M']),
                'gross_weight': round(random.uniform(0.1, 50.0), 3),
                'net_weight': round(random.uniform(0.05, 45.0), 3),
                'weight_unit': 'KG',
                'volume': round(random.uniform(0.001, 5.0), 4),
                'volume_unit': 'L',
                'created_date': self.fake.date_between(start_date='-2y', end_date='today'),
                'created_by': f"USER{random.randint(1, 50):03d}",
                'last_changed': self.fake.date_between(start_date='-1y', end_date='today'),
                'production_version': random.choice(self.production_versions),
                'bom_version': random.choice(self.bom_versions),
                'procurement_type': random.choice(['F', 'E']),  # In-house production, External procurement
                'mrp_type': random.choice(['PD', 'VV', 'VM']),
                'lot_size': random.randint(100, 10000),
                'safety_stock': random.randint(50, 1000),
                'reorder_point': random.randint(100, 2000),
                'standard_cost': round(random.uniform(1.0, 500.0), 2),
                'price_unit': 1,
                'currency': 'USD'
            }
            data.append(record)
        
        df = pd.DataFrame(data)
        return self._clean_dataframe_for_pyarrow(df)
    
    def generate_production_orders(self, num_records: int = 500) -> pd.DataFrame:
        """Generate SAP Production Orders (AUFK/AFKO) data"""
        data = []
        
        for i in range(num_records):
            order_number = f"PO{i+1:08d}"
            material_number = f"MAT{random.randint(1, 1000):06d}"
            
            record = {
                'production_order': order_number,
                'material_number': material_number,
                'plant_code': random.choice(self.plant_codes),
                'order_type': random.choice(['PP01', 'PP02', 'PP03']),
                'order_status': random.choice(['REL', 'CNF', 'TECO', 'DLT']),
                'production_qty': random.randint(100, 50000),
                'confirmed_qty': random.randint(50, 45000),
                'scrap_qty': random.randint(0, 500),
                'unit_of_measure': random.choice(['EA', 'KG', 'L']),
                'start_date': self.fake.date_between(start_date='-6m', end_date='+3m'),
                'finish_date': self.fake.date_between(start_date='-3m', end_date='+6m'),
                'actual_start': self.fake.date_between(start_date='-6m', end_date='today'),
                'actual_finish': self.fake.date_between(start_date='-3m', end_date='today'),
                'production_version': random.choice(self.production_versions),
                'bom_version': random.choice(self.bom_versions),
                'routing_version': random.choice(['001', '002', '003']),
                'work_center': f"WC{random.randint(1, 20):03d}",
                'planned_cost': round(random.uniform(1000.0, 100000.0), 2),
                'actual_cost': round(random.uniform(900.0, 110000.0), 2),
                'variance': round(random.uniform(-5000.0, 5000.0), 2),
                'created_by': f"USER{random.randint(1, 50):03d}",
                'created_date': self.fake.date_between(start_date='-1y', end_date='today'),
                'priority': random.choice(['1', '2', '3', '4', '5'])
            }
            data.append(record)
        
        df = pd.DataFrame(data)
        return self._clean_dataframe_for_pyarrow(df)
    
    def generate_inventory_movements(self, num_records: int = 2000) -> pd.DataFrame:
        """Generate SAP Inventory Movements (MSEG) data"""
        data = []
        
        movement_types = ['101', '102', '261', '262', '311', '312', '501', '502']
        
        for i in range(num_records):
            document_number = f"DOC{i+1:010d}"
            material_number = f"MAT{random.randint(1, 1000):06d}"
            
            record = {
                'document_number': document_number,
                'document_item': random.randint(1, 10),
                'material_number': material_number,
                'plant_code': random.choice(self.plant_codes),
                'storage_location': f"SL{random.randint(1, 10):02d}",
                'movement_type': random.choice(movement_types),
                'movement_indicator': random.choice(['B', 'H']),
                'quantity': random.randint(1, 10000),
                'unit_of_measure': random.choice(['EA', 'KG', 'L']),
                'amount': round(random.uniform(10.0, 50000.0), 2),
                'currency': 'USD',
                'posting_date': self.fake.date_between(start_date='-1y', end_date='today'),
                'document_date': self.fake.date_between(start_date='-1y', end_date='today'),
                'reference_document': f"REF{random.randint(1, 100000):08d}",
                'batch_number': f"BATCH{random.randint(1, 10000):06d}",
                'vendor_number': f"V{random.randint(1, 500):06d}" if random.random() < 0.3 else None,
                'customer_number': f"C{random.randint(1, 1000):06d}" if random.random() < 0.2 else None,
                'production_order': f"PO{random.randint(1, 500):08d}" if random.random() < 0.4 else None,
                'cost_center': f"CC{random.randint(1, 100):04d}",
                'profit_center': f"PC{random.randint(1, 50):03d}",
                'reason_code': random.choice(['01', '02', '03', '04', '05']) if random.random() < 0.3 else None,
                'user_name': f"USER{random.randint(1, 50):03d}",
                'time_stamp': self.fake.date_time_between(start_date='-1y', end_date='now')
            }
            data.append(record)
        
        df = pd.DataFrame(data)
        return self._clean_dataframe_for_pyarrow(df)
    
    def generate_bom_data(self, num_records: int = 800) -> pd.DataFrame:
        """Generate SAP Bill of Materials (STKO/STPO) data"""
        data = []
        
        for i in range(num_records):
            bom_number = f"BOM{i+1:08d}"
            parent_material = f"MAT{random.randint(1, 1000):06d}"
            
            # Generate components for this BOM
            num_components = random.randint(2, 8)
            for comp in range(num_components):
                component_material = f"MAT{random.randint(1, 1000):06d}"
                
                record = {
                    'bom_number': bom_number,
                    'bom_version': random.choice(self.bom_versions),
                    'parent_material': parent_material,
                    'component_material': component_material,
                    'item_number': f"{(comp + 1) * 10:04d}",
                    'component_quantity': round(random.uniform(0.1, 100.0), 3),
                    'component_unit': random.choice(['EA', 'KG', 'L', 'M']),
                    'component_scrap': round(random.uniform(0.0, 5.0), 2),
                    'operation_number': f"{random.randint(1, 10) * 10:04d}",
                    'item_category': random.choice(['L', 'N', 'R']),  # Stock, Non-stock, Variable-size
                    'procurement_type': random.choice(['F', 'E']),
                    'special_procurement': random.choice(['', '10', '20', '30']),
                    'valid_from': self.fake.date_between(start_date='-2y', end_date='today'),
                    'valid_to': self.fake.date_between(start_date='today', end_date='+2y'),
                    'plant_code': random.choice(self.plant_codes),
                    'created_by': f"USER{random.randint(1, 50):03d}",
                    'created_date': self.fake.date_between(start_date='-2y', end_date='today'),
                    'last_changed': self.fake.date_between(start_date='-1y', end_date='today'),
                    'change_number': f"ECN{random.randint(1, 10000):06d}" if random.random() < 0.3 else None,
                    'cost_element': f"CE{random.randint(1, 100):04d}",
                    'component_cost': round(random.uniform(0.1, 1000.0), 2)
                }
                data.append(record)
        
        df = pd.DataFrame(data)
        return self._clean_dataframe_for_pyarrow(df)
    
    def _is_numeric_string(self, value: str) -> bool:
        """Check if a string represents a valid numeric value"""
        try:
            float(value)
            return True
        except (ValueError, TypeError):
            return False
    
    def _clean_dataframe_for_pyarrow(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean DataFrame to ensure PyArrow compatibility"""
        # Convert date columns to datetime
        date_columns = ['created_date', 'last_changed', 'start_date', 'finish_date', 
                       'actual_start', 'actual_finish', 'posting_date', 'document_date',
                       'valid_from', 'valid_to']
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
    
    def inject_anomalies(self, df: pd.DataFrame, anomaly_rate: float) -> pd.DataFrame:
        """
        Inject controlled anomalies into the dataset
        
        Args:
            df: DataFrame to inject anomalies into
            anomaly_rate: Percentage of records to make anomalous (0.0 to 1.0)
        """
        anomaly_df = df.copy()
        num_anomalies = int(len(df) * anomaly_rate)
        anomaly_indices = np.random.choice(df.index, size=num_anomalies, replace=False)
        
        for idx in anomaly_indices:
            anomaly_type = random.choice([
                'negative_quantities',
                'excessive_precision', 
                'version_mismatch',
                'missing_values',
                'invalid_dates',
                'data_type_mismatch'
            ])
            
            if anomaly_type == 'negative_quantities':
                # Make quantities negative
                qty_columns = [col for col in df.columns if 'qty' in col.lower() or 'quantity' in col.lower()]
                if qty_columns:
                    col = random.choice(qty_columns)
                    current_value = anomaly_df.loc[idx, col]
                    try:
                        # Only convert to numeric if it's actually a numeric value
                        if isinstance(current_value, (int, float)):
                            anomaly_df.loc[idx, col] = -abs(current_value)
                        elif isinstance(current_value, str):
                            try:
                                numeric_value = float(current_value)
                                anomaly_df.loc[idx, col] = -abs(numeric_value)
                            except (ValueError, TypeError):
                                # Skip non-numeric strings
                                pass
                    except (ValueError, TypeError):
                        # Skip if value cannot be converted to numeric
                        pass
            
            elif anomaly_type == 'excessive_precision':
                # Add excessive decimal precision
                cost_columns = [col for col in df.columns if 'cost' in col.lower() or 'amount' in col.lower()]
                if cost_columns:
                    col = random.choice(cost_columns)
                    if pd.notna(anomaly_df.loc[idx, col]):
                        try:
                            # Only convert to float if it's actually a numeric value
                            current_value = anomaly_df.loc[idx, col]
                            if isinstance(current_value, (int, float)):
                                anomaly_df.loc[idx, col] = round(current_value * random.uniform(1.123456789, 1.987654321), 8)
                            elif isinstance(current_value, str):
                                try:
                                    numeric_value = float(current_value)
                                    anomaly_df.loc[idx, col] = round(numeric_value * random.uniform(1.123456789, 1.987654321), 8)
                                except (ValueError, TypeError):
                                    # Skip non-numeric strings
                                    pass
                        except (ValueError, TypeError):
                            # Skip if value cannot be converted to float
                            pass
            
            elif anomaly_type == 'version_mismatch':
                # Create version mismatches
                if 'production_version' in df.columns:
                    anomaly_df.loc[idx, 'production_version'] = 'INVALID'
                if 'bom_version' in df.columns:
                    anomaly_df.loc[idx, 'bom_version'] = 'MISMATCH'
            
            elif anomaly_type == 'missing_values':
                # Create missing values in critical fields
                critical_fields = ['material_number', 'plant_code']
                available_fields = [field for field in critical_fields if field in df.columns]
                if available_fields:
                    col = random.choice(available_fields)
                    anomaly_df.loc[idx, col] = None
            
            elif anomaly_type == 'invalid_dates':
                # Create invalid dates
                date_columns = [col for col in df.columns if 'date' in col.lower()]
                if date_columns:
                    col = random.choice(date_columns)
                    anomaly_df.loc[idx, col] = 'INVALID-DATE'
            
            elif anomaly_type == 'data_type_mismatch':
                # Create data type mismatches
                if 'material_number' in df.columns:
                    anomaly_df.loc[idx, 'material_number'] = 12345  # Should be string
                qty_columns = [col for col in df.columns if 'qty' in col.lower() or 'quantity' in col.lower()]
                if qty_columns:
                    col = random.choice(qty_columns)
                    anomaly_df.loc[idx, col] = 'NOT_A_NUMBER'
        
        return anomaly_df
    
    def generate_full_dataset(self, 
                            material_records: int = 1000,
                            production_records: int = 500,
                            inventory_records: int = 2000,
                            bom_records: int = 800,
                            anomaly_rate: float = 0.15) -> Dict[str, pd.DataFrame]:
        """
        Generate complete SAP dataset with all tables
        
        Returns:
            Dictionary containing all generated tables
        """
        datasets = {}
        
        # Generate base data
        datasets['material_master'] = self.generate_material_master(material_records)
        datasets['production_orders'] = self.generate_production_orders(production_records)
        datasets['inventory_movements'] = self.generate_inventory_movements(inventory_records)
        datasets['bom_data'] = self.generate_bom_data(bom_records)
        
        # Inject anomalies into each dataset
        for table_name, df in datasets.items():
            datasets[table_name] = self.inject_anomalies(df, anomaly_rate)
        
        return datasets
    
    def save_datasets(self, datasets: Dict[str, pd.DataFrame], output_dir: str = "data/synthetic/sap"):
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
    generator = SAPDataGenerator()
    datasets = generator.generate_full_dataset()
    generator.save_datasets(datasets)
    
    print("SAP synthetic data generation completed!")
    for table_name, df in datasets.items():
        print(f"{table_name}: {len(df)} records")
