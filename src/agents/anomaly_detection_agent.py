"""
Anomaly Detection Agent for Pfizer Demo
Compares synthetic SAP S/4 HANA staging data with Kinaxis master data
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Tuple, Optional
from dataclasses import dataclass
from datetime import datetime
import logging
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler


@dataclass
class AnomalyResult:
    """Result of anomaly detection"""
    record_id: str
    anomaly_type: str
    severity: str  # 'critical', 'high', 'medium', 'low'
    description: str
    sap_value: Any
    kinaxis_value: Any
    confidence: float
    auto_fixable: bool
    recommended_action: str


@dataclass
class DetectionSummary:
    """Summary of anomaly detection results"""
    total_records: int
    anomalies_found: int
    anomaly_rate: float
    auto_fixable_count: int
    manual_review_count: int
    anomaly_breakdown: Dict[str, int]
    processing_time: float


class AnomalyDetectionAgent:
    """
    Agent that detects anomalies between SAP staging data and Kinaxis master data
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.rules = self._initialize_rules()
        self.ml_models = {}
        self._setup_ml_models()
    
    def _initialize_rules(self) -> Dict[str, Dict]:
        """Initialize validation rules for anomaly detection"""
        return {
            'negative_quantities': {
                'fields': ['quantity', 'stock_level', 'production_qty'],
                'condition': lambda x: x < 0,
                'severity': 'critical',
                'auto_fixable': True,
                'fix_action': 'Set to absolute value or zero'
            },
            'excessive_precision': {
                'fields': ['unit_cost', 'total_value', 'weight'],
                'condition': lambda x: len(str(x).split('.')[-1]) > 4 if '.' in str(x) else False,
                'severity': 'medium',
                'auto_fixable': True,
                'fix_action': 'Round to 4 decimal places'
            },
            'version_mismatch': {
                'fields': ['production_version', 'bom_version'],
                'condition': 'compare_versions',
                'severity': 'high',
                'auto_fixable': False,
                'fix_action': 'Manual review required - sync versions'
            },
            'missing_values': {
                'fields': ['material_number', 'plant_code', 'item_id'],
                'condition': lambda x: pd.isna(x) or x == '' or x is None,
                'severity': 'critical',
                'auto_fixable': False,
                'fix_action': 'Manual data entry required'
            },
            'invalid_dates': {
                'fields': ['production_date', 'delivery_date', 'created_date'],
                'condition': 'validate_date',
                'severity': 'high',
                'auto_fixable': True,
                'fix_action': 'Set to current date or system default'
            },
            'data_type_mismatch': {
                'fields': ['material_number', 'quantity', 'unit_cost'],
                'condition': 'validate_data_type',
                'severity': 'high',
                'auto_fixable': True,
                'fix_action': 'Convert to correct data type'
            }
        }
    
    def _setup_ml_models(self):
        """Setup machine learning models for anomaly detection"""
        self.ml_models['isolation_forest'] = IsolationForest(
            contamination=0.1,
            random_state=42,
            n_estimators=100
        )
        self.scaler = StandardScaler()
    
    def detect_anomalies(self, sap_data: pd.DataFrame, kinaxis_data: pd.DataFrame) -> Tuple[List[AnomalyResult], DetectionSummary]:
        """
        Main method to detect anomalies between SAP and Kinaxis data
        
        Args:
            sap_data: SAP S/4 HANA staging data
            kinaxis_data: Kinaxis master data
            
        Returns:
            Tuple of (anomaly_results, summary)
        """
        start_time = datetime.now()
        anomalies = []
        
        # Merge datasets on common keys
        merged_data = self._merge_datasets(sap_data, kinaxis_data)
        
        # Rule-based anomaly detection
        rule_anomalies = self._detect_rule_based_anomalies(merged_data)
        anomalies.extend(rule_anomalies)
        
        # Statistical anomaly detection
        statistical_anomalies = self._detect_statistical_anomalies(merged_data)
        anomalies.extend(statistical_anomalies)
        
        # ML-based anomaly detection
        ml_anomalies = self._detect_ml_anomalies(merged_data)
        anomalies.extend(ml_anomalies)
        
        # Create summary
        processing_time = (datetime.now() - start_time).total_seconds()
        summary = self._create_summary(merged_data, anomalies, processing_time)
        
        return anomalies, summary
    
    def _merge_datasets(self, sap_data: pd.DataFrame, kinaxis_data: pd.DataFrame) -> pd.DataFrame:
        """Merge SAP and Kinaxis datasets on common keys"""
        # Common merge keys
        merge_keys = ['material_number', 'plant_code']
        
        # Ensure merge keys exist in both datasets
        sap_keys = [key for key in merge_keys if key in sap_data.columns]
        kinaxis_keys = [key for key in merge_keys if key in kinaxis_data.columns]
        
        if not sap_keys or not kinaxis_keys:
            # If no common keys, create a cross-join for comparison
            sap_data['_merge_key'] = 1
            kinaxis_data['_merge_key'] = 1
            merged = pd.merge(sap_data, kinaxis_data, on='_merge_key', suffixes=('_sap', '_kinaxis'))
            merged = merged.drop('_merge_key', axis=1)
        else:
            # Merge on common keys
            common_keys = list(set(sap_keys) & set(kinaxis_keys))
            merged = pd.merge(sap_data, kinaxis_data, on=common_keys, how='outer', suffixes=('_sap', '_kinaxis'))
        
        return merged
    
    def _detect_rule_based_anomalies(self, data: pd.DataFrame) -> List[AnomalyResult]:
        """Detect anomalies using predefined rules"""
        anomalies = []
        
        for rule_name, rule_config in self.rules.items():
            fields = rule_config['fields']
            condition = rule_config['condition']
            
            for field in fields:
                # Check both SAP and Kinaxis versions of the field
                sap_field = f"{field}_sap"
                kinaxis_field = f"{field}_kinaxis"
                
                for field_name in [sap_field, kinaxis_field]:
                    if field_name in data.columns:
                        anomaly_records = self._apply_rule(data, field_name, condition, rule_name, rule_config)
                        anomalies.extend(anomaly_records)
        
        return anomalies
    
    def _apply_rule(self, data: pd.DataFrame, field_name: str, condition, rule_name: str, rule_config: Dict) -> List[AnomalyResult]:
        """Apply a specific rule to detect anomalies"""
        anomalies = []
        
        for idx, row in data.iterrows():
            value = row.get(field_name)
            is_anomaly = False
            
            if callable(condition):
                try:
                    is_anomaly = condition(value)
                except:
                    continue
            elif condition == 'compare_versions':
                is_anomaly = self._check_version_mismatch(row, field_name)
            elif condition == 'validate_date':
                is_anomaly = self._check_invalid_date(value)
            elif condition == 'validate_data_type':
                is_anomaly = self._check_data_type_mismatch(value, field_name)
            
            if is_anomaly:
                anomaly = AnomalyResult(
                    record_id=str(idx),
                    anomaly_type=rule_name,
                    severity=rule_config['severity'],
                    description=f"{rule_name.replace('_', ' ').title()} detected in {field_name}",
                    sap_value=row.get(field_name.replace('_kinaxis', '_sap'), value),
                    kinaxis_value=row.get(field_name.replace('_sap', '_kinaxis'), value),
                    confidence=0.95,  # High confidence for rule-based detection
                    auto_fixable=rule_config['auto_fixable'],
                    recommended_action=rule_config['fix_action']
                )
                anomalies.append(anomaly)
        
        return anomalies
    
    def _check_version_mismatch(self, row: pd.Series, field_name: str) -> bool:
        """Check for version mismatches between SAP and Kinaxis"""
        base_field = field_name.replace('_sap', '').replace('_kinaxis', '')
        sap_version = row.get(f"{base_field}_sap")
        kinaxis_version = row.get(f"{base_field}_kinaxis")
        
        if pd.isna(sap_version) or pd.isna(kinaxis_version):
            return False
        
        return str(sap_version) != str(kinaxis_version)
    
    def _check_invalid_date(self, value) -> bool:
        """Check if date value is invalid"""
        if pd.isna(value):
            return True
        
        try:
            pd.to_datetime(value)
            return False
        except:
            return True
    
    def _check_data_type_mismatch(self, value, field_name: str) -> bool:
        """Check for data type mismatches"""
        if 'quantity' in field_name.lower() or 'cost' in field_name.lower():
            try:
                float(value)
                return False
            except:
                return True
        elif 'number' in field_name.lower() or 'id' in field_name.lower():
            return not isinstance(value, (str, int))
        
        return False
    
    def _detect_statistical_anomalies(self, data: pd.DataFrame) -> List[AnomalyResult]:
        """Detect anomalies using statistical methods"""
        anomalies = []
        
        # Identify numeric columns for statistical analysis
        numeric_columns = data.select_dtypes(include=[np.number]).columns
        
        for col in numeric_columns:
            if data[col].notna().sum() > 10:  # Need sufficient data
                # Calculate Z-scores
                mean_val = data[col].mean()
                std_val = data[col].std()
                
                if std_val > 0:
                    z_scores = np.abs((data[col] - mean_val) / std_val)
                    outliers = z_scores > 3  # 3-sigma rule
                    
                    for idx in data[outliers].index:
                        anomaly = AnomalyResult(
                            record_id=str(idx),
                            anomaly_type='statistical_outlier',
                            severity='medium',
                            description=f"Statistical outlier detected in {col}",
                            sap_value=data.loc[idx, col],
                            kinaxis_value=None,
                            confidence=min(0.9, z_scores.loc[idx] / 5),
                            auto_fixable=False,
                            recommended_action="Review for data entry errors"
                        )
                        anomalies.append(anomaly)
        
        return anomalies
    
    def _detect_ml_anomalies(self, data: pd.DataFrame) -> List[AnomalyResult]:
        """Detect anomalies using machine learning models"""
        anomalies = []
        
        # Prepare data for ML model
        numeric_data = data.select_dtypes(include=[np.number]).fillna(0)
        
        if len(numeric_data.columns) > 0 and len(numeric_data) > 10:
            try:
                # Scale the data
                scaled_data = self.scaler.fit_transform(numeric_data)
                
                # Fit isolation forest
                outlier_labels = self.ml_models['isolation_forest'].fit_predict(scaled_data)
                outlier_scores = self.ml_models['isolation_forest'].score_samples(scaled_data)
                
                # Identify anomalies (label = -1)
                anomaly_indices = np.where(outlier_labels == -1)[0]
                
                for idx in anomaly_indices:
                    original_idx = numeric_data.index[idx]
                    confidence = abs(outlier_scores[idx])
                    
                    anomaly = AnomalyResult(
                        record_id=str(original_idx),
                        anomaly_type='ml_detected_anomaly',
                        severity='low' if confidence < 0.1 else 'medium',
                        description="Machine learning model detected anomalous pattern",
                        sap_value=None,
                        kinaxis_value=None,
                        confidence=min(0.8, confidence * 2),
                        auto_fixable=False,
                        recommended_action="Investigate data pattern anomaly"
                    )
                    anomalies.append(anomaly)
            
            except Exception as e:
                self.logger.warning(f"ML anomaly detection failed: {e}")
        
        return anomalies
    
    def _create_summary(self, data: pd.DataFrame, anomalies: List[AnomalyResult], processing_time: float) -> DetectionSummary:
        """Create summary of anomaly detection results"""
        total_records = len(data)
        anomalies_found = len(anomalies)
        anomaly_rate = (anomalies_found / total_records * 100) if total_records > 0 else 0
        
        auto_fixable_count = sum(1 for a in anomalies if a.auto_fixable)
        manual_review_count = anomalies_found - auto_fixable_count
        
        # Breakdown by anomaly type
        anomaly_breakdown = {}
        for anomaly in anomalies:
            anomaly_type = anomaly.anomaly_type
            anomaly_breakdown[anomaly_type] = anomaly_breakdown.get(anomaly_type, 0) + 1
        
        return DetectionSummary(
            total_records=total_records,
            anomalies_found=anomalies_found,
            anomaly_rate=round(anomaly_rate, 2),
            auto_fixable_count=auto_fixable_count,
            manual_review_count=manual_review_count,
            anomaly_breakdown=anomaly_breakdown,
            processing_time=round(processing_time, 2)
        )
    
    def auto_remediate(self, anomalies: List[AnomalyResult], data: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
        """
        Automatically remediate fixable anomalies
        
        Returns:
            Tuple of (corrected_data, remediation_log)
        """
        corrected_data = data.copy()
        remediation_log = []
        
        for anomaly in anomalies:
            if anomaly.auto_fixable:
                try:
                    if anomaly.anomaly_type == 'negative_quantities':
                        # Fix negative quantities
                        corrected_data = self._fix_negative_quantities(corrected_data, anomaly)
                        remediation_log.append(f"Fixed negative quantity in record {anomaly.record_id}")
                    
                    elif anomaly.anomaly_type == 'excessive_precision':
                        # Fix excessive decimal precision
                        corrected_data = self._fix_excessive_precision(corrected_data, anomaly)
                        remediation_log.append(f"Rounded excessive precision in record {anomaly.record_id}")
                    
                    elif anomaly.anomaly_type == 'invalid_dates':
                        # Fix invalid dates
                        corrected_data = self._fix_invalid_dates(corrected_data, anomaly)
                        remediation_log.append(f"Fixed invalid date in record {anomaly.record_id}")
                
                except Exception as e:
                    remediation_log.append(f"Failed to fix anomaly in record {anomaly.record_id}: {e}")
        
        return corrected_data, remediation_log
    
    def _fix_negative_quantities(self, data: pd.DataFrame, anomaly: AnomalyResult) -> pd.DataFrame:
        """Fix negative quantities by taking absolute value"""
        idx = int(anomaly.record_id)
        for col in data.columns:
            if 'quantity' in col.lower() and data.loc[idx, col] < 0:
                data.loc[idx, col] = abs(data.loc[idx, col])
        return data
    
    def _fix_excessive_precision(self, data: pd.DataFrame, anomaly: AnomalyResult) -> pd.DataFrame:
        """Fix excessive decimal precision by rounding"""
        idx = int(anomaly.record_id)
        for col in data.columns:
            if col in ['unit_cost', 'total_value', 'weight'] and pd.notna(data.loc[idx, col]):
                data.loc[idx, col] = round(float(data.loc[idx, col]), 4)
        return data
    
    def _fix_invalid_dates(self, data: pd.DataFrame, anomaly: AnomalyResult) -> pd.DataFrame:
        """Fix invalid dates by setting to current date"""
        idx = int(anomaly.record_id)
        current_date = datetime.now().strftime('%Y-%m-%d')
        for col in data.columns:
            if 'date' in col.lower() and pd.isna(pd.to_datetime(data.loc[idx, col], errors='coerce')):
                data.loc[idx, col] = current_date
        return data
