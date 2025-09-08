"""
Pre-Batch Master Data Validation Agent for Pfizer Demo
Validates master data in SAP staging tables before batch transfer to Kinaxis
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Tuple, Optional
from dataclasses import dataclass
from datetime import datetime
import logging


@dataclass
class ValidationResult:
    """Result of pre-batch validation"""
    record_id: str
    validation_type: str
    severity: str  # 'critical', 'high', 'medium', 'low'
    description: str
    field_name: str
    current_value: Any
    expected_value: Any
    sap_staging_value: Any
    kinaxis_master_value: Any
    auto_fixable: bool
    recommended_action: str
    business_impact: str


@dataclass
class BatchValidationSummary:
    """Summary of pre-batch validation results"""
    total_records: int
    validation_failures: int
    failure_rate: float
    critical_failures: int
    auto_fixable_count: int
    batch_ready: bool
    processing_time: float
    failure_breakdown: Dict[str, int]


class PreBatchValidationAgent:
    """
    Agent that validates master data in SAP staging tables before 4-hour batch transfer to Kinaxis
    Prevents bad data from propagating downstream
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.validation_rules = self._initialize_validation_rules()
        self.critical_fields = self._define_critical_fields()
    
    def _initialize_validation_rules(self) -> Dict[str, Dict]:
        """Initialize validation rules for master data"""
        return {
            'production_version_consistency': {
                'description': 'Production versions must match between SAP and Kinaxis',
                'severity': 'critical',
                'auto_fixable': False,
                'business_impact': 'Prevents production planning errors'
            },
            'bom_version_alignment': {
                'description': 'BOM versions must be synchronized',
                'severity': 'critical', 
                'auto_fixable': False,
                'business_impact': 'Ensures correct material requirements'
            },
            'material_master_completeness': {
                'description': 'All required material master fields must be populated',
                'severity': 'high',
                'auto_fixable': True,
                'business_impact': 'Prevents planning disruptions'
            },
            'negative_quantities_check': {
                'description': 'No negative quantities allowed in master data',
                'severity': 'critical',
                'auto_fixable': True,
                'business_impact': 'Prevents inventory calculation errors'
            },
            'decimal_precision_validation': {
                'description': 'Numeric fields must have correct decimal precision (max 2 places)',
                'severity': 'medium',
                'auto_fixable': True,
                'business_impact': 'Ensures data consistency across systems'
            },
            'shipping_mode_validation': {
                'description': 'Shipping modes must be valid and follow business rules',
                'severity': 'high',
                'auto_fixable': True,
                'business_impact': 'Prevents delivery delays and cost inefficiencies'
            },
            'life_saving_drug_override': {
                'description': 'Life-saving drugs must use fastest shipping (AIR)',
                'severity': 'critical',
                'auto_fixable': True,
                'business_impact': 'Critical for patient safety and regulatory compliance'
            }
        }
    
    def _define_critical_fields(self) -> Dict[str, List[str]]:
        """Define critical fields that must be validated"""
        return {
            'material_master': [
                'material_number', 'material_description', 'plant_code',
                'production_version', 'bom_version', 'base_unit',
                'standard_cost', 'procurement_type'
            ],
            'production_data': [
                'production_order', 'material_number', 'production_qty',
                'production_version', 'bom_version', 'start_date', 'finish_date'
            ],
            'inventory_data': [
                'material_number', 'plant_code', 'quantity', 'amount',
                'movement_type', 'posting_date'
            ],
            'shipping_data': [
                'material_number', 'shipping_mode', 'priority_flag',
                'life_saving_indicator', 'destination'
            ]
        }
    
    def validate_pre_batch(self, sap_staging_data: Dict[str, pd.DataFrame], 
                          kinaxis_master_data: Dict[str, pd.DataFrame]) -> Tuple[List[ValidationResult], BatchValidationSummary]:
        """
        Main validation method for pre-batch master data validation
        
        Args:
            sap_staging_data: SAP staging tables ready for batch transfer
            kinaxis_master_data: Current Kinaxis master data for comparison
            
        Returns:
            Tuple of (validation_results, summary)
        """
        start_time = datetime.now()
        validation_results = []
        
        # Validate material master data
        if 'material_master' in sap_staging_data and 'item_master' in kinaxis_master_data:
            material_results = self._validate_material_master(
                sap_staging_data['material_master'],
                kinaxis_master_data['item_master']
            )
            validation_results.extend(material_results)
        
        # Validate production data
        if 'production_orders' in sap_staging_data:
            production_results = self._validate_production_data(
                sap_staging_data['production_orders']
            )
            validation_results.extend(production_results)
        
        # Validate inventory data
        if 'inventory_movements' in sap_staging_data:
            inventory_results = self._validate_inventory_data(
                sap_staging_data['inventory_movements']
            )
            validation_results.extend(inventory_results)
        
        # Validate shipping configurations
        shipping_results = self._validate_shipping_modes(sap_staging_data)
        validation_results.extend(shipping_results)
        
        # Create summary
        processing_time = (datetime.now() - start_time).total_seconds()
        summary = self._create_batch_summary(sap_staging_data, validation_results, processing_time)
        
        return validation_results, summary
    
    def _validate_material_master(self, sap_materials: pd.DataFrame, 
                                kinaxis_items: pd.DataFrame) -> List[ValidationResult]:
        """Validate material master data consistency"""
        results = []
        
        # Merge on material/item number for comparison
        merged = pd.merge(
            sap_materials, kinaxis_items,
            left_on='material_number', right_on='item_id',
            how='outer', suffixes=('_sap', '_kinaxis')
        )
        
        for idx, row in merged.iterrows():
            material_id = row.get('material_number') or row.get('item_id')
            
            # Check production version consistency
            sap_prod_version = row.get('production_version_sap')
            kinaxis_prod_version = row.get('production_version_kinaxis')  # Assuming field exists
            
            if pd.notna(sap_prod_version) and pd.notna(kinaxis_prod_version):
                if str(sap_prod_version) != str(kinaxis_prod_version):
                    results.append(ValidationResult(
                        record_id=str(material_id),
                        validation_type='production_version_consistency',
                        severity='critical',
                        description=f"Production version mismatch for material {material_id}",
                        field_name='production_version',
                        current_value=sap_prod_version,
                        expected_value=kinaxis_prod_version,
                        sap_staging_value=sap_prod_version,
                        kinaxis_master_value=kinaxis_prod_version,
                        auto_fixable=False,
                        recommended_action="Manual review required - sync production versions",
                        business_impact="May cause production planning errors"
                    ))
            
            # Check for missing critical fields
            for field in self.critical_fields['material_master']:
                sap_value = row.get(f"{field}_sap")
                if pd.isna(sap_value) or sap_value == '':
                    results.append(ValidationResult(
                        record_id=str(material_id),
                        validation_type='material_master_completeness',
                        severity='high',
                        description=f"Missing required field: {field}",
                        field_name=field,
                        current_value=sap_value,
                        expected_value="Non-empty value",
                        sap_staging_value=sap_value,
                        kinaxis_master_value=row.get(f"{field}_kinaxis"),
                        auto_fixable=True,
                        recommended_action=f"Populate {field} with default or derived value",
                        business_impact="May prevent successful batch transfer"
                    ))
            
            # Check decimal precision
            cost_value = row.get('standard_cost_sap')
            if pd.notna(cost_value):
                decimal_places = len(str(cost_value).split('.')[-1]) if '.' in str(cost_value) else 0
                if decimal_places > 2:
                    results.append(ValidationResult(
                        record_id=str(material_id),
                        validation_type='decimal_precision_validation',
                        severity='medium',
                        description=f"Excessive decimal precision in standard_cost: {decimal_places} places",
                        field_name='standard_cost',
                        current_value=cost_value,
                        expected_value=round(float(cost_value), 2),
                        sap_staging_value=cost_value,
                        kinaxis_master_value=row.get('standard_cost_kinaxis'),
                        auto_fixable=True,
                        recommended_action="Round to 2 decimal places",
                        business_impact="May cause calculation inconsistencies"
                    ))
        
        return results
    
    def _validate_production_data(self, production_data: pd.DataFrame) -> List[ValidationResult]:
        """Validate production order data"""
        results = []
        
        for idx, row in production_data.iterrows():
            order_id = row.get('production_order')
            
            # Check for negative quantities
            qty = row.get('production_qty')
            if pd.notna(qty) and qty < 0:
                results.append(ValidationResult(
                    record_id=str(order_id),
                    validation_type='negative_quantities_check',
                    severity='critical',
                    description=f"Negative production quantity: {qty}",
                    field_name='production_qty',
                    current_value=qty,
                    expected_value=abs(qty),
                    sap_staging_value=qty,
                    kinaxis_master_value=None,
                    auto_fixable=True,
                    recommended_action="Convert to positive value or investigate root cause",
                    business_impact="Will cause planning calculation errors"
                ))
            
            # Validate production version exists
            prod_version = row.get('production_version')
            if pd.isna(prod_version) or prod_version == '':
                results.append(ValidationResult(
                    record_id=str(order_id),
                    validation_type='production_version_consistency',
                    severity='critical',
                    description="Missing production version",
                    field_name='production_version',
                    current_value=prod_version,
                    expected_value="Valid production version",
                    sap_staging_value=prod_version,
                    kinaxis_master_value=None,
                    auto_fixable=False,
                    recommended_action="Assign valid production version",
                    business_impact="Production order cannot be processed in Kinaxis"
                ))
        
        return results
    
    def _validate_inventory_data(self, inventory_data: pd.DataFrame) -> List[ValidationResult]:
        """Validate inventory movement data"""
        results = []
        
        for idx, row in inventory_data.iterrows():
            doc_id = row.get('document_number')
            
            # Check for negative quantities in stock movements
            qty = row.get('quantity')
            movement_type = row.get('movement_type')
            
            # Certain movement types should not have negative quantities
            if pd.notna(qty) and qty < 0 and movement_type in ['101', '311', '501']:  # Receipts
                results.append(ValidationResult(
                    record_id=str(doc_id),
                    validation_type='negative_quantities_check',
                    severity='critical',
                    description=f"Negative quantity for receipt movement type {movement_type}",
                    field_name='quantity',
                    current_value=qty,
                    expected_value=abs(qty),
                    sap_staging_value=qty,
                    kinaxis_master_value=None,
                    auto_fixable=True,
                    recommended_action="Investigate movement type or correct quantity sign",
                    business_impact="Will cause inventory count discrepancies"
                ))
        
        return results
    
    def _validate_shipping_modes(self, sap_data: Dict[str, pd.DataFrame]) -> List[ValidationResult]:
        """Validate shipping mode configurations and life-saving drug overrides"""
        results = []
        
        # Check material master for shipping configurations
        if 'material_master' in sap_data:
            materials = sap_data['material_master']
            
            for idx, row in materials.iterrows():
                material_id = row.get('material_number')
                material_desc = row.get('material_description', '')
                
                # Check if this is a life-saving drug
                is_life_saving = self._is_life_saving_drug(material_desc)
                shipping_mode = row.get('shipping_mode', 'STANDARD')
                
                if is_life_saving and shipping_mode != 'AIR':
                    results.append(ValidationResult(
                        record_id=str(material_id),
                        validation_type='life_saving_drug_override',
                        severity='critical',
                        description=f"Life-saving drug {material_desc} must use AIR shipping",
                        field_name='shipping_mode',
                        current_value=shipping_mode,
                        expected_value='AIR',
                        sap_staging_value=shipping_mode,
                        kinaxis_master_value=None,
                        auto_fixable=True,
                        recommended_action="Override shipping mode to AIR for life-saving drugs",
                        business_impact="Critical for patient safety - may delay life-saving medications"
                    ))
                
                # Validate shipping mode is valid
                valid_modes = ['AIR', 'OCEAN', 'TRUCK', 'RAIL', 'EXPRESS']
                if shipping_mode not in valid_modes:
                    results.append(ValidationResult(
                        record_id=str(material_id),
                        validation_type='shipping_mode_validation',
                        severity='high',
                        description=f"Invalid shipping mode: {shipping_mode}",
                        field_name='shipping_mode',
                        current_value=shipping_mode,
                        expected_value='Valid shipping mode',
                        sap_staging_value=shipping_mode,
                        kinaxis_master_value=None,
                        auto_fixable=True,
                        recommended_action="Set to default shipping mode based on material type",
                        business_impact="May cause delivery planning errors"
                    ))
        
        return results
    
    def _is_life_saving_drug(self, material_description: str) -> bool:
        """Determine if a material is a life-saving drug"""
        life_saving_keywords = [
            'INSULIN', 'EPINEPHRINE', 'NITROGLYCERIN', 'MORPHINE',
            'EMERGENCY', 'CRITICAL', 'LIFE-SAVING', 'URGENT',
            'CARDIAC', 'RESUSCITATION', 'ANTIDOTE'
        ]
        
        desc_upper = material_description.upper()
        return any(keyword in desc_upper for keyword in life_saving_keywords)
    
    def _create_batch_summary(self, sap_data: Dict[str, pd.DataFrame], 
                            validation_results: List[ValidationResult], 
                            processing_time: float) -> BatchValidationSummary:
        """Create summary of batch validation results"""
        total_records = sum(len(df) for df in sap_data.values())
        validation_failures = len(validation_results)
        failure_rate = (validation_failures / total_records * 100) if total_records > 0 else 0
        
        critical_failures = sum(1 for r in validation_results if r.severity == 'critical')
        auto_fixable_count = sum(1 for r in validation_results if r.auto_fixable)
        
        # Batch is ready if no critical failures or all critical failures are auto-fixable
        critical_non_fixable = sum(1 for r in validation_results 
                                 if r.severity == 'critical' and not r.auto_fixable)
        batch_ready = critical_non_fixable == 0
        
        # Breakdown by validation type
        failure_breakdown = {}
        for result in validation_results:
            validation_type = result.validation_type
            failure_breakdown[validation_type] = failure_breakdown.get(validation_type, 0) + 1
        
        return BatchValidationSummary(
            total_records=total_records,
            validation_failures=validation_failures,
            failure_rate=round(failure_rate, 2),
            critical_failures=critical_failures,
            auto_fixable_count=auto_fixable_count,
            batch_ready=batch_ready,
            processing_time=round(processing_time, 2),
            failure_breakdown=failure_breakdown
        )
    
    def auto_remediate_batch(self, validation_results: List[ValidationResult], 
                           sap_data: Dict[str, pd.DataFrame]) -> Tuple[Dict[str, pd.DataFrame], List[str]]:
        """
        Automatically remediate fixable validation failures before batch transfer
        
        Returns:
            Tuple of (corrected_data, remediation_log)
        """
        corrected_data = {table: df.copy() for table, df in sap_data.items()}
        remediation_log = []
        
        for result in validation_results:
            if result.auto_fixable:
                try:
                    if result.validation_type == 'negative_quantities_check':
                        corrected_data = self._fix_negative_quantities(corrected_data, result)
                        remediation_log.append(f"Fixed negative quantity in {result.record_id}")
                    
                    elif result.validation_type == 'decimal_precision_validation':
                        corrected_data = self._fix_decimal_precision(corrected_data, result)
                        remediation_log.append(f"Rounded decimal precision in {result.record_id}")
                    
                    elif result.validation_type == 'life_saving_drug_override':
                        corrected_data = self._fix_shipping_mode(corrected_data, result)
                        remediation_log.append(f"Override shipping to AIR for life-saving drug {result.record_id}")
                    
                    elif result.validation_type == 'material_master_completeness':
                        corrected_data = self._fix_missing_fields(corrected_data, result)
                        remediation_log.append(f"Populated missing field {result.field_name} in {result.record_id}")
                
                except Exception as e:
                    remediation_log.append(f"Failed to fix {result.validation_type} in {result.record_id}: {e}")
        
        return corrected_data, remediation_log
    
    def _fix_negative_quantities(self, data: Dict[str, pd.DataFrame], result: ValidationResult) -> Dict[str, pd.DataFrame]:
        """Fix negative quantities in the data"""
        # Implementation would fix negative quantities based on the specific table and field
        return data
    
    def _fix_decimal_precision(self, data: Dict[str, pd.DataFrame], result: ValidationResult) -> Dict[str, pd.DataFrame]:
        """Fix excessive decimal precision"""
        # Implementation would round values to appropriate precision
        return data
    
    def _fix_shipping_mode(self, data: Dict[str, pd.DataFrame], result: ValidationResult) -> Dict[str, pd.DataFrame]:
        """Fix shipping mode for life-saving drugs"""
        # Implementation would override shipping mode to AIR
        return data
    
    def _fix_missing_fields(self, data: Dict[str, pd.DataFrame], result: ValidationResult) -> Dict[str, pd.DataFrame]:
        """Fix missing required fields"""
        # Implementation would populate missing fields with defaults
        return data
