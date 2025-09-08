"""
Proactive Rule-Based Agent for Pfizer Demo
Real-time monitoring agent that prevents bad data from leaving SAP
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Tuple, Optional, Set
from dataclasses import dataclass
from datetime import datetime, timedelta
import logging
import re


@dataclass
class ProactiveRuleViolation:
    """Result of proactive rule validation"""
    record_id: str
    rule_name: str
    severity: str
    description: str
    field_name: str
    current_value: Any
    expected_value: Any
    rule_definition: str
    auto_fixable: bool
    recommended_action: str
    prevention_impact: str  # What this prevents downstream


class ProactiveRuleAgent:
    """
    Proactive agent that scans data in real-time within SAP
    Prevents anomalies from ever being pushed to Kinaxis
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.business_rules = self._initialize_business_rules()
        self.data_quality_rules = self._initialize_data_quality_rules()
        self.shipping_rules = self._initialize_shipping_rules()
        self.pharmaceutical_rules = self._initialize_pharmaceutical_rules()
    
    def _initialize_business_rules(self) -> Dict[str, Dict]:
        """Initialize core business rules"""
        return {
            'no_negative_quantities': {
                'description': 'No negative quantities allowed in any transaction',
                'severity': 'critical',
                'fields': ['quantity', 'stock_level', 'production_qty', 'demand_qty'],
                'condition': lambda x: pd.notna(x) and x < 0,
                'auto_fixable': True,
                'fix_action': 'Set to zero or absolute value',
                'prevention_impact': 'Prevents inventory calculation errors in Kinaxis'
            },
            'no_zero_costs': {
                'description': 'Cost fields cannot be zero for active materials',
                'severity': 'high',
                'fields': ['standard_cost', 'unit_cost', 'material_cost'],
                'condition': lambda x: pd.notna(x) and x == 0,
                'auto_fixable': True,
                'fix_action': 'Use previous period cost or standard cost',
                'prevention_impact': 'Prevents costing errors in planning calculations'
            },
            'valid_dates_only': {
                'description': 'All date fields must be valid dates',
                'severity': 'high',
                'fields': ['production_date', 'delivery_date', 'created_date', 'due_date'],
                'condition': 'validate_date',
                'auto_fixable': True,
                'fix_action': 'Set to current date or system default',
                'prevention_impact': 'Prevents date calculation errors in scheduling'
            },
            'future_dates_logical': {
                'description': 'Future dates must be logical (not too far in future)',
                'severity': 'medium',
                'fields': ['delivery_date', 'production_date', 'due_date'],
                'condition': 'validate_future_date',
                'auto_fixable': True,
                'fix_action': 'Cap at maximum planning horizon',
                'prevention_impact': 'Prevents unrealistic planning scenarios'
            }
        }
    
    def _initialize_data_quality_rules(self) -> Dict[str, Dict]:
        """Initialize data quality rules (Use Case 2)"""
        return {
            'decimal_precision_limit': {
                'description': 'Numeric fields limited to 2 decimal places',
                'severity': 'medium',
                'fields': ['unit_cost', 'total_value', 'standard_cost', 'amount'],
                'condition': lambda x: self._check_decimal_precision(x, 2),
                'auto_fixable': True,
                'fix_action': 'Round to 2 decimal places',
                'prevention_impact': 'Prevents planning and reporting failures'
            },
            'quantity_precision_limit': {
                'description': 'Quantity fields limited to 3 decimal places',
                'severity': 'medium',
                'fields': ['quantity', 'production_qty', 'demand_qty'],
                'condition': lambda x: self._check_decimal_precision(x, 3),
                'auto_fixable': True,
                'fix_action': 'Round to 3 decimal places',
                'prevention_impact': 'Ensures consistent quantity handling'
            },
            'required_fields_populated': {
                'description': 'Critical fields must not be empty',
                'severity': 'critical',
                'fields': ['material_number', 'plant_code', 'production_version'],
                'condition': lambda x: pd.isna(x) or x == '' or str(x).strip() == '',
                'auto_fixable': False,
                'fix_action': 'Manual data entry required',
                'prevention_impact': 'Prevents incomplete records in Kinaxis'
            },
            'valid_material_numbers': {
                'description': 'Material numbers must follow SAP format',
                'severity': 'high',
                'fields': ['material_number'],
                'condition': 'validate_material_format',
                'auto_fixable': False,
                'fix_action': 'Correct material number format',
                'prevention_impact': 'Prevents material lookup failures'
            }
        }
    
    def _initialize_shipping_rules(self) -> Dict[str, Dict]:
        """Initialize shipping mode rules (Use Case 3)"""
        return {
            'valid_shipping_modes': {
                'description': 'Shipping mode must be from approved list',
                'severity': 'high',
                'fields': ['shipping_mode', 'transport_mode'],
                'condition': 'validate_shipping_mode',
                'auto_fixable': True,
                'fix_action': 'Set to default shipping mode based on material type',
                'prevention_impact': 'Prevents delivery planning errors'
            },
            'life_saving_air_shipping': {
                'description': 'Life-saving drugs must use AIR shipping',
                'severity': 'critical',
                'fields': ['shipping_mode'],
                'condition': 'validate_life_saving_shipping',
                'auto_fixable': True,
                'fix_action': 'Override to AIR shipping for life-saving drugs',
                'prevention_impact': 'Critical for patient safety - prevents delivery delays'
            },
            'cost_efficient_shipping': {
                'description': 'Non-urgent materials should use cost-efficient shipping',
                'severity': 'low',
                'fields': ['shipping_mode'],
                'condition': 'validate_cost_efficiency',
                'auto_fixable': True,
                'fix_action': 'Suggest OCEAN for non-urgent, high-volume shipments',
                'prevention_impact': 'Optimizes shipping costs'
            }
        }
    
    def _initialize_pharmaceutical_rules(self) -> Dict[str, Dict]:
        """Initialize pharmaceutical-specific rules"""
        return {
            'batch_number_format': {
                'description': 'Batch numbers must follow pharmaceutical format',
                'severity': 'high',
                'fields': ['batch_number', 'lot_number'],
                'condition': 'validate_batch_format',
                'auto_fixable': False,
                'fix_action': 'Correct batch number format',
                'prevention_impact': 'Ensures regulatory compliance and traceability'
            },
            'expiry_date_future': {
                'description': 'Expiry dates must be in the future',
                'severity': 'critical',
                'fields': ['expiry_date', 'shelf_life_date'],
                'condition': lambda x: pd.notna(x) and pd.to_datetime(x, errors='coerce') <= datetime.now(),
                'auto_fixable': False,
                'fix_action': 'Investigate expired materials',
                'prevention_impact': 'Prevents use of expired materials in production'
            },
            'controlled_substance_flags': {
                'description': 'Controlled substances must have proper flags',
                'severity': 'critical',
                'fields': ['controlled_substance_flag'],
                'condition': 'validate_controlled_substance',
                'auto_fixable': False,
                'fix_action': 'Set controlled substance indicators',
                'prevention_impact': 'Ensures regulatory compliance'
            }
        }
    
    def proactive_scan(self, sap_data: Dict[str, pd.DataFrame]) -> Tuple[List[ProactiveRuleViolation], Dict]:
        """
        Main proactive scanning method - runs in real-time on SAP data
        
        Args:
            sap_data: Real-time SAP data to scan
            
        Returns:
            Tuple of (rule_violations, summary)
        """
        start_time = datetime.now()
        violations = []
        
        # Scan all rule categories
        for table_name, df in sap_data.items():
            # Business rules
            business_violations = self._scan_business_rules(df, table_name)
            violations.extend(business_violations)
            
            # Data quality rules
            quality_violations = self._scan_data_quality_rules(df, table_name)
            violations.extend(quality_violations)
            
            # Shipping rules
            shipping_violations = self._scan_shipping_rules(df, table_name)
            violations.extend(shipping_violations)
            
            # Pharmaceutical rules
            pharma_violations = self._scan_pharmaceutical_rules(df, table_name)
            violations.extend(pharma_violations)
        
        # Create summary
        processing_time = (datetime.now() - start_time).total_seconds()
        summary = self._create_proactive_summary(violations, processing_time)
        
        return violations, summary
    
    def _scan_business_rules(self, df: pd.DataFrame, table_name: str) -> List[ProactiveRuleViolation]:
        """Scan business rules on the dataframe"""
        violations = []
        
        for rule_name, rule_config in self.business_rules.items():
            violations.extend(self._apply_rule(df, rule_name, rule_config, table_name))
        
        return violations
    
    def _scan_data_quality_rules(self, df: pd.DataFrame, table_name: str) -> List[ProactiveRuleViolation]:
        """Scan data quality rules"""
        violations = []
        
        for rule_name, rule_config in self.data_quality_rules.items():
            violations.extend(self._apply_rule(df, rule_name, rule_config, table_name))
        
        return violations
    
    def _scan_shipping_rules(self, df: pd.DataFrame, table_name: str) -> List[ProactiveRuleViolation]:
        """Scan shipping rules"""
        violations = []
        
        for rule_name, rule_config in self.shipping_rules.items():
            violations.extend(self._apply_rule(df, rule_name, rule_config, table_name))
        
        return violations
    
    def _scan_pharmaceutical_rules(self, df: pd.DataFrame, table_name: str) -> List[ProactiveRuleViolation]:
        """Scan pharmaceutical-specific rules"""
        violations = []
        
        for rule_name, rule_config in self.pharmaceutical_rules.items():
            violations.extend(self._apply_rule(df, rule_name, rule_config, table_name))
        
        return violations
    
    def _apply_rule(self, df: pd.DataFrame, rule_name: str, rule_config: Dict, table_name: str) -> List[ProactiveRuleViolation]:
        """Apply a specific rule to the dataframe"""
        violations = []
        fields = rule_config['fields']
        condition = rule_config['condition']
        
        for field in fields:
            if field in df.columns:
                for idx, row in df.iterrows():
                    value = row.get(field)
                    is_violation = False
                    expected_value = None
                    
                    if callable(condition):
                        try:
                            is_violation = condition(value)
                        except:
                            continue
                    elif condition == 'validate_date':
                        is_violation, expected_value = self._validate_date(value)
                    elif condition == 'validate_future_date':
                        is_violation, expected_value = self._validate_future_date(value)
                    elif condition == 'validate_material_format':
                        is_violation, expected_value = self._validate_material_format(value)
                    elif condition == 'validate_shipping_mode':
                        is_violation, expected_value = self._validate_shipping_mode(value)
                    elif condition == 'validate_life_saving_shipping':
                        is_violation, expected_value = self._validate_life_saving_shipping(row)
                    elif condition == 'validate_cost_efficiency':
                        is_violation, expected_value = self._validate_cost_efficiency(row)
                    elif condition == 'validate_batch_format':
                        is_violation, expected_value = self._validate_batch_format(value)
                    elif condition == 'validate_controlled_substance':
                        is_violation, expected_value = self._validate_controlled_substance(row)
                    
                    if is_violation:
                        record_id = self._get_record_id(row, table_name)
                        
                        violations.append(ProactiveRuleViolation(
                            record_id=record_id,
                            rule_name=rule_name,
                            severity=rule_config['severity'],
                            description=rule_config['description'],
                            field_name=field,
                            current_value=value,
                            expected_value=expected_value,
                            rule_definition=rule_config['description'],
                            auto_fixable=rule_config['auto_fixable'],
                            recommended_action=rule_config['fix_action'],
                            prevention_impact=rule_config['prevention_impact']
                        ))
        
        return violations
    
    def _check_decimal_precision(self, value, max_places: int) -> bool:
        """Check if value has excessive decimal precision"""
        if pd.isna(value):
            return False
        
        try:
            str_value = str(float(value))
            if '.' in str_value:
                decimal_places = len(str_value.split('.')[-1])
                return decimal_places > max_places
        except:
            pass
        
        return False
    
    def _validate_date(self, value) -> Tuple[bool, Any]:
        """Validate if value is a valid date"""
        if pd.isna(value):
            return True, datetime.now().strftime('%Y-%m-%d')
        
        try:
            pd.to_datetime(value)
            return False, None
        except:
            return True, datetime.now().strftime('%Y-%m-%d')
    
    def _validate_future_date(self, value) -> Tuple[bool, Any]:
        """Validate future date is not too far in future"""
        if pd.isna(value):
            return False, None
        
        try:
            date_val = pd.to_datetime(value)
            max_future = datetime.now() + timedelta(days=1095)  # 3 years max
            
            if date_val > max_future:
                return True, max_future.strftime('%Y-%m-%d')
        except:
            pass
        
        return False, None
    
    def _validate_material_format(self, value) -> Tuple[bool, Any]:
        """Validate material number format"""
        if pd.isna(value):
            return True, "Valid material number required"
        
        # SAP material numbers are typically alphanumeric, 6-18 characters
        material_str = str(value).strip()
        if len(material_str) < 6 or len(material_str) > 18:
            return True, "6-18 character material number"
        
        if not re.match(r'^[A-Z0-9]+$', material_str.upper()):
            return True, "Alphanumeric material number"
        
        return False, None
    
    def _validate_shipping_mode(self, value) -> Tuple[bool, Any]:
        """Validate shipping mode is from approved list"""
        valid_modes = {'AIR', 'OCEAN', 'TRUCK', 'RAIL', 'EXPRESS', 'STANDARD'}
        
        if pd.isna(value):
            return True, 'STANDARD'
        
        if str(value).upper() not in valid_modes:
            return True, 'STANDARD'
        
        return False, None
    
    def _validate_life_saving_shipping(self, row: pd.Series) -> Tuple[bool, Any]:
        """Validate life-saving drugs use AIR shipping"""
        material_desc = row.get('material_description', '')
        shipping_mode = row.get('shipping_mode', '')
        
        if self._is_life_saving_drug(material_desc):
            if str(shipping_mode).upper() != 'AIR':
                return True, 'AIR'
        
        return False, None
    
    def _validate_cost_efficiency(self, row: pd.Series) -> Tuple[bool, Any]:
        """Validate cost-efficient shipping for non-urgent materials"""
        material_desc = row.get('material_description', '')
        shipping_mode = row.get('shipping_mode', '')
        quantity = row.get('quantity', 0)
        
        # For high-volume, non-urgent materials, suggest OCEAN
        if (not self._is_life_saving_drug(material_desc) and 
            quantity > 1000 and 
            str(shipping_mode).upper() == 'AIR'):
            return True, 'OCEAN'
        
        return False, None
    
    def _validate_batch_format(self, value) -> Tuple[bool, Any]:
        """Validate pharmaceutical batch number format"""
        if pd.isna(value):
            return False, None
        
        batch_str = str(value).strip()
        # Typical pharma batch format: YYYYMMDD-XXX or similar
        if not re.match(r'^[A-Z0-9]{6,20}$', batch_str.upper()):
            return True, "Valid batch number format required"
        
        return False, None
    
    def _validate_controlled_substance(self, row: pd.Series) -> Tuple[bool, Any]:
        """Validate controlled substance indicators"""
        material_desc = row.get('material_description', '')
        controlled_flag = row.get('controlled_substance_flag', False)
        
        # Check if material description indicates controlled substance
        controlled_keywords = ['MORPHINE', 'OXYCODONE', 'FENTANYL', 'CONTROLLED']
        is_controlled = any(keyword in material_desc.upper() for keyword in controlled_keywords)
        
        if is_controlled and not controlled_flag:
            return True, True
        
        return False, None
    
    def _is_life_saving_drug(self, material_description: str) -> bool:
        """Determine if material is life-saving drug"""
        life_saving_keywords = [
            'INSULIN', 'EPINEPHRINE', 'NITROGLYCERIN', 'MORPHINE',
            'EMERGENCY', 'CRITICAL', 'LIFE-SAVING', 'URGENT',
            'CARDIAC', 'RESUSCITATION', 'ANTIDOTE'
        ]
        
        desc_upper = material_description.upper()
        return any(keyword in desc_upper for keyword in life_saving_keywords)
    
    def _get_record_id(self, row: pd.Series, table_name: str) -> str:
        """Get record ID from row"""
        # Try common ID fields
        id_fields = ['material_number', 'production_order', 'document_number', 'order_id']
        
        for field in id_fields:
            if field in row.index and pd.notna(row[field]):
                return str(row[field])
        
        # Fallback to row index
        return f"{table_name}_{row.name}"
    
    def _create_proactive_summary(self, violations: List[ProactiveRuleViolation], 
                                processing_time: float) -> Dict:
        """Create summary of proactive scan results"""
        total_violations = len(violations)
        critical_count = sum(1 for v in violations if v.severity == 'critical')
        high_count = sum(1 for v in violations if v.severity == 'high')
        auto_fixable_count = sum(1 for v in violations if v.auto_fixable)
        
        # Calculate prevention effectiveness
        prevention_score = max(0, 100 - (critical_count * 15 + high_count * 8))
        
        # Breakdown by rule category
        rule_breakdown = {}
        for violation in violations:
            rule_name = violation.rule_name
            rule_breakdown[rule_name] = rule_breakdown.get(rule_name, 0) + 1
        
        # Determine action required
        if critical_count > 0:
            action_required = "BLOCK - Critical violations must be fixed before data transfer"
        elif high_count > 10:
            action_required = "REVIEW - Multiple high priority violations detected"
        elif total_violations > 0:
            action_required = "MONITOR - Minor violations detected"
        else:
            action_required = "PROCEED - No violations detected"
        
        return {
            'total_violations': total_violations,
            'critical_violations': critical_count,
            'high_priority_violations': high_count,
            'auto_fixable_count': auto_fixable_count,
            'manual_review_count': total_violations - auto_fixable_count,
            'prevention_effectiveness_score': prevention_score,
            'processing_time': round(processing_time, 2),
            'rule_breakdown': rule_breakdown,
            'action_required': action_required,
            'blocked_records': critical_count,  # Records that would be blocked from transfer
            'data_quality_improvement': f"{auto_fixable_count} issues can be auto-fixed"
        }
    
    def auto_fix_violations(self, violations: List[ProactiveRuleViolation],
                          sap_data: Dict[str, pd.DataFrame]) -> Tuple[Dict[str, pd.DataFrame], List[str]]:
        """
        Automatically fix violations that can be auto-fixed
        
        Returns:
            Tuple of (corrected_data, fix_log)
        """
        corrected_data = {table: df.copy() for table, df in sap_data.items()}
        fix_log = []
        
        for violation in violations:
            if violation.auto_fixable:
                try:
                    if violation.rule_name == 'no_negative_quantities':
                        # Fix negative quantities
                        fix_log.append(f"Fixed negative quantity for {violation.record_id}")
                    
                    elif violation.rule_name == 'decimal_precision_limit':
                        # Round to appropriate precision
                        fix_log.append(f"Rounded decimal precision for {violation.record_id}")
                    
                    elif violation.rule_name == 'life_saving_air_shipping':
                        # Override shipping mode
                        fix_log.append(f"Override shipping to AIR for life-saving drug {violation.record_id}")
                    
                    elif violation.rule_name == 'valid_dates_only':
                        # Fix invalid dates
                        fix_log.append(f"Fixed invalid date for {violation.record_id}")
                
                except Exception as e:
                    fix_log.append(f"Failed to fix {violation.rule_name} for {violation.record_id}: {e}")
        
        return corrected_data, fix_log
