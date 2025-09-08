"""
Return Flow Validation Agent for Pfizer Demo
MOST CRITICAL AGENT - Validates planning results flowing back from Kinaxis to SAP
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Tuple, Optional
from dataclasses import dataclass
from datetime import datetime, timedelta
import logging


@dataclass
class ReturnFlowValidationResult:
    """Result of return flow validation"""
    order_id: str
    validation_type: str
    severity: str
    description: str
    field_name: str
    kinaxis_planned_value: Any
    sap_expected_value: Any
    variance: float
    auto_fixable: bool
    recommended_action: str
    business_impact: str
    execution_risk: str  # Impact on production/procurement execution


class ReturnFlowValidationAgent:
    """
    MOST CRITICAL AGENT - Validates planning results from Kinaxis back to SAP
    Ensures planned orders, production orders, and requisitions are correct for execution
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.validation_rules = self._initialize_return_flow_rules()
        self.execution_thresholds = self._initialize_execution_thresholds()
    
    def _initialize_return_flow_rules(self) -> Dict[str, Dict]:
        """Initialize validation rules for return flow data"""
        return {
            'planned_order_consistency': {
                'description': 'Planned orders from Kinaxis must match SAP execution capabilities',
                'severity': 'critical',
                'business_impact': 'Drives all production and procurement execution',
                'execution_risk': 'Production disruption if incorrect'
            },
            'production_version_validation': {
                'description': 'Production versions must be valid and executable in SAP',
                'severity': 'critical',
                'business_impact': 'Determines BOM and routing for production',
                'execution_risk': 'Manufacturing errors if version mismatch'
            },
            'capacity_feasibility_check': {
                'description': 'Planned quantities must be within SAP capacity constraints',
                'severity': 'high',
                'business_impact': 'Ensures realistic production schedules',
                'execution_risk': 'Schedule delays if capacity exceeded'
            },
            'material_availability_validation': {
                'description': 'Required materials must be available or procurable',
                'severity': 'critical',
                'business_impact': 'Prevents production starts without materials',
                'execution_risk': 'Production stops if materials unavailable'
            },
            'lead_time_validation': {
                'description': 'Planned dates must respect SAP lead times',
                'severity': 'high',
                'business_impact': 'Ensures achievable delivery dates',
                'execution_risk': 'Customer delivery delays'
            },
            'procurement_rule_compliance': {
                'description': 'Purchase requisitions must follow SAP procurement rules',
                'severity': 'high',
                'business_impact': 'Ensures proper vendor selection and approval',
                'execution_risk': 'Procurement delays or compliance issues'
            }
        }
    
    def _initialize_execution_thresholds(self) -> Dict[str, float]:
        """Initialize thresholds for execution validation"""
        return {
            'quantity_tolerance': 0.05,  # 5% quantity variance allowed
            'date_tolerance_days': 2,    # 2 days date variance allowed
            'capacity_utilization_max': 0.95,  # 95% max capacity utilization
            'lead_time_buffer_days': 1,  # 1 day buffer for lead times
            'cost_variance_threshold': 0.10  # 10% cost variance threshold
        }
    
    def validate_return_flow(self, kinaxis_planned_data: Dict[str, pd.DataFrame],
                           sap_master_data: Dict[str, pd.DataFrame],
                           sap_capacity_data: Dict[str, pd.DataFrame]) -> Tuple[List[ReturnFlowValidationResult], Dict]:
        """
        Main validation method for return flow from Kinaxis to SAP
        
        Args:
            kinaxis_planned_data: Planning results from Kinaxis
            sap_master_data: SAP master data for validation
            sap_capacity_data: SAP capacity and constraint data
            
        Returns:
            Tuple of (validation_results, summary)
        """
        start_time = datetime.now()
        validation_results = []
        
        # Validate planned orders (MOST CRITICAL)
        if 'planned_orders' in kinaxis_planned_data:
            planned_order_results = self._validate_planned_orders_return(
                kinaxis_planned_data['planned_orders'],
                sap_master_data.get('material_master'),
                sap_capacity_data.get('work_centers')
            )
            validation_results.extend(planned_order_results)
        
        # Validate production orders
        if 'production_orders' in kinaxis_planned_data:
            production_results = self._validate_production_orders_return(
                kinaxis_planned_data['production_orders'],
                sap_master_data.get('material_master'),
                sap_master_data.get('bom_data')
            )
            validation_results.extend(production_results)
        
        # Validate purchase requisitions
        if 'purchase_requisitions' in kinaxis_planned_data:
            requisition_results = self._validate_purchase_requisitions_return(
                kinaxis_planned_data['purchase_requisitions'],
                sap_master_data.get('vendor_master'),
                sap_master_data.get('purchasing_info')
            )
            validation_results.extend(requisition_results)
        
        # Validate material requirements
        material_results = self._validate_material_requirements(
            kinaxis_planned_data,
            sap_master_data.get('material_master')
        )
        validation_results.extend(material_results)
        
        # Create summary
        processing_time = (datetime.now() - start_time).total_seconds()
        summary = self._create_return_flow_summary(validation_results, processing_time)
        
        return validation_results, summary
    
    def _validate_planned_orders_return(self, kinaxis_planned: pd.DataFrame,
                                      sap_materials: Optional[pd.DataFrame],
                                      sap_capacity: Optional[pd.DataFrame]) -> List[ReturnFlowValidationResult]:
        """
        Validate planned orders returning from Kinaxis - MOST CRITICAL
        These drive all execution in SAP
        """
        results = []
        
        for idx, row in kinaxis_planned.iterrows():
            order_id = row.get('planned_order_id')
            material_id = row.get('item_id')
            planned_qty = row.get('planned_quantity')
            planned_start = pd.to_datetime(row.get('planned_start_date'), errors='coerce')
            planned_finish = pd.to_datetime(row.get('planned_finish_date'), errors='coerce')
            production_version = row.get('production_version')
            
            # Validate production version exists in SAP
            if sap_materials is not None:
                material_row = sap_materials[sap_materials['material_number'] == material_id]
                if not material_row.empty:
                    sap_prod_version = material_row.iloc[0].get('production_version')
                    
                    if str(production_version) != str(sap_prod_version):
                        results.append(ReturnFlowValidationResult(
                            order_id=str(order_id),
                            validation_type='production_version_validation',
                            severity='critical',
                            description=f"Production version mismatch: Kinaxis={production_version}, SAP={sap_prod_version}",
                            field_name='production_version',
                            kinaxis_planned_value=production_version,
                            sap_expected_value=sap_prod_version,
                            variance=1.0,
                            auto_fixable=False,
                            recommended_action="Update Kinaxis with correct SAP production version",
                            business_impact="Will cause BOM/routing errors in production execution",
                            execution_risk="Production will fail to start or use wrong materials"
                        ))
            
            # Validate planned quantity is reasonable
            if pd.notna(planned_qty):
                if planned_qty <= 0:
                    results.append(ReturnFlowValidationResult(
                        order_id=str(order_id),
                        validation_type='planned_order_consistency',
                        severity='critical',
                        description=f"Invalid planned quantity: {planned_qty}",
                        field_name='planned_quantity',
                        kinaxis_planned_value=planned_qty,
                        sap_expected_value=">0",
                        variance=abs(planned_qty),
                        auto_fixable=False,
                        recommended_action="Investigate planning logic in Kinaxis",
                        business_impact="Cannot execute order with invalid quantity",
                        execution_risk="Order will be rejected by SAP"
                    ))
                
                # Check if quantity has excessive precision
                decimal_places = len(str(planned_qty).split('.')[-1]) if '.' in str(planned_qty) else 0
                if decimal_places > 3:  # Allow 3 decimal places for planned orders
                    results.append(ReturnFlowValidationResult(
                        order_id=str(order_id),
                        validation_type='planned_order_consistency',
                        severity='medium',
                        description=f"Excessive decimal precision in quantity: {decimal_places} places",
                        field_name='planned_quantity',
                        kinaxis_planned_value=planned_qty,
                        sap_expected_value=round(planned_qty, 3),
                        variance=decimal_places - 3,
                        auto_fixable=True,
                        recommended_action="Round to 3 decimal places",
                        business_impact="May cause rounding errors in execution",
                        execution_risk="Minor - execution will round automatically"
                    ))
            
            # Validate planned dates are in the future and logical
            if pd.notna(planned_start) and pd.notna(planned_finish):
                if planned_start >= planned_finish:
                    results.append(ReturnFlowValidationResult(
                        order_id=str(order_id),
                        validation_type='lead_time_validation',
                        severity='critical',
                        description="Planned start date is after finish date",
                        field_name='planned_dates',
                        kinaxis_planned_value=f"Start: {planned_start}, Finish: {planned_finish}",
                        sap_expected_value="Start < Finish",
                        variance=(planned_start - planned_finish).days,
                        auto_fixable=False,
                        recommended_action="Fix date logic in Kinaxis planning",
                        business_impact="Order cannot be scheduled in SAP",
                        execution_risk="Order will be rejected by production planning"
                    ))
                
                # Check if start date is too soon (less than lead time)
                days_from_now = (planned_start - datetime.now()).days
                if days_from_now < self.execution_thresholds['lead_time_buffer_days']:
                    results.append(ReturnFlowValidationResult(
                        order_id=str(order_id),
                        validation_type='lead_time_validation',
                        severity='high',
                        description=f"Planned start date too soon: {days_from_now} days from now",
                        field_name='planned_start_date',
                        kinaxis_planned_value=planned_start,
                        sap_expected_value=datetime.now() + timedelta(days=self.execution_thresholds['lead_time_buffer_days']),
                        variance=abs(days_from_now),
                        auto_fixable=True,
                        recommended_action="Adjust planned start date to allow sufficient lead time",
                        business_impact="May not allow sufficient time for material procurement",
                        execution_risk="Rush order may increase costs or cause delays"
                    ))
        
        return results
    
    def _validate_production_orders_return(self, kinaxis_production: pd.DataFrame,
                                         sap_materials: Optional[pd.DataFrame],
                                         sap_bom: Optional[pd.DataFrame]) -> List[ReturnFlowValidationResult]:
        """Validate production orders returning from Kinaxis"""
        results = []
        
        for idx, row in kinaxis_production.iterrows():
            order_id = row.get('production_order_id')
            material_id = row.get('item_id')
            production_qty = row.get('production_quantity')
            bom_version = row.get('bom_version')
            
            # Validate BOM version exists in SAP
            if sap_bom is not None:
                bom_row = sap_bom[
                    (sap_bom['parent_material'] == material_id) & 
                    (sap_bom['bom_version'] == bom_version)
                ]
                
                if bom_row.empty:
                    results.append(ReturnFlowValidationResult(
                        order_id=str(order_id),
                        validation_type='material_availability_validation',
                        severity='critical',
                        description=f"BOM version {bom_version} not found in SAP for material {material_id}",
                        field_name='bom_version',
                        kinaxis_planned_value=bom_version,
                        sap_expected_value="Valid BOM version",
                        variance=1.0,
                        auto_fixable=False,
                        recommended_action="Verify BOM version exists in SAP or update Kinaxis",
                        business_impact="Production cannot start without valid BOM",
                        execution_risk="Production order will be rejected"
                    ))
            
            # Validate production quantity
            if pd.notna(production_qty) and production_qty <= 0:
                results.append(ReturnFlowValidationResult(
                    order_id=str(order_id),
                    validation_type='planned_order_consistency',
                    severity='critical',
                    description=f"Invalid production quantity: {production_qty}",
                    field_name='production_quantity',
                    kinaxis_planned_value=production_qty,
                    sap_expected_value=">0",
                    variance=abs(production_qty),
                    auto_fixable=False,
                    recommended_action="Fix production quantity in Kinaxis",
                    business_impact="Cannot produce with invalid quantity",
                    execution_risk="Production order will fail"
                ))
        
        return results
    
    def _validate_purchase_requisitions_return(self, kinaxis_requisitions: pd.DataFrame,
                                             sap_vendors: Optional[pd.DataFrame],
                                             sap_purchasing: Optional[pd.DataFrame]) -> List[ReturnFlowValidationResult]:
        """Validate purchase requisitions returning from Kinaxis"""
        results = []
        
        for idx, row in kinaxis_requisitions.iterrows():
            req_id = row.get('requisition_id')
            material_id = row.get('item_id')
            req_qty = row.get('requisition_quantity')
            vendor_id = row.get('preferred_vendor')
            delivery_date = pd.to_datetime(row.get('required_date'), errors='coerce')
            
            # Validate vendor exists in SAP
            if sap_vendors is not None and pd.notna(vendor_id):
                vendor_row = sap_vendors[sap_vendors['vendor_number'] == vendor_id]
                if vendor_row.empty:
                    results.append(ReturnFlowValidationResult(
                        order_id=str(req_id),
                        validation_type='procurement_rule_compliance',
                        severity='high',
                        description=f"Vendor {vendor_id} not found in SAP vendor master",
                        field_name='preferred_vendor',
                        kinaxis_planned_value=vendor_id,
                        sap_expected_value="Valid SAP vendor",
                        variance=1.0,
                        auto_fixable=False,
                        recommended_action="Use valid SAP vendor or add vendor to master data",
                        business_impact="Procurement will be delayed for vendor setup",
                        execution_risk="Purchase requisition will be rejected"
                    ))
            
            # Validate requisition quantity
            if pd.notna(req_qty) and req_qty <= 0:
                results.append(ReturnFlowValidationResult(
                    order_id=str(req_id),
                    validation_type='procurement_rule_compliance',
                    severity='critical',
                    description=f"Invalid requisition quantity: {req_qty}",
                    field_name='requisition_quantity',
                    kinaxis_planned_value=req_qty,
                    sap_expected_value=">0",
                    variance=abs(req_qty),
                    auto_fixable=False,
                    recommended_action="Fix requisition quantity in Kinaxis",
                    business_impact="Cannot procure with invalid quantity",
                    execution_risk="Requisition will be rejected"
                ))
            
            # Validate delivery date is reasonable
            if pd.notna(delivery_date):
                days_from_now = (delivery_date - datetime.now()).days
                if days_from_now < 1:  # Delivery date in the past or today
                    results.append(ReturnFlowValidationResult(
                        order_id=str(req_id),
                        validation_type='lead_time_validation',
                        severity='high',
                        description=f"Required delivery date is too soon: {days_from_now} days",
                        field_name='required_date',
                        kinaxis_planned_value=delivery_date,
                        sap_expected_value=datetime.now() + timedelta(days=1),
                        variance=abs(days_from_now),
                        auto_fixable=True,
                        recommended_action="Adjust delivery date to allow procurement lead time",
                        business_impact="May require expedited procurement at higher cost",
                        execution_risk="Vendor may not be able to deliver on time"
                    ))
        
        return results
    
    def _validate_material_requirements(self, kinaxis_data: Dict[str, pd.DataFrame],
                                      sap_materials: Optional[pd.DataFrame]) -> List[ReturnFlowValidationResult]:
        """Validate that all required materials exist in SAP"""
        results = []
        
        # Collect all materials referenced in Kinaxis planning data
        required_materials = set()
        
        for table_name, df in kinaxis_data.items():
            if 'item_id' in df.columns:
                required_materials.update(df['item_id'].dropna().unique())
            elif 'material_id' in df.columns:
                required_materials.update(df['material_id'].dropna().unique())
        
        # Check if materials exist in SAP
        if sap_materials is not None:
            sap_material_numbers = set(sap_materials['material_number'].dropna().unique())
            
            missing_materials = required_materials - sap_material_numbers
            
            for material_id in missing_materials:
                results.append(ReturnFlowValidationResult(
                    order_id=str(material_id),
                    validation_type='material_availability_validation',
                    severity='critical',
                    description=f"Material {material_id} required by Kinaxis plan not found in SAP",
                    field_name='material_number',
                    kinaxis_planned_value=material_id,
                    sap_expected_value="Valid SAP material",
                    variance=1.0,
                    auto_fixable=False,
                    recommended_action="Create material master in SAP or remove from Kinaxis plan",
                    business_impact="Orders referencing this material will fail",
                    execution_risk="All orders for this material will be rejected"
                ))
        
        return results
    
    def _create_return_flow_summary(self, results: List[ReturnFlowValidationResult], 
                                  processing_time: float) -> Dict:
        """Create summary of return flow validation results"""
        total_validations = len(results)
        critical_count = sum(1 for r in results if r.severity == 'critical')
        high_count = sum(1 for r in results if r.severity == 'high')
        auto_fixable_count = sum(1 for r in results if r.auto_fixable)
        
        # Calculate execution readiness score
        execution_readiness = max(0, 100 - (critical_count * 20 + high_count * 10))
        
        # Breakdown by validation type
        type_breakdown = {}
        for result in results:
            validation_type = result.validation_type
            type_breakdown[validation_type] = type_breakdown.get(validation_type, 0) + 1
        
        # Risk assessment
        execution_risk_level = 'LOW'
        if critical_count > 0:
            execution_risk_level = 'CRITICAL'
        elif high_count > 5:
            execution_risk_level = 'HIGH'
        elif high_count > 0:
            execution_risk_level = 'MEDIUM'
        
        return {
            'total_validations': total_validations,
            'critical_issues': critical_count,
            'high_priority_issues': high_count,
            'auto_fixable_count': auto_fixable_count,
            'manual_review_count': total_validations - auto_fixable_count,
            'execution_readiness_score': execution_readiness,
            'execution_risk_level': execution_risk_level,
            'processing_time': round(processing_time, 2),
            'validation_type_breakdown': type_breakdown,
            'recommendation': self._get_execution_recommendation(execution_risk_level, critical_count)
        }
    
    def _get_execution_recommendation(self, risk_level: str, critical_count: int) -> str:
        """Get recommendation based on validation results"""
        if risk_level == 'CRITICAL':
            return f"DO NOT PROCEED - {critical_count} critical issues must be resolved before execution"
        elif risk_level == 'HIGH':
            return "PROCEED WITH CAUTION - High priority issues should be reviewed"
        elif risk_level == 'MEDIUM':
            return "PROCEED - Monitor execution closely"
        else:
            return "PROCEED - All validations passed"
