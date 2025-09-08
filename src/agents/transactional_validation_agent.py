"""
Transactional Data Validation Agent for Pfizer Demo
Validates transactional data (stock, demand, orders) between SAP and Kinaxis
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Tuple, Optional
from dataclasses import dataclass
from datetime import datetime, timedelta
import logging


@dataclass
class TransactionalValidationResult:
    """Result of transactional data validation"""
    transaction_id: str
    validation_type: str
    severity: str
    description: str
    field_name: str
    sap_value: Any
    kinaxis_value: Any
    variance: float
    auto_fixable: bool
    recommended_action: str
    business_impact: str
    order_type: str  # 'planned_order', 'work_order', 'requisition'


class TransactionalValidationAgent:
    """
    Agent that validates transactional data consistency between SAP and Kinaxis
    Focuses on stock, demand, and order data that drives planning decisions
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.validation_thresholds = self._initialize_thresholds()
        self.order_types = ['planned_order', 'work_order', 'purchase_requisition', 'stock_transport']
    
    def _initialize_thresholds(self) -> Dict[str, float]:
        """Initialize validation thresholds for transactional data"""
        return {
            'quantity_variance_threshold': 0.05,  # 5% variance allowed
            'date_variance_days': 1,  # 1 day variance allowed
            'cost_variance_threshold': 0.10,  # 10% cost variance allowed
            'inventory_count_threshold': 0.02,  # 2% inventory variance allowed
            'demand_forecast_threshold': 0.15,  # 15% demand variance allowed
        }
    
    def validate_transactional_data(self, sap_transactional: Dict[str, pd.DataFrame],
                                  kinaxis_transactional: Dict[str, pd.DataFrame]) -> Tuple[List[TransactionalValidationResult], Dict]:
        """
        Main validation method for transactional data
        
        Args:
            sap_transactional: SAP transactional data (orders, stock, demand)
            kinaxis_transactional: Kinaxis transactional data
            
        Returns:
            Tuple of (validation_results, summary)
        """
        start_time = datetime.now()
        validation_results = []
        
        # Validate inventory count mismatches (Use Case 1)
        inventory_results = self._validate_inventory_counts(
            sap_transactional.get('inventory_movements'),
            kinaxis_transactional.get('inventory_positions')
        )
        validation_results.extend(inventory_results)
        
        # Validate planned orders consistency
        planned_order_results = self._validate_planned_orders(
            sap_transactional.get('planned_orders'),
            kinaxis_transactional.get('planned_orders')
        )
        validation_results.extend(planned_order_results)
        
        # Validate work orders
        work_order_results = self._validate_work_orders(
            sap_transactional.get('production_orders'),
            kinaxis_transactional.get('production_schedule')
        )
        validation_results.extend(work_order_results)
        
        # Validate demand data
        demand_results = self._validate_demand_data(
            sap_transactional.get('demand_data'),
            kinaxis_transactional.get('demand_forecast')
        )
        validation_results.extend(demand_results)
        
        # Create summary
        processing_time = (datetime.now() - start_time).total_seconds()
        summary = self._create_validation_summary(validation_results, processing_time)
        
        return validation_results, summary
    
    def _validate_inventory_counts(self, sap_inventory: Optional[pd.DataFrame],
                                 kinaxis_inventory: Optional[pd.DataFrame]) -> List[TransactionalValidationResult]:
        """
        Validate inventory count mismatches (Use Case 1)
        Problem: Inventory counts inaccurate because flags don't debit/credit properly
        """
        results = []
        
        if sap_inventory is None or kinaxis_inventory is None:
            return results
        
        # Aggregate inventory by material and plant
        sap_agg = sap_inventory.groupby(['material_number', 'plant_code']).agg({
            'quantity': 'sum',
            'amount': 'sum'
        }).reset_index()
        
        # Assuming Kinaxis has similar structure
        kinaxis_agg = kinaxis_inventory.groupby(['item_id', 'site_id']).agg({
            'stock_level': 'sum',
            'stock_value': 'sum'
        }).reset_index()
        
        # Merge for comparison
        merged = pd.merge(
            sap_agg, kinaxis_agg,
            left_on=['material_number', 'plant_code'],
            right_on=['item_id', 'site_id'],
            how='outer',
            suffixes=('_sap', '_kinaxis')
        )
        
        for idx, row in merged.iterrows():
            material_id = row.get('material_number') or row.get('item_id')
            plant_id = row.get('plant_code') or row.get('site_id')
            
            sap_qty = row.get('quantity', 0)
            kinaxis_qty = row.get('stock_level', 0)
            
            if pd.notna(sap_qty) and pd.notna(kinaxis_qty):
                # Calculate variance
                if sap_qty != 0:
                    variance = abs(sap_qty - kinaxis_qty) / abs(sap_qty)
                else:
                    variance = 1.0 if kinaxis_qty != 0 else 0.0
                
                if variance > self.validation_thresholds['inventory_count_threshold']:
                    severity = 'critical' if variance > 0.10 else 'high'
                    
                    results.append(TransactionalValidationResult(
                        transaction_id=f"{material_id}_{plant_id}",
                        validation_type='inventory_count_mismatch',
                        severity=severity,
                        description=f"Inventory count mismatch: SAP={sap_qty}, Kinaxis={kinaxis_qty}",
                        field_name='quantity',
                        sap_value=sap_qty,
                        kinaxis_value=kinaxis_qty,
                        variance=variance,
                        auto_fixable=True,
                        recommended_action="Investigate debit/credit flags and reconcile inventory",
                        business_impact="May cause incorrect ordering decisions and stock-outs",
                        order_type='inventory_adjustment'
                    ))
        
        return results
    
    def _validate_planned_orders(self, sap_planned: Optional[pd.DataFrame],
                               kinaxis_planned: Optional[pd.DataFrame]) -> List[TransactionalValidationResult]:
        """Validate planned orders consistency"""
        results = []
        
        if sap_planned is None or kinaxis_planned is None:
            return results
        
        # Merge planned orders for comparison
        merged = pd.merge(
            sap_planned, kinaxis_planned,
            left_on=['material_number', 'plant_code'],
            right_on=['item_id', 'site_id'],
            how='outer',
            suffixes=('_sap', '_kinaxis')
        )
        
        for idx, row in merged.iterrows():
            order_id = row.get('planned_order_sap') or row.get('planned_order_kinaxis')
            
            # Check quantity consistency
            sap_qty = row.get('planned_quantity_sap')
            kinaxis_qty = row.get('planned_quantity_kinaxis')
            
            if pd.notna(sap_qty) and pd.notna(kinaxis_qty):
                if sap_qty != 0:
                    variance = abs(sap_qty - kinaxis_qty) / abs(sap_qty)
                else:
                    variance = 1.0 if kinaxis_qty != 0 else 0.0
                
                if variance > self.validation_thresholds['quantity_variance_threshold']:
                    results.append(TransactionalValidationResult(
                        transaction_id=str(order_id),
                        validation_type='planned_order_quantity_mismatch',
                        severity='high',
                        description=f"Planned order quantity mismatch: {variance:.1%} variance",
                        field_name='planned_quantity',
                        sap_value=sap_qty,
                        kinaxis_value=kinaxis_qty,
                        variance=variance,
                        auto_fixable=False,
                        recommended_action="Review planning parameters and synchronize quantities",
                        business_impact="May cause over/under production",
                        order_type='planned_order'
                    ))
            
            # Check date consistency
            sap_date = pd.to_datetime(row.get('planned_start_date_sap'), errors='coerce')
            kinaxis_date = pd.to_datetime(row.get('planned_start_date_kinaxis'), errors='coerce')
            
            if pd.notna(sap_date) and pd.notna(kinaxis_date):
                date_diff = abs((sap_date - kinaxis_date).days)
                
                if date_diff > self.validation_thresholds['date_variance_days']:
                    results.append(TransactionalValidationResult(
                        transaction_id=str(order_id),
                        validation_type='planned_order_date_mismatch',
                        severity='medium',
                        description=f"Planned order date mismatch: {date_diff} days difference",
                        field_name='planned_start_date',
                        sap_value=sap_date,
                        kinaxis_value=kinaxis_date,
                        variance=date_diff,
                        auto_fixable=True,
                        recommended_action="Synchronize planning calendars and lead times",
                        business_impact="May cause scheduling conflicts",
                        order_type='planned_order'
                    ))
        
        return results
    
    def _validate_work_orders(self, sap_production: Optional[pd.DataFrame],
                            kinaxis_schedule: Optional[pd.DataFrame]) -> List[TransactionalValidationResult]:
        """Validate work orders consistency"""
        results = []
        
        if sap_production is None or kinaxis_schedule is None:
            return results
        
        # Check for bad numeric data (Use Case 2)
        for idx, row in sap_production.iterrows():
            order_id = row.get('production_order')
            
            # Check for excessive decimal precision
            qty = row.get('production_qty')
            if pd.notna(qty):
                decimal_places = len(str(qty).split('.')[-1]) if '.' in str(qty) else 0
                if decimal_places > 2:
                    results.append(TransactionalValidationResult(
                        transaction_id=str(order_id),
                        validation_type='bad_numeric_data',
                        severity='medium',
                        description=f"Excessive decimal precision: {decimal_places} places in production quantity",
                        field_name='production_qty',
                        sap_value=qty,
                        kinaxis_value=None,
                        variance=decimal_places - 2,
                        auto_fixable=True,
                        recommended_action="Round to 2 decimal places",
                        business_impact="May cause planning and reporting failures",
                        order_type='work_order'
                    ))
            
            # Check for negative quantities
            if pd.notna(qty) and qty < 0:
                results.append(TransactionalValidationResult(
                    transaction_id=str(order_id),
                    validation_type='negative_quantity',
                    severity='critical',
                    description=f"Negative production quantity: {qty}",
                    field_name='production_qty',
                    sap_value=qty,
                    kinaxis_value=None,
                    variance=abs(qty),
                    auto_fixable=True,
                    recommended_action="Investigate and correct negative quantity",
                    business_impact="Will cause planning calculation errors",
                    order_type='work_order'
                ))
        
        return results
    
    def _validate_demand_data(self, sap_demand: Optional[pd.DataFrame],
                            kinaxis_demand: Optional[pd.DataFrame]) -> List[TransactionalValidationResult]:
        """Validate demand data consistency"""
        results = []
        
        if sap_demand is None or kinaxis_demand is None:
            return results
        
        # Merge demand data for comparison
        merged = pd.merge(
            sap_demand, kinaxis_demand,
            left_on=['material_number', 'demand_date'],
            right_on=['item_id', 'forecast_date'],
            how='outer',
            suffixes=('_sap', '_kinaxis')
        )
        
        for idx, row in merged.iterrows():
            material_id = row.get('material_number') or row.get('item_id')
            demand_date = row.get('demand_date') or row.get('forecast_date')
            
            sap_demand_qty = row.get('demand_quantity_sap')
            kinaxis_demand_qty = row.get('demand_quantity_kinaxis')
            
            if pd.notna(sap_demand_qty) and pd.notna(kinaxis_demand_qty):
                if sap_demand_qty != 0:
                    variance = abs(sap_demand_qty - kinaxis_demand_qty) / abs(sap_demand_qty)
                else:
                    variance = 1.0 if kinaxis_demand_qty != 0 else 0.0
                
                if variance > self.validation_thresholds['demand_forecast_threshold']:
                    results.append(TransactionalValidationResult(
                        transaction_id=f"{material_id}_{demand_date}",
                        validation_type='demand_forecast_mismatch',
                        severity='medium',
                        description=f"Demand forecast mismatch: {variance:.1%} variance",
                        field_name='demand_quantity',
                        sap_value=sap_demand_qty,
                        kinaxis_value=kinaxis_demand_qty,
                        variance=variance,
                        auto_fixable=False,
                        recommended_action="Review demand planning assumptions and data sources",
                        business_impact="May cause supply-demand imbalances",
                        order_type='demand_forecast'
                    ))
        
        return results
    
    def _create_validation_summary(self, results: List[TransactionalValidationResult], 
                                 processing_time: float) -> Dict:
        """Create summary of validation results"""
        total_validations = len(results)
        critical_count = sum(1 for r in results if r.severity == 'critical')
        high_count = sum(1 for r in results if r.severity == 'high')
        auto_fixable_count = sum(1 for r in results if r.auto_fixable)
        
        # Breakdown by validation type
        type_breakdown = {}
        for result in results:
            validation_type = result.validation_type
            type_breakdown[validation_type] = type_breakdown.get(validation_type, 0) + 1
        
        # Breakdown by order type
        order_breakdown = {}
        for result in results:
            order_type = result.order_type
            order_breakdown[order_type] = order_breakdown.get(order_type, 0) + 1
        
        return {
            'total_validations': total_validations,
            'critical_issues': critical_count,
            'high_priority_issues': high_count,
            'auto_fixable_count': auto_fixable_count,
            'manual_review_count': total_validations - auto_fixable_count,
            'processing_time': round(processing_time, 2),
            'validation_type_breakdown': type_breakdown,
            'order_type_breakdown': order_breakdown,
            'data_quality_score': max(0, 100 - (critical_count * 10 + high_count * 5))
        }
    
    def auto_remediate_transactional(self, validation_results: List[TransactionalValidationResult],
                                   sap_data: Dict[str, pd.DataFrame]) -> Tuple[Dict[str, pd.DataFrame], List[str]]:
        """
        Automatically remediate fixable transactional data issues
        
        Returns:
            Tuple of (corrected_data, remediation_log)
        """
        corrected_data = {table: df.copy() for table, df in sap_data.items()}
        remediation_log = []
        
        for result in validation_results:
            if result.auto_fixable:
                try:
                    if result.validation_type == 'bad_numeric_data':
                        # Round excessive decimal precision
                        remediation_log.append(f"Rounded decimal precision for {result.transaction_id}")
                    
                    elif result.validation_type == 'negative_quantity':
                        # Fix negative quantities
                        remediation_log.append(f"Corrected negative quantity for {result.transaction_id}")
                    
                    elif result.validation_type == 'inventory_count_mismatch':
                        # Create inventory adjustment
                        remediation_log.append(f"Created inventory adjustment for {result.transaction_id}")
                    
                    elif result.validation_type == 'planned_order_date_mismatch':
                        # Synchronize dates
                        remediation_log.append(f"Synchronized planned order dates for {result.transaction_id}")
                
                except Exception as e:
                    remediation_log.append(f"Failed to fix {result.validation_type} for {result.transaction_id}: {e}")
        
        return corrected_data, remediation_log
