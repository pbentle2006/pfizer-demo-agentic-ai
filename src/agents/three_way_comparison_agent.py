"""
Three-Way Comparison Agent for Pfizer Demo
Advanced agent that compares batch data, streaming data, and reconciled results
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Tuple, Optional
from dataclasses import dataclass
from datetime import datetime, timedelta
import logging


@dataclass
class ThreeWayComparisonResult:
    """Result of three-way comparison analysis"""
    record_id: str
    comparison_type: str
    severity: str
    description: str
    batch_value: Any
    streaming_value: Any
    reconciled_value: Any
    variance_batch_stream: float
    variance_batch_reconciled: float
    variance_stream_reconciled: float
    consistency_score: float
    recommended_action: str
    data_source_reliability: str


class ThreeWayComparisonAgent:
    """
    Advanced agent that performs three-way comparison between:
    1. Batch data (scheduled transfers)
    2. Streaming data (real-time updates)
    3. Reconciled data (post-validation results)
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.comparison_rules = self._initialize_comparison_rules()
        self.tolerance_thresholds = self._initialize_tolerance_thresholds()
    
    def _initialize_comparison_rules(self) -> Dict[str, Dict]:
        """Initialize three-way comparison rules"""
        return {
            'quantity_consistency': {
                'description': 'Quantities should be consistent across all three data sources',
                'severity': 'high',
                'fields': ['quantity', 'production_qty', 'demand_qty', 'stock_level'],
                'tolerance_type': 'percentage',
                'business_impact': 'Inconsistent quantities affect planning accuracy'
            },
            'timing_consistency': {
                'description': 'Dates and timing should align across data sources',
                'severity': 'medium',
                'fields': ['production_date', 'delivery_date', 'due_date'],
                'tolerance_type': 'days',
                'business_impact': 'Timing inconsistencies affect schedule reliability'
            },
            'cost_consistency': {
                'description': 'Cost data should be consistent across sources',
                'severity': 'medium',
                'fields': ['unit_cost', 'total_cost', 'standard_cost'],
                'tolerance_type': 'percentage',
                'business_impact': 'Cost inconsistencies affect financial planning'
            },
            'status_consistency': {
                'description': 'Status fields should be synchronized',
                'severity': 'high',
                'fields': ['order_status', 'production_status', 'material_status'],
                'tolerance_type': 'exact',
                'business_impact': 'Status mismatches cause workflow confusion'
            },
            'master_data_consistency': {
                'description': 'Master data should be identical across sources',
                'severity': 'critical',
                'fields': ['material_number', 'plant_code', 'production_version'],
                'tolerance_type': 'exact',
                'business_impact': 'Master data inconsistencies cause system errors'
            }
        }
    
    def _initialize_tolerance_thresholds(self) -> Dict[str, float]:
        """Initialize tolerance thresholds for comparisons"""
        return {
            'quantity_percentage_tolerance': 0.02,  # 2% tolerance for quantities
            'cost_percentage_tolerance': 0.05,      # 5% tolerance for costs
            'date_tolerance_days': 1,               # 1 day tolerance for dates
            'consistency_score_threshold': 0.85,   # 85% consistency required
            'reliability_threshold': 0.90          # 90% reliability threshold
        }
    
    def perform_three_way_comparison(self, 
                                   batch_data: Dict[str, pd.DataFrame],
                                   streaming_data: Dict[str, pd.DataFrame],
                                   reconciled_data: Dict[str, pd.DataFrame]) -> Tuple[List[ThreeWayComparisonResult], Dict]:
        """
        Perform comprehensive three-way comparison
        
        Args:
            batch_data: Scheduled batch transfer data
            streaming_data: Real-time streaming data
            reconciled_data: Post-validation reconciled data
            
        Returns:
            Tuple of (comparison_results, summary)
        """
        start_time = datetime.now()
        comparison_results = []
        
        # Find common tables across all three data sources
        common_tables = set(batch_data.keys()) & set(streaming_data.keys()) & set(reconciled_data.keys())
        
        for table_name in common_tables:
            batch_df = batch_data[table_name]
            streaming_df = streaming_data[table_name]
            reconciled_df = reconciled_data[table_name]
            
            # Perform comparison for this table
            table_results = self._compare_table_data(
                batch_df, streaming_df, reconciled_df, table_name
            )
            comparison_results.extend(table_results)
        
        # Create summary
        processing_time = (datetime.now() - start_time).total_seconds()
        summary = self._create_comparison_summary(comparison_results, processing_time)
        
        return comparison_results, summary
    
    def _compare_table_data(self, batch_df: pd.DataFrame, streaming_df: pd.DataFrame,
                          reconciled_df: pd.DataFrame, table_name: str) -> List[ThreeWayComparisonResult]:
        """Compare data across three sources for a specific table"""
        results = []
        
        # Align dataframes by common key (assuming first column is key)
        if batch_df.empty or streaming_df.empty or reconciled_df.empty:
            return results
        
        # Find common records across all three sources
        batch_keys = set(batch_df.iloc[:, 0].astype(str))
        streaming_keys = set(streaming_df.iloc[:, 0].astype(str))
        reconciled_keys = set(reconciled_df.iloc[:, 0].astype(str))
        
        common_keys = batch_keys & streaming_keys & reconciled_keys
        
        for key in common_keys:
            # Get records from each source
            batch_record = batch_df[batch_df.iloc[:, 0].astype(str) == key].iloc[0]
            streaming_record = streaming_df[streaming_df.iloc[:, 0].astype(str) == key].iloc[0]
            reconciled_record = reconciled_df[reconciled_df.iloc[:, 0].astype(str) == key].iloc[0]
            
            # Compare each rule
            for rule_name, rule_config in self.comparison_rules.items():
                rule_results = self._apply_comparison_rule(
                    key, batch_record, streaming_record, reconciled_record,
                    rule_name, rule_config, table_name
                )
                results.extend(rule_results)
        
        return results
    
    def _apply_comparison_rule(self, record_id: str, batch_record: pd.Series,
                             streaming_record: pd.Series, reconciled_record: pd.Series,
                             rule_name: str, rule_config: Dict, table_name: str) -> List[ThreeWayComparisonResult]:
        """Apply a specific comparison rule to three records"""
        results = []
        fields = rule_config['fields']
        tolerance_type = rule_config['tolerance_type']
        
        for field in fields:
            if field in batch_record.index and field in streaming_record.index and field in reconciled_record.index:
                batch_value = batch_record[field]
                streaming_value = streaming_record[field]
                reconciled_value = reconciled_record[field]
                
                # Calculate variances
                variance_batch_stream = self._calculate_variance(
                    batch_value, streaming_value, tolerance_type
                )
                variance_batch_reconciled = self._calculate_variance(
                    batch_value, reconciled_value, tolerance_type
                )
                variance_stream_reconciled = self._calculate_variance(
                    streaming_value, reconciled_value, tolerance_type
                )
                
                # Calculate consistency score
                consistency_score = self._calculate_consistency_score(
                    variance_batch_stream, variance_batch_reconciled, variance_stream_reconciled
                )
                
                # Determine if this is a significant inconsistency
                is_inconsistent = self._is_significant_inconsistency(
                    variance_batch_stream, variance_batch_reconciled, 
                    variance_stream_reconciled, tolerance_type
                )
                
                if is_inconsistent:
                    # Determine data source reliability
                    reliability = self._assess_data_source_reliability(
                        batch_value, streaming_value, reconciled_value,
                        variance_batch_stream, variance_batch_reconciled, variance_stream_reconciled
                    )
                    
                    results.append(ThreeWayComparisonResult(
                        record_id=record_id,
                        comparison_type=rule_name,
                        severity=rule_config['severity'],
                        description=f"{rule_config['description']} - Field: {field}",
                        batch_value=batch_value,
                        streaming_value=streaming_value,
                        reconciled_value=reconciled_value,
                        variance_batch_stream=variance_batch_stream,
                        variance_batch_reconciled=variance_batch_reconciled,
                        variance_stream_reconciled=variance_stream_reconciled,
                        consistency_score=consistency_score,
                        recommended_action=self._get_recommended_action(
                            variance_batch_stream, variance_batch_reconciled, 
                            variance_stream_reconciled, reliability
                        ),
                        data_source_reliability=reliability
                    ))
        
        return results
    
    def _calculate_variance(self, value1: Any, value2: Any, tolerance_type: str) -> float:
        """Calculate variance between two values based on tolerance type"""
        if pd.isna(value1) or pd.isna(value2):
            return 1.0 if pd.isna(value1) != pd.isna(value2) else 0.0
        
        if tolerance_type == 'exact':
            return 0.0 if str(value1) == str(value2) else 1.0
        
        elif tolerance_type == 'percentage':
            try:
                val1, val2 = float(value1), float(value2)
                if val1 == 0 and val2 == 0:
                    return 0.0
                elif val1 == 0 or val2 == 0:
                    return 1.0
                else:
                    return abs(val1 - val2) / max(abs(val1), abs(val2))
            except:
                return 1.0
        
        elif tolerance_type == 'days':
            try:
                date1 = pd.to_datetime(value1)
                date2 = pd.to_datetime(value2)
                return abs((date1 - date2).days)
            except:
                return 1.0
        
        return 0.0
    
    def _calculate_consistency_score(self, var_bs: float, var_br: float, var_sr: float) -> float:
        """Calculate overall consistency score (0-1, higher is better)"""
        avg_variance = (var_bs + var_br + var_sr) / 3
        return max(0.0, 1.0 - avg_variance)
    
    def _is_significant_inconsistency(self, var_bs: float, var_br: float, 
                                    var_sr: float, tolerance_type: str) -> bool:
        """Determine if variances represent significant inconsistency"""
        if tolerance_type == 'percentage':
            threshold = self.tolerance_thresholds['quantity_percentage_tolerance']
            return any(var > threshold for var in [var_bs, var_br, var_sr])
        
        elif tolerance_type == 'days':
            threshold = self.tolerance_thresholds['date_tolerance_days']
            return any(var > threshold for var in [var_bs, var_br, var_sr])
        
        elif tolerance_type == 'exact':
            return any(var > 0 for var in [var_bs, var_br, var_sr])
        
        return False
    
    def _assess_data_source_reliability(self, batch_val: Any, streaming_val: Any, 
                                      reconciled_val: Any, var_bs: float, 
                                      var_br: float, var_sr: float) -> str:
        """Assess which data source appears most reliable"""
        # Reconciled data is generally most reliable (post-validation)
        if var_br < var_bs and var_sr < var_bs:
            return "RECONCILED_MOST_RELIABLE"
        
        # If batch and reconciled are close, streaming might be the outlier
        elif var_br < var_sr:
            return "BATCH_RECONCILED_CONSISTENT"
        
        # If streaming and reconciled are close, batch might be outdated
        elif var_sr < var_br:
            return "STREAMING_RECONCILED_CONSISTENT"
        
        # All sources significantly different
        else:
            return "ALL_SOURCES_INCONSISTENT"
    
    def _get_recommended_action(self, var_bs: float, var_br: float, 
                              var_sr: float, reliability: str) -> str:
        """Get recommended action based on variance pattern"""
        if reliability == "RECONCILED_MOST_RELIABLE":
            return "Use reconciled value as authoritative source"
        
        elif reliability == "BATCH_RECONCILED_CONSISTENT":
            return "Investigate streaming data source for errors"
        
        elif reliability == "STREAMING_RECONCILED_CONSISTENT":
            return "Update batch data source - may be outdated"
        
        elif reliability == "ALL_SOURCES_INCONSISTENT":
            return "Manual investigation required - all sources differ significantly"
        
        else:
            return "Review data synchronization processes"
    
    def _create_comparison_summary(self, results: List[ThreeWayComparisonResult], 
                                 processing_time: float) -> Dict:
        """Create summary of three-way comparison results"""
        total_comparisons = len(results)
        critical_count = sum(1 for r in results if r.severity == 'critical')
        high_count = sum(1 for r in results if r.severity == 'high')
        
        # Calculate average consistency score
        avg_consistency = np.mean([r.consistency_score for r in results]) if results else 1.0
        
        # Reliability assessment
        reliability_counts = {}
        for result in results:
            reliability = result.data_source_reliability
            reliability_counts[reliability] = reliability_counts.get(reliability, 0) + 1
        
        # Comparison type breakdown
        type_breakdown = {}
        for result in results:
            comp_type = result.comparison_type
            type_breakdown[comp_type] = type_breakdown.get(comp_type, 0) + 1
        
        # Overall data quality assessment
        if avg_consistency >= self.tolerance_thresholds['consistency_score_threshold']:
            data_quality = "EXCELLENT"
        elif avg_consistency >= 0.70:
            data_quality = "GOOD"
        elif avg_consistency >= 0.50:
            data_quality = "FAIR"
        else:
            data_quality = "POOR"
        
        return {
            'total_comparisons': total_comparisons,
            'critical_inconsistencies': critical_count,
            'high_priority_inconsistencies': high_count,
            'average_consistency_score': round(avg_consistency, 3),
            'data_quality_assessment': data_quality,
            'processing_time': round(processing_time, 2),
            'reliability_breakdown': reliability_counts,
            'comparison_type_breakdown': type_breakdown,
            'synchronization_health': self._assess_synchronization_health(results),
            'recommended_actions': self._get_summary_recommendations(results, avg_consistency)
        }
    
    def _assess_synchronization_health(self, results: List[ThreeWayComparisonResult]) -> str:
        """Assess overall synchronization health"""
        if not results:
            return "HEALTHY"
        
        critical_count = sum(1 for r in results if r.severity == 'critical')
        high_count = sum(1 for r in results if r.severity == 'high')
        
        if critical_count > 0:
            return "CRITICAL_ISSUES"
        elif high_count > len(results) * 0.2:  # More than 20% high priority
            return "SYNCHRONIZATION_PROBLEMS"
        elif high_count > 0:
            return "MINOR_ISSUES"
        else:
            return "HEALTHY"
    
    def _get_summary_recommendations(self, results: List[ThreeWayComparisonResult], 
                                   avg_consistency: float) -> List[str]:
        """Get summary-level recommendations"""
        recommendations = []
        
        if avg_consistency < 0.5:
            recommendations.append("URGENT: Review all data synchronization processes")
        
        # Check for patterns in reliability
        reconciled_issues = sum(1 for r in results if "RECONCILED" not in r.data_source_reliability)
        if reconciled_issues > len(results) * 0.3:
            recommendations.append("Review reconciliation process - may not be working correctly")
        
        streaming_issues = sum(1 for r in results if "STREAMING" in r.data_source_reliability and "INCONSISTENT" in r.data_source_reliability)
        if streaming_issues > len(results) * 0.2:
            recommendations.append("Investigate streaming data pipeline for latency or errors")
        
        batch_issues = sum(1 for r in results if "BATCH" in r.data_source_reliability and "outdated" in r.recommended_action.lower())
        if batch_issues > len(results) * 0.2:
            recommendations.append("Batch data may be stale - review batch processing frequency")
        
        if not recommendations:
            recommendations.append("Data synchronization is healthy - continue monitoring")
        
        return recommendations
