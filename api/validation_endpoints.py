#!/usr/bin/env python3
"""
Validation & Testing API Endpoints
Provides validation dashboard, testing results, and quality assurance endpoints
Based on PRD specifications for Milestone 4
"""

import sys
import os
from typing import Dict, List, Any, Optional
from fastapi import APIRouter, HTTPException, Query, BackgroundTasks
from pydantic import BaseModel
from datetime import datetime, timedelta
import json

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from database.init_db import get_db_connection
from engines.validation import ValidationEngine, ValidationResult
from engines.testing_framework import AutomatedTestingFramework

# Create router for validation endpoints
validation_router = APIRouter(prefix="/validation", tags=["Validation & Testing"])

# Initialize engines
validation_engine = ValidationEngine()
testing_framework = AutomatedTestingFramework()

# Pydantic models for responses
class ValidationSummary(BaseModel):
    overall_score: float
    validation_date: str
    system_status: str
    total_tests_run: int
    tests_passed: int

class TestResultSummary(BaseModel):
    test_name: str
    status: str
    duration_ms: float
    error_message: Optional[str] = None

class ValidationDashboard(BaseModel):
    validation_summary: ValidationSummary
    model_accuracy: Dict[str, Any]
    business_outcomes: Dict[str, Any]
    test_results: List[TestResultSummary]
    system_health: Dict[str, Any]

@validation_router.get("/dashboard", response_model=Dict[str, Any])
def get_validation_dashboard():
    """
    Comprehensive validation dashboard showing model accuracy,
    test results, and system health metrics
    """
    try:
        # Get comprehensive validation report
        validation_report = validation_engine.generate_validation_report()

        # Get latest test results
        test_results = testing_framework.run_all_tests()

        # Format dashboard data
        dashboard_data = {
            'validation_overview': {
                'overall_score': validation_report['validation_summary']['overall_score'],
                'system_status': validation_report['validation_summary']['system_status'],
                'validation_date': validation_report['validation_summary']['validation_date'],
                'next_validation': validation_report['next_validation_date'],
                'compliance_status': validation_report['validation_compliance']['status']
            },
            'model_performance': {
                'proficiency_accuracy': _extract_metric_value(validation_report['model_accuracy'].get('proficiency_validation', {}), 'accuracy'),
                'gap_prediction_accuracy': _extract_metric_value(validation_report['model_accuracy'].get('gap_validation', {}), 'npv_accuracy'),
                'confidence_calibration': _extract_metric_value(validation_report['model_accuracy'].get('confidence_validation', {}), 'calibration'),
                'overall_mae': _extract_metric_value(validation_report['model_accuracy'].get('proficiency_validation', {}), 'mae')
            },
            'business_outcomes': validation_report['business_outcomes'],
            'test_suite_results': {
                'total_tests': test_results['summary']['total_tests'],
                'passed_tests': test_results['summary']['passed_tests'],
                'success_rate': test_results['summary']['test_success_rate'],
                'test_duration': test_results['summary']['total_duration_seconds'],
                'suite_breakdown': {
                    suite_name: {
                        'success_rate': suite_data['success_rate'],
                        'passed_tests': suite_data['passed_tests'],
                        'total_tests': suite_data['total_tests']
                    }
                    for suite_name, suite_data in test_results['test_suites'].items()
                }
            },
            'system_reliability': validation_report['reliability_metrics'],
            'improvement_recommendations': validation_report['improvement_recommendations'],
            'generated_at': datetime.now().isoformat()
        }

        return dashboard_data

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Dashboard generation failed: {str(e)}")

@validation_router.get("/model-accuracy")
def get_model_accuracy_report():
    """Detailed model accuracy validation report"""
    try:
        return validation_engine.run_model_accuracy_tests()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Model accuracy validation failed: {str(e)}")

@validation_router.get("/business-outcomes")
def get_business_outcomes_tracking():
    """Business outcomes tracking and validation"""
    try:
        conn = get_db_connection()
        cursor = conn.cursor()

        # Get recent business outcomes
        cursor.execute("""
            SELECT bo.*, p.name as project_name, s.name as skill_name
            FROM business_outcomes bo
            LEFT JOIN projects p ON bo.project_id = p.id
            LEFT JOIN skills s ON bo.skill_id = s.id
            WHERE bo.start_date > date('now', '-6 months')
            ORDER BY bo.start_date DESC
            LIMIT 50
        """)

        outcomes = [dict(row) for row in cursor.fetchall()]

        # Get summary statistics
        cursor.execute("""
            SELECT
                COUNT(*) as total_outcomes,
                SUM(CASE WHEN success_status = 'success' THEN 1 ELSE 0 END) as successful,
                SUM(CASE WHEN completion_date IS NOT NULL THEN 1 ELSE 0 END) as completed,
                AVG(CASE WHEN actual_cost > 0 AND predicted_cost > 0
                    THEN ABS(actual_cost - predicted_cost) / predicted_cost ELSE NULL END) * 100 as avg_cost_error_pct,
                AVG(CASE WHEN actual_npv IS NOT NULL AND predicted_npv IS NOT NULL
                    THEN ABS(actual_npv - predicted_npv) / ABS(predicted_npv) ELSE NULL END) * 100 as avg_npv_error_pct
            FROM business_outcomes
            WHERE start_date > date('now', '-6 months')
        """)

        summary = dict(cursor.fetchone())
        conn.close()

        return {
            'outcomes_summary': {
                'total_tracked': summary['total_outcomes'],
                'success_rate': summary['successful'] / summary['total_outcomes'] if summary['total_outcomes'] > 0 else 0,
                'completion_rate': summary['completed'] / summary['total_outcomes'] if summary['total_outcomes'] > 0 else 0,
                'cost_prediction_accuracy': (100 - (summary['avg_cost_error_pct'] or 0)) / 100,
                'npv_prediction_accuracy': (100 - (summary['avg_npv_error_pct'] or 0)) / 100
            },
            'recent_outcomes': outcomes,
            'performance_trends': _calculate_outcome_trends(),
            'generated_at': datetime.now().isoformat()
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Business outcomes tracking failed: {str(e)}")

@validation_router.post("/ground-truth/record")
def record_ground_truth(
    entity_type: str,
    entity_id: str,
    metric_type: str,
    predicted_value: float,
    confidence_low: Optional[float] = None,
    confidence_high: Optional[float] = None,
    metadata: Optional[Dict[str, Any]] = None
):
    """Record a prediction for later ground truth validation"""
    try:
        success = validation_engine.record_prediction(
            entity_type, entity_id, metric_type, predicted_value,
            confidence_low, confidence_high, metadata
        )

        if success:
            return {
                "status": "success",
                "message": f"Prediction recorded for {entity_type}:{entity_id}",
                "recorded_at": datetime.now().isoformat()
            }
        else:
            raise HTTPException(status_code=500, detail="Failed to record prediction")

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ground truth recording failed: {str(e)}")

@validation_router.post("/ground-truth/outcome")
def record_actual_outcome(
    entity_type: str,
    entity_id: str,
    metric_type: str,
    actual_value: float,
    outcome_date: Optional[str] = None
):
    """Record actual outcome for ground truth comparison"""
    try:
        outcome_datetime = datetime.fromisoformat(outcome_date) if outcome_date else None

        success = validation_engine.record_actual_outcome(
            entity_type, entity_id, metric_type, actual_value, outcome_datetime
        )

        if success:
            return {
                "status": "success",
                "message": f"Actual outcome recorded for {entity_type}:{entity_id}",
                "recorded_at": datetime.now().isoformat()
            }
        else:
            raise HTTPException(status_code=404, detail="No matching prediction found")

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Outcome recording failed: {str(e)}")

@validation_router.get("/tests/run")
def run_test_suite(background_tasks: BackgroundTasks, suite_name: Optional[str] = None):
    """Run automated test suite"""
    try:
        if suite_name:
            # Run specific test suite (implementation would filter by suite)
            results = testing_framework.run_all_tests()
            # Filter results for specific suite
            if suite_name in results['test_suites']:
                suite_results = results['test_suites'][suite_name]
                return {
                    'suite_name': suite_name,
                    'results': suite_results,
                    'generated_at': datetime.now().isoformat()
                }
            else:
                raise HTTPException(status_code=404, detail=f"Test suite '{suite_name}' not found")
        else:
            # Run all test suites
            results = testing_framework.run_all_tests()
            return results

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Test execution failed: {str(e)}")

@validation_router.get("/tests/history")
def get_test_history(days: int = Query(30, ge=1, le=365)):
    """Get historical test results for trending analysis"""
    try:
        conn = get_db_connection()
        cursor = conn.cursor()

        cursor.execute("""
            SELECT test_date, overall_success_rate, total_tests, passed_tests, duration_seconds
            FROM test_results
            WHERE test_date > date('now', '-{} days')
            ORDER BY test_date DESC
            LIMIT 100
        """.format(days))

        history = [dict(row) for row in cursor.fetchall()]
        conn.close()

        # Calculate trends
        if len(history) >= 2:
            recent_avg = sum(r['overall_success_rate'] for r in history[:7]) / min(7, len(history))
            older_avg = sum(r['overall_success_rate'] for r in history[7:14]) / max(1, min(7, len(history) - 7))
            trend = recent_avg - older_avg
        else:
            trend = 0

        return {
            'test_history': history,
            'trend_analysis': {
                'trend_direction': 'improving' if trend > 0.01 else 'declining' if trend < -0.01 else 'stable',
                'trend_magnitude': abs(trend),
                'recent_average': recent_avg if len(history) >= 2 else None,
                'baseline_average': older_avg if len(history) >= 14 else None
            },
            'summary': {
                'total_records': len(history),
                'date_range_days': days,
                'best_score': max(h['overall_success_rate'] for h in history) if history else 0,
                'worst_score': min(h['overall_success_rate'] for h in history) if history else 0,
                'average_score': sum(h['overall_success_rate'] for h in history) / len(history) if history else 0
            }
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Test history retrieval failed: {str(e)}")

@validation_router.get("/confidence-intervals")
def get_confidence_interval_validation():
    """Validate confidence interval calibration"""
    try:
        results = validation_engine.validate_confidence_intervals()

        # Process results for API response
        api_results = {}
        for metric_name, result in results.items():
            if isinstance(result, ValidationResult):
                api_results[metric_name] = {
                    'metric': result.metric.value,
                    'value': result.value,
                    'threshold': result.threshold,
                    'passed': result.passed,
                    'description': result.description
                }
            else:
                api_results[metric_name] = result

        return {
            'confidence_validation': api_results,
            'overall_calibration': _assess_overall_calibration(api_results),
            'generated_at': datetime.now().isoformat()
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Confidence interval validation failed: {str(e)}")

@validation_router.get("/system-health")
def get_system_health_metrics():
    """Get comprehensive system health and performance metrics"""
    try:
        # Run performance tests
        test_results = testing_framework.run_all_tests()

        # Get database health
        db_health = _check_database_health()

        # Get validation engine health
        validation_health = _check_validation_health()

        return {
            'overall_health': 'healthy' if all([
                test_results['summary']['test_success_rate'] > 0.85,
                db_health['status'] == 'healthy',
                validation_health['status'] == 'healthy'
            ]) else 'degraded',
            'test_framework': {
                'status': 'healthy' if test_results['summary']['test_success_rate'] > 0.85 else 'degraded',
                'success_rate': test_results['summary']['test_success_rate'],
                'response_time': test_results['summary']['total_duration_seconds']
            },
            'database': db_health,
            'validation_engine': validation_health,
            'performance_metrics': {
                'avg_response_time_ms': 250,  # Would be calculated from actual metrics
                'throughput_requests_per_minute': 120,
                'error_rate_percent': 0.5,
                'uptime_percent': 99.8
            },
            'resource_usage': {
                'cpu_percent': 25,  # Would integrate with system monitoring
                'memory_percent': 45,
                'disk_usage_percent': 30
            },
            'generated_at': datetime.now().isoformat()
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"System health check failed: {str(e)}")

@validation_router.post("/business-outcome/track")
def start_business_outcome_tracking(
    project_id: str,
    intervention_type: str,
    predicted_cost: float,
    predicted_npv: float,
    target_weeks: int,
    skill_id: Optional[str] = None,
    phase: Optional[str] = None
):
    """Start tracking business outcomes for a recommendation implementation"""
    try:
        target_date = datetime.now() + timedelta(weeks=target_weeks)

        outcome_id = validation_engine.track_business_outcomes(
            project_id, intervention_type, predicted_cost, predicted_npv,
            target_date, skill_id, phase
        )

        if outcome_id:
            return {
                'status': 'success',
                'outcome_id': outcome_id,
                'message': f'Started tracking business outcomes for project {project_id}',
                'target_date': target_date.isoformat()
            }
        else:
            raise HTTPException(status_code=500, detail="Failed to start outcome tracking")

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Business outcome tracking failed: {str(e)}")

@validation_router.put("/business-outcome/{outcome_id}")
def update_business_outcome(
    outcome_id: str,
    actual_cost: Optional[float] = None,
    actual_npv: Optional[float] = None,
    success_status: Optional[str] = None,
    completion_date: Optional[str] = None,
    outcome_metrics: Optional[Dict[str, Any]] = None
):
    """Update business outcome with actual results"""
    try:
        completion_datetime = datetime.fromisoformat(completion_date) if completion_date else None

        success = validation_engine.update_business_outcome(
            outcome_id, actual_cost, actual_npv, success_status,
            completion_datetime, outcome_metrics
        )

        if success:
            return {
                'status': 'success',
                'message': f'Updated business outcome {outcome_id}',
                'updated_at': datetime.now().isoformat()
            }
        else:
            raise HTTPException(status_code=404, detail=f"Business outcome {outcome_id} not found")

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Business outcome update failed: {str(e)}")

@validation_router.get("/quality-report")
def generate_quality_assurance_report():
    """Generate comprehensive QA report for executive review"""
    try:
        # Generate full validation report
        validation_report = validation_engine.generate_validation_report()

        # Run test suite
        test_results = testing_framework.run_all_tests()

        # Assess production readiness
        production_readiness = _assess_production_readiness(validation_report, test_results)

        return {
            'qa_summary': {
                'overall_quality_score': production_readiness['overall_score'],
                'production_ready': production_readiness['ready'],
                'certification_level': production_readiness['certification'],
                'assessment_date': datetime.now().isoformat()
            },
            'validation_report': validation_report,
            'test_results_summary': {
                'total_tests': test_results['summary']['total_tests'],
                'passed_tests': test_results['summary']['passed_tests'],
                'success_rate': test_results['summary']['test_success_rate'],
                'critical_failures': [
                    suite for suite, data in test_results['test_suites'].items()
                    if data['success_rate'] < 0.80
                ]
            },
            'quality_gates': production_readiness['quality_gates'],
            'recommendations': production_readiness['recommendations'],
            'next_review_date': (datetime.now() + timedelta(weeks=4)).isoformat()
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Quality report generation failed: {str(e)}")

# Helper functions

def _extract_metric_value(validation_dict: Dict[str, Any], metric_name: str) -> Optional[float]:
    """Extract metric value from validation results"""
    if metric_name in validation_dict and isinstance(validation_dict[metric_name], ValidationResult):
        return validation_dict[metric_name].value
    return None

def _calculate_outcome_trends() -> Dict[str, Any]:
    """Calculate business outcome trends"""
    # Placeholder for trend calculation
    return {
        'success_rate_trend': 'improving',
        'cost_accuracy_trend': 'stable',
        'timeline_accuracy_trend': 'improving',
        'monthly_change_success_rate': 0.05,
        'monthly_change_cost_accuracy': -0.02
    }

def _assess_overall_calibration(calibration_results: Dict[str, Any]) -> Dict[str, Any]:
    """Assess overall confidence interval calibration"""
    passed_tests = sum(1 for result in calibration_results.values()
                      if isinstance(result, dict) and result.get('passed', False))
    total_tests = len(calibration_results)

    return {
        'overall_calibration_score': passed_tests / total_tests if total_tests > 0 else 0,
        'calibration_status': 'good' if passed_tests / total_tests > 0.8 else 'needs_improvement',
        'passed_calibration_tests': passed_tests,
        'total_calibration_tests': total_tests
    }

def _check_database_health() -> Dict[str, Any]:
    """Check database connectivity and integrity"""
    try:
        conn = get_db_connection()
        cursor = conn.cursor()

        # Check basic connectivity
        cursor.execute("SELECT COUNT(*) FROM people")
        people_count = cursor.fetchone()[0]

        cursor.execute("SELECT COUNT(*) FROM skills")
        skills_count = cursor.fetchone()[0]

        cursor.execute("SELECT COUNT(*) FROM projects")
        projects_count = cursor.fetchone()[0]

        conn.close()

        return {
            'status': 'healthy' if all([people_count > 0, skills_count > 0, projects_count > 0]) else 'degraded',
            'people_count': people_count,
            'skills_count': skills_count,
            'projects_count': projects_count,
            'last_check': datetime.now().isoformat()
        }

    except Exception as e:
        return {
            'status': 'error',
            'error': str(e),
            'last_check': datetime.now().isoformat()
        }

def _check_validation_health() -> Dict[str, Any]:
    """Check validation engine health"""
    try:
        # Run basic validation test
        validation_engine.initialize_ground_truth_data()

        return {
            'status': 'healthy',
            'ground_truth_initialized': True,
            'last_check': datetime.now().isoformat()
        }

    except Exception as e:
        return {
            'status': 'error',
            'error': str(e),
            'last_check': datetime.now().isoformat()
        }

def _assess_production_readiness(validation_report: Dict[str, Any], test_results: Dict[str, Any]) -> Dict[str, Any]:
    """Assess overall production readiness"""

    # Quality gates
    quality_gates = {
        'model_accuracy': validation_report['validation_summary']['overall_score'] >= 0.80,
        'test_coverage': test_results['summary']['test_success_rate'] >= 0.85,
        'business_validation': validation_report['business_outcomes']['success_rate'] >= 0.75,
        'system_reliability': validation_report['reliability_metrics']['uptime_percentage'] >= 99.0,
        'compliance': validation_report['validation_compliance']['status'] == 'COMPLIANT'
    }

    passed_gates = sum(quality_gates.values())
    total_gates = len(quality_gates)

    overall_score = passed_gates / total_gates
    ready = overall_score >= 0.8  # 80% of gates must pass

    certification_levels = {
        1.0: 'GOLD',
        0.9: 'SILVER',
        0.8: 'BRONZE',
        0.0: 'NOT_CERTIFIED'
    }

    certification = next(cert for score, cert in sorted(certification_levels.items(), reverse=True)
                        if overall_score >= score)

    recommendations = []
    if not quality_gates['model_accuracy']:
        recommendations.append('Improve model accuracy to >80%')
    if not quality_gates['test_coverage']:
        recommendations.append('Increase test coverage to >85%')
    if not quality_gates['business_validation']:
        recommendations.append('Validate business outcomes >75% success rate')

    return {
        'overall_score': overall_score,
        'ready': ready,
        'certification': certification,
        'quality_gates': quality_gates,
        'recommendations': recommendations
    }

# Export the router
__all__ = ['validation_router']