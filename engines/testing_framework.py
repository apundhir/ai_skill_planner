#!/usr/bin/env python3
"""
Automated Testing Framework for AI Skill Planner
Implements continuous validation, regression testing, and quality assurance
Based on PRD specifications for Milestone 4
"""

import sys
import os
import time
import traceback
import unittest
import numpy as np
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed
import json

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from database.init_db import get_db_connection
from engines.proficiency import ProficiencyCalculator
from engines.gap_analysis import GapAnalysisEngine
from engines.financial import FinancialEngine
from engines.risk_assessment import RiskAssessmentEngine
from engines.recommendations import RecommendationEngine
from engines.validation import ValidationEngine

@dataclass
class TestResult:
    """Individual test result"""
    test_name: str
    status: str  # PASS, FAIL, SKIP, ERROR
    duration_ms: float
    error_message: Optional[str] = None
    assertions: int = 0
    passed_assertions: int = 0

class TestSuite:
    """Base class for test suites"""

    def __init__(self, name: str):
        self.name = name
        self.results: List[TestResult] = []

    def run_test(self, test_func: Callable, test_name: str) -> TestResult:
        """Run a single test function"""
        start_time = time.time()

        try:
            result = test_func()
            duration = (time.time() - start_time) * 1000

            if isinstance(result, dict) and 'assertions' in result:
                return TestResult(
                    test_name=test_name,
                    status='PASS' if result.get('passed', True) else 'FAIL',
                    duration_ms=duration,
                    error_message=result.get('error'),
                    assertions=result.get('assertions', 1),
                    passed_assertions=result.get('passed_assertions', 1 if result.get('passed', True) else 0)
                )
            else:
                return TestResult(
                    test_name=test_name,
                    status='PASS',
                    duration_ms=duration,
                    assertions=1,
                    passed_assertions=1
                )

        except Exception as e:
            duration = (time.time() - start_time) * 1000
            return TestResult(
                test_name=test_name,
                status='ERROR',
                duration_ms=duration,
                error_message=str(e),
                assertions=1,
                passed_assertions=0
            )

class ProficiencyTestSuite(TestSuite):
    """Test suite for proficiency calculation engine"""

    def __init__(self):
        super().__init__("ProficiencyEngine")
        self.proficiency_calc = ProficiencyCalculator()

    def test_evidence_aggregation(self) -> Dict[str, Any]:
        """Test evidence aggregation correctness"""
        assertions = 0
        passed = 0

        # Test case 1: Basic evidence aggregation
        evidence_list = [
            {'evidence_type': 'certification', 'description': 'Test cert'},
            {'evidence_type': 'project_outcome', 'description': 'Successful project'},
            {'evidence_type': 'peer_review', 'description': 'Positive review'}
        ]

        score = self.proficiency_calc._calculate_evidence_score_safe(evidence_list)
        assertions += 1
        if 0.0 <= score <= 1.0:
            passed += 1
        else:
            return {'passed': False, 'error': f'Evidence score {score} out of bounds', 'assertions': assertions, 'passed_assertions': passed}

        # Test case 2: Empty evidence list
        empty_score = self.proficiency_calc._calculate_evidence_score_safe([])
        assertions += 1
        if empty_score == 0.0:
            passed += 1
        else:
            return {'passed': False, 'error': f'Empty evidence score should be 0.0, got {empty_score}', 'assertions': assertions, 'passed_assertions': passed}

        # Test case 3: Evidence score monotonicity (more evidence = higher or equal score)
        single_evidence = [evidence_list[0]]
        single_score = self.proficiency_calc._calculate_evidence_score_safe(single_evidence)
        assertions += 1
        if single_score <= score:
            passed += 1
        else:
            return {'passed': False, 'error': f'Evidence aggregation not monotonic: {single_score} > {score}', 'assertions': assertions, 'passed_assertions': passed}

        return {'passed': True, 'assertions': assertions, 'passed_assertions': passed}

    def test_recency_decay(self) -> Dict[str, Any]:
        """Test recency decay calculations"""
        assertions = 0
        passed = 0

        # Test case 1: Decay factor calculation
        decay_rate = 0.15  # 15% annual decay
        months_since = 6   # 6 months ago

        decay_factor = self.proficiency_calc._calculate_recency_decay(decay_rate, months_since)

        assertions += 1
        if 0.0 <= decay_factor <= 1.0:
            passed += 1
        else:
            return {'passed': False, 'error': f'Decay factor {decay_factor} out of bounds', 'assertions': assertions, 'passed_assertions': passed}

        # Test case 2: Recent usage should have minimal decay
        recent_decay = self.proficiency_calc._calculate_recency_decay(decay_rate, 1)  # 1 month ago
        assertions += 1
        if recent_decay > 0.98:  # Should be close to 1.0
            passed += 1
        else:
            return {'passed': False, 'error': f'Recent decay too high: {recent_decay}', 'assertions': assertions, 'passed_assertions': passed}

        # Test case 3: Old usage should have significant decay
        old_decay = self.proficiency_calc._calculate_recency_decay(decay_rate, 24)  # 2 years ago
        assertions += 1
        if old_decay < 0.8:  # Should be noticeably less than 1.0
            passed += 1
        else:
            return {'passed': False, 'error': f'Old decay too low: {old_decay}', 'assertions': assertions, 'passed_assertions': passed}

        return {'passed': True, 'assertions': assertions, 'passed_assertions': passed}

    def test_skill_level_updates(self) -> Dict[str, Any]:
        """Test skill level update process"""
        assertions = 0
        passed = 0

        try:
            # Test updating skills for a person
            updated_skills = self.proficiency_calc.update_person_skill_levels('sarah_chen')

            assertions += 1
            if isinstance(updated_skills, list) and len(updated_skills) > 0:
                passed += 1
            else:
                return {'passed': False, 'error': 'No skills updated for test person', 'assertions': assertions, 'passed_assertions': passed}

            # Test skill level bounds
            for skill in updated_skills:
                skill_level = skill.get('effective_level', 0)
                assertions += 1
                if 0.0 <= skill_level <= 5.0:
                    passed += 1
                else:
                    return {'passed': False, 'error': f'Skill level {skill_level} out of bounds for {skill.get("skill_name")}', 'assertions': assertions, 'passed_assertions': passed}

            return {'passed': True, 'assertions': assertions, 'passed_assertions': passed}

        except Exception as e:
            return {'passed': False, 'error': str(e), 'assertions': assertions, 'passed_assertions': passed}

class GapAnalysisTestSuite(TestSuite):
    """Test suite for gap analysis engine"""

    def __init__(self):
        super().__init__("GapAnalysisEngine")
        self.gap_engine = GapAnalysisEngine()

    def test_gap_detection(self) -> Dict[str, Any]:
        """Test gap detection logic"""
        assertions = 0
        passed = 0

        try:
            # Test specific skill gap detection
            gap_result = self.gap_engine.detect_skill_gap(
                'personalization_engine', 'modeling', 'recommendation_systems'
            )

            assertions += 1
            if isinstance(gap_result, dict) and 'gap_analysis' in gap_result:
                passed += 1
            else:
                return {'passed': False, 'error': 'Gap analysis result missing key fields', 'assertions': assertions, 'passed_assertions': passed}

            # Test gap severity classification
            gap_analysis = gap_result['gap_analysis']
            severity = gap_analysis.get('gap_severity', '')
            assertions += 1
            if severity in ['none', 'low', 'medium', 'high', 'critical']:
                passed += 1
            else:
                return {'passed': False, 'error': f'Invalid gap severity: {severity}', 'assertions': assertions, 'passed_assertions': passed}

            # Test coverage ratio bounds
            coverage_ratio = gap_analysis.get('coverage_ratio', -1)
            assertions += 1
            if 0.0 <= coverage_ratio <= 2.0:  # Allow slight over-coverage
                passed += 1
            else:
                return {'passed': False, 'error': f'Coverage ratio {coverage_ratio} out of bounds', 'assertions': assertions, 'passed_assertions': passed}

            return {'passed': True, 'assertions': assertions, 'passed_assertions': passed}

        except Exception as e:
            return {'passed': False, 'error': str(e), 'assertions': assertions, 'passed_assertions': passed}

    def test_project_analysis(self) -> Dict[str, Any]:
        """Test project-level gap analysis"""
        assertions = 0
        passed = 0

        try:
            # Test project gap analysis
            project_result = self.gap_engine.analyze_project_gaps('personalization_engine')

            assertions += 1
            if isinstance(project_result, dict) and 'project_summary' in project_result:
                passed += 1
            else:
                return {'passed': False, 'error': 'Project analysis result missing summary', 'assertions': assertions, 'passed_assertions': passed}

            # Test recommendation field
            recommendation = project_result['project_summary'].get('recommendation', '')
            assertions += 1
            if recommendation in ['go', 'conditional', 'no-go']:
                passed += 1
            else:
                return {'passed': False, 'error': f'Invalid recommendation: {recommendation}', 'assertions': assertions, 'passed_assertions': passed}

            # Test gap counts
            critical_gaps = project_result['project_summary'].get('critical_gaps', -1)
            assertions += 1
            if critical_gaps >= 0:
                passed += 1
            else:
                return {'passed': False, 'error': f'Invalid critical gaps count: {critical_gaps}', 'assertions': assertions, 'passed_assertions': passed}

            return {'passed': True, 'assertions': assertions, 'passed_assertions': passed}

        except Exception as e:
            return {'passed': False, 'error': str(e), 'assertions': assertions, 'passed_assertions': passed}

class FinancialTestSuite(TestSuite):
    """Test suite for financial analysis engine"""

    def __init__(self):
        super().__init__("FinancialEngine")
        self.financial_engine = FinancialEngine()

    def test_npv_calculations(self) -> Dict[str, Any]:
        """Test NPV calculation accuracy"""
        assertions = 0
        passed = 0

        try:
            # Test skill gap NPV analysis
            npv_result = self.financial_engine.calculate_skill_gap_npv(
                'personalization_engine', 'modeling', 'recommendation_systems'
            )

            assertions += 1
            if isinstance(npv_result, dict) and 'recommendation' in npv_result:
                passed += 1
            else:
                return {'passed': False, 'error': 'NPV analysis missing recommendation', 'assertions': assertions, 'passed_assertions': passed}

            # Test NPV value reasonableness
            expected_npv = npv_result['recommendation'].get('expected_npv', 0)
            assertions += 1
            if -1000000 <= expected_npv <= 5000000:  # Reasonable NPV range
                passed += 1
            else:
                return {'passed': False, 'error': f'NPV value unreasonable: ${expected_npv:,.0f}', 'assertions': assertions, 'passed_assertions': passed}

            # Test intervention options
            interventions = npv_result.get('intervention_options', {})
            assertions += 1
            if len(interventions) >= 2:  # Should have multiple options
                passed += 1
            else:
                return {'passed': False, 'error': f'Insufficient intervention options: {len(interventions)}', 'assertions': assertions, 'passed_assertions': passed}

            return {'passed': True, 'assertions': assertions, 'passed_assertions': passed}

        except Exception as e:
            return {'passed': False, 'error': str(e), 'assertions': assertions, 'passed_assertions': passed}

    def test_monte_carlo_simulation(self) -> Dict[str, Any]:
        """Test Monte Carlo simulation components"""
        assertions = 0
        passed = 0

        try:
            # Test with simplified intervention
            test_intervention = {
                'strategy': 'hire_fte',
                'upfront_cost': 50000,
                'ongoing_cost_weekly': 2000,
                'lead_time_weeks': 8,
                'success_probability': 0.8
            }

            test_impact = {
                'cost_impact_weekly': 15000
            }

            mc_result = self.financial_engine._monte_carlo_npv(
                test_intervention, test_impact, 52, n_simulations=100
            )

            assertions += 1
            if isinstance(mc_result, dict) and 'mean' in mc_result:
                passed += 1
            else:
                return {'passed': False, 'error': 'Monte Carlo result missing mean', 'assertions': assertions, 'passed_assertions': passed}

            # Test confidence intervals
            conf_low = mc_result.get('confidence_95_low', 0)
            conf_high = mc_result.get('confidence_95_high', 0)
            assertions += 1
            if conf_low <= mc_result['mean'] <= conf_high:
                passed += 1
            else:
                return {'passed': False, 'error': f'Confidence interval inconsistent: {conf_low} <= {mc_result["mean"]} <= {conf_high}', 'assertions': assertions, 'passed_assertions': passed}

            return {'passed': True, 'assertions': assertions, 'passed_assertions': passed}

        except Exception as e:
            return {'passed': False, 'error': str(e), 'assertions': assertions, 'passed_assertions': passed}

class PerformanceTestSuite(TestSuite):
    """Test suite for performance and scalability"""

    def __init__(self):
        super().__init__("Performance")
        self.proficiency_calc = ProficiencyCalculator()
        self.gap_engine = GapAnalysisEngine()
        self.financial_engine = FinancialEngine()

    def test_response_times(self) -> Dict[str, Any]:
        """Test API response time requirements"""
        assertions = 0
        passed = 0

        # Test proficiency calculation performance
        start_time = time.time()
        try:
            self.proficiency_calc.update_person_skill_levels('sarah_chen')
            proficiency_time = time.time() - start_time

            assertions += 1
            if proficiency_time < 2.0:  # Should complete within 2 seconds
                passed += 1
            else:
                return {'passed': False, 'error': f'Proficiency calculation too slow: {proficiency_time:.2f}s', 'assertions': assertions, 'passed_assertions': passed}
        except Exception as e:
            return {'passed': False, 'error': f'Proficiency calculation failed: {e}', 'assertions': assertions, 'passed_assertions': passed}

        # Test gap analysis performance
        start_time = time.time()
        try:
            self.gap_engine.detect_skill_gap('personalization_engine', 'modeling', 'recommendation_systems')
            gap_time = time.time() - start_time

            assertions += 1
            if gap_time < 3.0:  # Should complete within 3 seconds
                passed += 1
            else:
                return {'passed': False, 'error': f'Gap analysis too slow: {gap_time:.2f}s', 'assertions': assertions, 'passed_assertions': passed}
        except Exception as e:
            return {'passed': False, 'error': f'Gap analysis failed: {e}', 'assertions': assertions, 'passed_assertions': passed}

        return {'passed': True, 'assertions': assertions, 'passed_assertions': passed}

    def test_concurrent_access(self) -> Dict[str, Any]:
        """Test system behavior under concurrent load"""
        assertions = 0
        passed = 0

        def concurrent_test():
            try:
                # Simulate concurrent gap analysis
                self.gap_engine.detect_skill_gap('personalization_engine', 'modeling', 'recommendation_systems')
                return True
            except Exception:
                return False

        # Run 5 concurrent operations
        with ThreadPoolExecutor(max_workers=5) as executor:
            futures = [executor.submit(concurrent_test) for _ in range(5)]
            results = [future.result() for future in as_completed(futures)]

        assertions += 1
        successful = sum(results)
        if successful >= 4:  # At least 80% success rate
            passed += 1
        else:
            return {'passed': False, 'error': f'Concurrent access failed: {successful}/5 successful', 'assertions': assertions, 'passed_assertions': passed}

        return {'passed': True, 'assertions': assertions, 'passed_assertions': passed}

class IntegrationTestSuite(TestSuite):
    """Test suite for end-to-end integration testing"""

    def __init__(self):
        super().__init__("Integration")
        self.validation_engine = ValidationEngine()
        self.recommendation_engine = RecommendationEngine()

    def test_end_to_end_workflow(self) -> Dict[str, Any]:
        """Test complete workflow from gap detection to recommendations"""
        assertions = 0
        passed = 0

        try:
            # Step 1: Generate recommendations
            recommendations = self.recommendation_engine.generate_skill_gap_recommendations(
                'personalization_engine', 'modeling', 'recommendation_systems'
            )

            assertions += 1
            if isinstance(recommendations, list) and len(recommendations) > 0:
                passed += 1
            else:
                return {'passed': False, 'error': 'No recommendations generated', 'assertions': assertions, 'passed_assertions': passed}

            # Step 2: Test recommendation structure
            rec = recommendations[0]
            assertions += 1
            if hasattr(rec, 'title') and hasattr(rec, 'priority') and hasattr(rec, 'expected_impact'):
                passed += 1
            else:
                return {'passed': False, 'error': 'Recommendation missing required fields', 'assertions': assertions, 'passed_assertions': passed}

            # Step 3: Test roadmap generation
            roadmap = self.recommendation_engine.create_implementation_roadmap(recommendations)

            assertions += 1
            if isinstance(roadmap, dict) and 'roadmap_phases' in roadmap:
                passed += 1
            else:
                return {'passed': False, 'error': 'Roadmap generation failed', 'assertions': assertions, 'passed_assertions': passed}

            return {'passed': True, 'assertions': assertions, 'passed_assertions': passed}

        except Exception as e:
            return {'passed': False, 'error': str(e), 'assertions': assertions, 'passed_assertions': passed}

    def test_data_consistency(self) -> Dict[str, Any]:
        """Test data consistency across engines"""
        assertions = 0
        passed = 0

        try:
            # Check database connectivity
            conn = get_db_connection()
            cursor = conn.cursor()

            # Test basic data integrity
            cursor.execute("SELECT COUNT(*) as count FROM people")
            people_count = cursor.fetchone()['count']

            cursor.execute("SELECT COUNT(*) as count FROM skills")
            skills_count = cursor.fetchone()['count']

            cursor.execute("SELECT COUNT(*) as count FROM projects")
            projects_count = cursor.fetchone()['count']

            conn.close()

            assertions += 3
            if people_count > 0 and skills_count > 0 and projects_count > 0:
                passed += 3
            else:
                return {'passed': False, 'error': f'Missing core data: {people_count} people, {skills_count} skills, {projects_count} projects', 'assertions': assertions, 'passed_assertions': passed}

            return {'passed': True, 'assertions': assertions, 'passed_assertions': passed}

        except Exception as e:
            return {'passed': False, 'error': str(e), 'assertions': assertions, 'passed_assertions': passed}

class AutomatedTestingFramework:
    """Main testing framework coordinator"""

    def __init__(self):
        self.test_suites = [
            ProficiencyTestSuite(),
            GapAnalysisTestSuite(),
            FinancialTestSuite(),
            PerformanceTestSuite(),
            IntegrationTestSuite()
        ]

        self.validation_engine = ValidationEngine()

    def run_all_tests(self) -> Dict[str, Any]:
        """Run all test suites and return comprehensive results"""

        print("🧪 Running Automated Test Suite...")
        start_time = time.time()

        all_results = {}
        total_tests = 0
        total_passed = 0
        total_assertions = 0
        total_passed_assertions = 0

        for suite in self.test_suites:
            print(f"\n🔧 Running {suite.name} Tests...")
            suite_results = self._run_suite(suite)
            all_results[suite.name] = suite_results

            # Aggregate statistics
            for result in suite_results['test_results']:
                total_tests += 1
                if result.status == 'PASS':
                    total_passed += 1
                total_assertions += result.assertions
                total_passed_assertions += result.passed_assertions

        total_duration = time.time() - start_time

        # Calculate overall score
        overall_score = total_passed / total_tests if total_tests > 0 else 0
        assertion_score = total_passed_assertions / total_assertions if total_assertions > 0 else 0

        return {
            'summary': {
                'total_tests': total_tests,
                'passed_tests': total_passed,
                'test_success_rate': overall_score,
                'total_assertions': total_assertions,
                'passed_assertions': total_passed_assertions,
                'assertion_success_rate': assertion_score,
                'total_duration_seconds': total_duration,
                'status': 'PASS' if overall_score >= 0.85 else 'FAIL'
            },
            'test_suites': all_results,
            'validation_report': self._run_validation_tests(),
            'generated_at': datetime.now().isoformat()
        }

    def _run_suite(self, suite: TestSuite) -> Dict[str, Any]:
        """Run a single test suite"""

        suite_start = time.time()
        test_methods = [method for method in dir(suite) if method.startswith('test_')]

        for method_name in test_methods:
            test_method = getattr(suite, method_name)
            result = suite.run_test(test_method, method_name)
            suite.results.append(result)

            # Print test result
            status_emoji = "✅" if result.status == "PASS" else "❌" if result.status == "FAIL" else "⚠️"
            print(f"   {status_emoji} {method_name}: {result.status} ({result.duration_ms:.0f}ms)")
            if result.error_message and result.status in ['FAIL', 'ERROR']:
                print(f"      Error: {result.error_message}")

        suite_duration = time.time() - suite_start
        passed_tests = len([r for r in suite.results if r.status == 'PASS'])

        return {
            'test_results': suite.results,
            'passed_tests': passed_tests,
            'total_tests': len(suite.results),
            'success_rate': passed_tests / len(suite.results) if suite.results else 0,
            'duration_seconds': suite_duration
        }

    def _run_validation_tests(self) -> Dict[str, Any]:
        """Run validation framework tests"""

        try:
            print("\n📊 Running Validation Framework Tests...")
            validation_report = self.validation_engine.generate_validation_report()

            validation_score = validation_report['validation_summary']['overall_score']
            print(f"   ✅ Validation Score: {validation_score:.1%}")

            return validation_report

        except Exception as e:
            print(f"   ❌ Validation tests failed: {e}")
            return {'error': str(e)}

    def run_regression_tests(self) -> Dict[str, Any]:
        """Run regression tests to ensure no functionality degradation"""

        print("🔄 Running Regression Test Suite...")

        # Store baseline results
        baseline_file = "test_baseline.json"
        current_results = self.run_all_tests()

        if os.path.exists(baseline_file):
            with open(baseline_file, 'r') as f:
                baseline_results = json.load(f)

            regression_analysis = self._analyze_regression(baseline_results, current_results)
            current_results['regression_analysis'] = regression_analysis
        else:
            print("   📝 No baseline found, creating new baseline")

        # Save current results as new baseline
        with open(baseline_file, 'w') as f:
            json.dump(current_results, f, indent=2, default=str)

        return current_results

    def _analyze_regression(self, baseline: Dict[str, Any], current: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze test results for regressions"""

        baseline_score = baseline['summary']['test_success_rate']
        current_score = current['summary']['test_success_rate']

        score_change = current_score - baseline_score

        # Identify specific regressions
        regressions = []
        improvements = []

        for suite_name in baseline['test_suites']:
            if suite_name in current['test_suites']:
                baseline_suite = baseline['test_suites'][suite_name]['success_rate']
                current_suite = current['test_suites'][suite_name]['success_rate']

                if current_suite < baseline_suite - 0.05:  # 5% degradation threshold
                    regressions.append({
                        'suite': suite_name,
                        'baseline_score': baseline_suite,
                        'current_score': current_suite,
                        'degradation': baseline_suite - current_suite
                    })
                elif current_suite > baseline_suite + 0.05:  # 5% improvement threshold
                    improvements.append({
                        'suite': suite_name,
                        'baseline_score': baseline_suite,
                        'current_score': current_suite,
                        'improvement': current_suite - baseline_suite
                    })

        return {
            'overall_score_change': score_change,
            'regressions': regressions,
            'improvements': improvements,
            'regression_status': 'PASS' if score_change >= -0.05 else 'FAIL',
            'analysis_date': datetime.now().isoformat()
        }

    def continuous_testing_loop(self, interval_hours: int = 24):
        """Run continuous testing loop (for production monitoring)"""

        print(f"🔁 Starting continuous testing loop (every {interval_hours} hours)")

        while True:
            try:
                print(f"\n⏰ Running scheduled tests at {datetime.now()}")
                results = self.run_all_tests()

                # Alert on failures
                if results['summary']['status'] == 'FAIL':
                    self._send_failure_alert(results)

                # Store results for trending
                self._store_test_results(results)

                print(f"✅ Testing complete. Next run in {interval_hours} hours.")

            except Exception as e:
                print(f"❌ Testing loop error: {e}")

            # Wait for next iteration
            time.sleep(interval_hours * 3600)

    def _send_failure_alert(self, results: Dict[str, Any]):
        """Send alert on test failures (placeholder for notification system)"""

        print("🚨 ALERT: Test failures detected!")
        print(f"   Overall success rate: {results['summary']['test_success_rate']:.1%}")
        print(f"   Failed tests: {results['summary']['total_tests'] - results['summary']['passed_tests']}")

        # In production, this would integrate with notification systems
        # (email, Slack, PagerDuty, etc.)

    def _store_test_results(self, results: Dict[str, Any]):
        """Store test results for historical analysis"""

        # Store in database for trending and analysis
        conn = get_db_connection()
        cursor = conn.cursor()

        try:
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS test_results (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    test_date TEXT NOT NULL,
                    overall_success_rate REAL,
                    total_tests INTEGER,
                    passed_tests INTEGER,
                    duration_seconds REAL,
                    detailed_results TEXT,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
            """)

            cursor.execute("""
                INSERT INTO test_results
                (test_date, overall_success_rate, total_tests, passed_tests, duration_seconds, detailed_results)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (
                datetime.now().isoformat(),
                results['summary']['test_success_rate'],
                results['summary']['total_tests'],
                results['summary']['passed_tests'],
                results['summary']['total_duration_seconds'],
                json.dumps(results, default=str)
            ))

            conn.commit()

        except Exception as e:
            print(f"Error storing test results: {e}")

        finally:
            conn.close()

# Export the main framework
__all__ = ['AutomatedTestingFramework', 'TestResult', 'TestSuite']