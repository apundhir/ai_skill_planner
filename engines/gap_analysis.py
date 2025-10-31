#!/usr/bin/env python3
"""
Gap Analysis Engine with Confidence Bands
Combines ProficiencyCalculator and CapacityModel to detect skill gaps
with uncertainty quantification and priority ranking
"""

import sys
import os
from datetime import datetime, date
from typing import Dict, List, Any, Optional, Tuple
import sqlite3

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from database.init_db import get_db_connection
from engines.proficiency import ProficiencyCalculator
from engines.capacity import CapacityModel


class GapAnalysisEngine:
    """
    Comprehensive gap analysis engine that detects, quantifies, and prioritizes
    skill gaps with confidence intervals and business impact assessment
    """

    def __init__(self, database: Optional[str] = None):
        """Initialize the gap analysis engine"""
        self.database = database
        self.proficiency_calc = ProficiencyCalculator(database)
        self.capacity_model = CapacityModel(database)

    def detect_skill_gap(self, project_id: str, phase: str, skill_id: str,
                        conn: Optional[sqlite3.Connection] = None) -> Dict[str, Any]:
        """
        Detect and quantify a specific skill gap with confidence bands

        Returns:
            Comprehensive gap analysis with uncertainty quantification
        """
        should_close_conn = conn is None
        if conn is None:
            conn = get_db_connection(self.database)

        try:
            cursor = conn.cursor()

            # Get basic capacity analysis
            coverage = self.capacity_model.calculate_coverage(project_id, phase, skill_id, conn)

            if not coverage['requirement']:
                return self._empty_gap_result(project_id, phase, skill_id)

            # Get skill and project information
            cursor.execute("""
                SELECT s.name, s.category, s.decay_rate,
                       p.name as project_name, p.cost_of_delay_weekly
                FROM skills s
                CROSS JOIN projects p
                WHERE s.id = ? AND p.id = ?
            """, (skill_id, project_id))

            info = dict(cursor.fetchone())

            # Calculate confidence-adjusted gap metrics
            contributors = coverage['contributors']
            confidence_adjusted_supply = 0.0
            supply_uncertainty = 0.0
            contributor_details = []

            for contrib in contributors:
                person_id = contrib['person_id']

                # Get confidence intervals for this person's skill
                cursor.execute("""
                    SELECT effective_level, confidence_low, confidence_high
                    FROM person_skills
                    WHERE person_id = ? AND skill_id = ?
                """, (person_id, skill_id))

                skill_conf = cursor.fetchone()
                if skill_conf:
                    skill_conf = dict(skill_conf)

                    # Use confidence intervals to adjust supply calculation
                    level = skill_conf.get('effective_level', contrib['person_level'])
                    conf_low = skill_conf.get('confidence_low', level * 0.8)
                    conf_high = skill_conf.get('confidence_high', level * 1.2)

                    # Conservative estimate uses lower confidence bound
                    conservative_level = max(conf_low, coverage['requirement']['min_level'])
                    conservative_match = min(conservative_level / coverage['requirement']['required_level'], 1.0)

                    # Optimistic estimate uses upper confidence bound
                    optimistic_level = min(conf_high, 5.0)
                    optimistic_match = min(optimistic_level / coverage['requirement']['required_level'], 1.0)

                    # Conservative supply contribution
                    conservative_supply = (
                        conservative_match * contrib['availability'] * contrib['fte']
                    )

                    # Optimistic supply contribution
                    optimistic_supply = (
                        optimistic_match * contrib['availability'] * contrib['fte']
                    )

                    # Track uncertainty
                    uncertainty = optimistic_supply - conservative_supply
                    supply_uncertainty += uncertainty ** 2  # Variance addition

                    confidence_adjusted_supply += conservative_supply

                    contributor_details.append({
                        'person_id': person_id,
                        'person_name': contrib['person_name'],
                        'skill_level': level,
                        'confidence_range': [conf_low, conf_high],
                        'conservative_contribution': round(conservative_supply, 2),
                        'optimistic_contribution': round(optimistic_supply, 2),
                        'uncertainty': round(uncertainty, 2)
                    })

            # Overall supply uncertainty (standard deviation)
            supply_std_dev = supply_uncertainty ** 0.5

            # Gap calculations with confidence bands
            demand = coverage['total_demand_fte']
            conservative_gap = max(0, demand - confidence_adjusted_supply)
            optimistic_gap = max(0, demand - (confidence_adjusted_supply + 2 * supply_std_dev))

            # Expected gap (mean estimate)
            expected_gap = max(0, demand - coverage['total_supply_fte'])

            # Gap severity classification
            gap_severity = self._classify_gap_severity(
                coverage['coverage_ratio'], coverage['bus_factor'],
                coverage['requirement']['criticality']
            )

            # Business impact calculation
            impact = self._calculate_business_impact(
                conservative_gap, info['cost_of_delay_weekly'],
                coverage['requirement']['criticality']
            )

            return {
                'project_id': project_id,
                'project_name': info['project_name'],
                'phase': phase,
                'skill_id': skill_id,
                'skill_name': info['name'],
                'skill_category': info['category'],
                'requirement': coverage['requirement'],
                'supply_analysis': {
                    'total_supply_fte': coverage['total_supply_fte'],
                    'confidence_adjusted_supply': round(confidence_adjusted_supply, 2),
                    'supply_uncertainty': round(supply_std_dev, 2),
                    'contributors': contributor_details
                },
                'gap_analysis': {
                    'expected_gap_fte': round(expected_gap, 2),
                    'conservative_gap_fte': round(conservative_gap, 2),
                    'optimistic_gap_fte': round(optimistic_gap, 2),
                    'coverage_ratio': coverage['coverage_ratio'],
                    'bus_factor': coverage['bus_factor'],
                    'gap_severity': gap_severity
                },
                'business_impact': impact,
                'confidence_metrics': {
                    'supply_confidence': 1.0 - (supply_std_dev / max(confidence_adjusted_supply, 0.1)),
                    'gap_uncertainty_range': [round(optimistic_gap, 2), round(conservative_gap, 2)]
                },
                'generated_at': datetime.now().isoformat()
            }

        finally:
            if should_close_conn:
                conn.close()

    def _empty_gap_result(self, project_id: str, phase: str, skill_id: str) -> Dict[str, Any]:
        """Return empty gap result for skills with no requirements"""
        return {
            'project_id': project_id,
            'phase': phase,
            'skill_id': skill_id,
            'requirement': None,
            'gap_analysis': None,
            'business_impact': {'cost_impact_weekly': 0.0, 'priority': 'none'},
            'generated_at': datetime.now().isoformat()
        }

    def _classify_gap_severity(self, coverage_ratio: float, bus_factor: float,
                              criticality: float) -> str:
        """Classify gap severity based on coverage, risk, and criticality"""
        if coverage_ratio >= 1.0 and bus_factor <= 0.25:
            return 'none'
        elif coverage_ratio >= 0.8 and bus_factor <= 0.33 and criticality <= 0.7:
            return 'low'
        elif coverage_ratio >= 0.6 or (coverage_ratio >= 0.5 and criticality <= 0.8):
            return 'medium'
        elif coverage_ratio >= 0.3:
            return 'high'
        else:
            return 'critical'

    def _calculate_business_impact(self, gap_fte: float, cost_of_delay_weekly: float,
                                  criticality: float) -> Dict[str, Any]:
        """Calculate business impact of skill gap"""
        # Base cost impact proportional to gap and criticality
        base_impact = gap_fte * criticality * cost_of_delay_weekly * 0.1

        # Priority classification
        if base_impact > 10000:
            priority = 'critical'
        elif base_impact > 5000:
            priority = 'high'
        elif base_impact > 1000:
            priority = 'medium'
        elif base_impact > 100:
            priority = 'low'
        else:
            priority = 'minimal'

        return {
            'cost_impact_weekly': round(base_impact, 0),
            'priority': priority,
            'criticality_factor': criticality,
            'gap_multiplier': round(gap_fte, 2)
        }

    def analyze_project_gaps(self, project_id: str, phase: Optional[str] = None,
                           conn: Optional[sqlite3.Connection] = None) -> Dict[str, Any]:
        """
        Comprehensive gap analysis for a project (all phases or specific phase)
        """
        should_close_conn = conn is None
        if conn is None:
            conn = get_db_connection(self.database)

        try:
            cursor = conn.cursor()

            # Get phases to analyze
            if phase:
                phases = [phase]
            else:
                cursor.execute("""
                    SELECT phase_name FROM phases
                    WHERE project_id = ?
                    ORDER BY start_date
                """, (project_id,))
                phases = [row['phase_name'] for row in cursor.fetchall()]

            # Get project info
            cursor.execute("""
                SELECT name, cost_of_delay_weekly, risk_tolerance
                FROM projects
                WHERE id = ?
            """, (project_id,))

            project_info = dict(cursor.fetchone())

            # Analyze gaps for each phase
            phase_gap_analyses = {}
            all_gaps = []

            for phase_name in phases:
                # Get all skills required for this phase
                cursor.execute("""
                    SELECT skill_id
                    FROM project_requirements
                    WHERE project_id = ? AND phase_name = ?
                """, (project_id, phase_name))

                skill_ids = [row['skill_id'] for row in cursor.fetchall()]

                # Analyze each skill gap
                phase_gaps = []
                for skill_id in skill_ids:
                    gap_analysis = self.detect_skill_gap(project_id, phase_name, skill_id, conn)
                    if gap_analysis['requirement']:  # Only include skills with requirements
                        phase_gaps.append(gap_analysis)
                        all_gaps.append(gap_analysis)

                # Sort by business impact
                phase_gaps.sort(key=lambda x: x['business_impact']['cost_impact_weekly'], reverse=True)
                phase_gap_analyses[phase_name] = phase_gaps

            # Calculate project-level summary
            summary = self._calculate_project_gap_summary(all_gaps, project_info)

            return {
                'project_id': project_id,
                'project_info': project_info,
                'phases_analyzed': phases,
                'phase_gap_analyses': phase_gap_analyses,
                'project_summary': summary,
                'top_gaps': sorted(all_gaps, key=lambda x: x['business_impact']['cost_impact_weekly'], reverse=True)[:10],
                'generated_at': datetime.now().isoformat()
            }

        finally:
            if should_close_conn:
                conn.close()

    def _calculate_project_gap_summary(self, all_gaps: List[Dict[str, Any]],
                                     project_info: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate project-level gap summary statistics"""
        if not all_gaps:
            return {
                'total_gaps': 0,
                'total_cost_impact': 0.0,
                'critical_gaps': 0,
                'high_gaps': 0,
                'medium_gaps': 0,
                'project_risk_score': 0.0,
                'recommendation': 'proceed'
            }

        # Count gaps by severity
        gap_counts = {'critical': 0, 'high': 0, 'medium': 0, 'low': 0, 'none': 0}
        total_cost_impact = 0.0

        for gap in all_gaps:
            severity = gap['gap_analysis']['gap_severity']
            gap_counts[severity] += 1
            total_cost_impact += gap['business_impact']['cost_impact_weekly']

        # Calculate project risk score (0-1)
        risk_score = (
            gap_counts['critical'] * 1.0 +
            gap_counts['high'] * 0.7 +
            gap_counts['medium'] * 0.4 +
            gap_counts['low'] * 0.1
        ) / max(len(all_gaps), 1)

        # Project recommendation
        recommendation = 'proceed'
        if gap_counts['critical'] > 0 or risk_score > 0.7:
            recommendation = 'no-go'
        elif gap_counts['high'] >= 2 or risk_score > 0.4:
            recommendation = 'conditional'

        return {
            'total_gaps': len(all_gaps),
            'total_cost_impact': round(total_cost_impact, 0),
            'critical_gaps': gap_counts['critical'],
            'high_gaps': gap_counts['high'],
            'medium_gaps': gap_counts['medium'],
            'low_gaps': gap_counts['low'],
            'project_risk_score': round(risk_score, 2),
            'recommendation': recommendation,
            'cost_impact_as_pct_cod': round(total_cost_impact / project_info.get('cost_of_delay_weekly', 1) * 100, 1)
        }

    def get_organization_gap_overview(self,
                                    conn: Optional[sqlite3.Connection] = None) -> Dict[str, Any]:
        """
        Organization-wide gap analysis overview
        """
        should_close_conn = conn is None
        if conn is None:
            conn = get_db_connection(self.database)

        try:
            cursor = conn.cursor()

            # Get all active projects
            cursor.execute("SELECT id, name FROM projects ORDER BY cost_of_delay_weekly DESC")
            projects = [dict(row) for row in cursor.fetchall()]

            # Analyze gaps for each project
            project_summaries = []
            organization_gaps = []

            for project in projects:
                project_gaps = self.analyze_project_gaps(project['id'], conn=conn)
                project_summaries.append({
                    'project_id': project['id'],
                    'project_name': project['name'],
                    **project_gaps['project_summary']
                })

                # Collect all gaps for organization-level analysis
                organization_gaps.extend(project_gaps['top_gaps'])

            # Organization-level statistics
            total_cost_impact = sum(p['total_cost_impact'] for p in project_summaries)
            critical_projects = len([p for p in project_summaries if p['recommendation'] == 'no-go'])

            # Top skills with most gaps across organization
            skill_gap_counts = {}
            for gap in organization_gaps:
                skill_name = gap['skill_name']
                if skill_name not in skill_gap_counts:
                    skill_gap_counts[skill_name] = {
                        'count': 0,
                        'total_impact': 0.0,
                        'category': gap['skill_category']
                    }
                skill_gap_counts[skill_name]['count'] += 1
                skill_gap_counts[skill_name]['total_impact'] += gap['business_impact']['cost_impact_weekly']

            top_gap_skills = sorted(
                skill_gap_counts.items(),
                key=lambda x: x[1]['total_impact'],
                reverse=True
            )[:10]

            return {
                'project_summaries': project_summaries,
                'organization_overview': {
                    'total_projects': len(projects),
                    'critical_projects': critical_projects,
                    'total_cost_impact': round(total_cost_impact, 0),
                    'projects_at_risk_pct': round(critical_projects / len(projects) * 100, 1) if projects else 0
                },
                'top_gap_skills': [
                    {
                        'skill_name': skill,
                        'gap_count': data['count'],
                        'total_impact': round(data['total_impact'], 0),
                        'category': data['category']
                    }
                    for skill, data in top_gap_skills
                ],
                'generated_at': datetime.now().isoformat()
            }

        finally:
            if should_close_conn:
                conn.close()


if __name__ == "__main__":
    # Test the gap analysis engine
    gap_engine = GapAnalysisEngine()

    print("🔍 Testing Gap Analysis Engine...")

    # Test single skill gap detection
    print("\n🎯 Testing specific skill gap detection...")
    gap = gap_engine.detect_skill_gap(
        'personalization_engine', 'modeling', 'recommendation_systems'
    )

    print(f"Recommendation Systems Gap in Personalization Engine (Modeling):")
    if gap['requirement']:
        print(f"  Gap Severity: {gap['gap_analysis']['gap_severity']}")
        print(f"  Expected Gap: {gap['gap_analysis']['expected_gap_fte']} FTE")
        print(f"  Confidence Range: {gap['confidence_metrics']['gap_uncertainty_range']}")
        print(f"  Business Impact: ${gap['business_impact']['cost_impact_weekly']}/week ({gap['business_impact']['priority']} priority)")
        print(f"  Contributors: {len(gap['supply_analysis']['contributors'])} people")

    # Test project-level gap analysis
    print(f"\n📊 Testing project gap analysis...")
    project_gaps = gap_engine.analyze_project_gaps('personalization_engine')

    summary = project_gaps['project_summary']
    print(f"Personalization Engine - Full Gap Analysis:")
    print(f"  Recommendation: {summary['recommendation'].upper()}")
    print(f"  Risk Score: {summary['project_risk_score']}")
    print(f"  Critical Gaps: {summary['critical_gaps']}")
    print(f"  High Gaps: {summary['high_gaps']}")
    print(f"  Total Cost Impact: ${summary['total_cost_impact']}/week ({summary['cost_impact_as_pct_cod']}% of CoD)")

    # Show top 3 gaps
    print(f"\n⚠️  Top Critical Gaps:")
    for i, gap in enumerate(project_gaps['top_gaps'][:3], 1):
        if gap['requirement']:
            print(f"    {i}. {gap['skill_name']} ({gap['phase']}): "
                  f"{gap['gap_analysis']['gap_severity']} severity, "
                  f"${gap['business_impact']['cost_impact_weekly']}/week impact")

    # Test organization overview
    print(f"\n🏢 Testing organization gap overview...")
    org_overview = gap_engine.get_organization_gap_overview()

    org_stats = org_overview['organization_overview']
    print(f"Organization Gap Overview:")
    print(f"  Projects at Risk: {org_stats['projects_at_risk_pct']}% ({org_stats['critical_projects']}/{org_stats['total_projects']})")
    print(f"  Total Cost Impact: ${org_stats['total_cost_impact']}/week")

    print(f"\n🔥 Top Gap Skills Organization-wide:")
    for skill in org_overview['top_gap_skills'][:5]:
        print(f"    {skill['skill_name']}: {skill['gap_count']} gaps, ${skill['total_impact']}/week impact")

    print(f"\n✅ Gap analysis complete!")