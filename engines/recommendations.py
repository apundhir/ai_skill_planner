#!/usr/bin/env python3
"""
Strategic Recommendation Engine for AI Skill Planner
Generates data-driven strategic recommendations for skill gap management
Based on PRD specifications for Milestone 3
"""

import sys
import os
import math
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime, date, timedelta
from enum import Enum
import sqlite3

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from database.init_db import get_db_connection
from engines.gap_analysis import GapAnalysisEngine
from engines.financial import FinancialEngine
from engines.risk_assessment import RiskAssessmentEngine, RiskSeverity

class RecommendationType(Enum):
    IMMEDIATE_ACTION = "immediate_action"
    STRATEGIC_INVESTMENT = "strategic_investment"
    RISK_MITIGATION = "risk_mitigation"
    OPTIMIZATION = "optimization"
    LONG_TERM_PLANNING = "long_term_planning"

class Priority(Enum):
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"

@dataclass
class Recommendation:
    """Individual strategic recommendation"""
    id: str
    title: str
    type: RecommendationType
    priority: Priority
    description: str
    rationale: str
    expected_impact: Dict[str, Any]
    implementation_plan: List[Dict[str, Any]]
    success_metrics: List[str]
    timeline_weeks: int
    estimated_cost: float
    risk_level: str
    dependencies: List[str]
    alternatives: List[str]

class RecommendationEngine:
    """
    Strategic recommendation engine that analyzes gaps, financial impact,
    and risks to generate actionable recommendations
    """

    def __init__(self):
        self.gap_engine = GapAnalysisEngine()
        self.financial_engine = FinancialEngine()
        self.risk_engine = RiskAssessmentEngine()

        # Recommendation templates and heuristics
        self.RECOMMENDATION_TEMPLATES = {
            'critical_skill_gap': {
                'type': RecommendationType.IMMEDIATE_ACTION,
                'priority': Priority.CRITICAL,
                'timeline_weeks': 4,
                'base_title': "Address Critical Skill Gap: {skill_name}"
            },
            'high_npv_investment': {
                'type': RecommendationType.STRATEGIC_INVESTMENT,
                'priority': Priority.HIGH,
                'timeline_weeks': 16,
                'base_title': "High-NPV Skill Investment: {skill_name}"
            },
            'risk_mitigation': {
                'type': RecommendationType.RISK_MITIGATION,
                'priority': Priority.MEDIUM,
                'timeline_weeks': 8,
                'base_title': "Risk Mitigation: {risk_area}"
            },
            'capability_building': {
                'type': RecommendationType.LONG_TERM_PLANNING,
                'priority': Priority.MEDIUM,
                'timeline_weeks': 26,
                'base_title': "Build Organizational Capability: {capability}"
            },
            'portfolio_optimization': {
                'type': RecommendationType.OPTIMIZATION,
                'priority': Priority.HIGH,
                'timeline_weeks': 12,
                'base_title': "Optimize Portfolio: {focus_area}"
            }
        }

        # Impact quantification factors
        self.IMPACT_FACTORS = {
            'cost_reduction_weekly': 1.0,
            'risk_reduction_score': 0.5,
            'capability_improvement': 0.3,
            'strategic_alignment': 0.2
        }

    def generate_skill_gap_recommendations(self, project_id: str, phase: str,
                                         skill_id: str) -> List[Recommendation]:
        """Generate recommendations for specific skill gap"""

        recommendations = []

        # Get comprehensive analysis
        gap_analysis = self.gap_engine.detect_skill_gap(project_id, phase, skill_id)
        financial_analysis = self.financial_engine.calculate_skill_gap_npv(project_id, phase, skill_id)
        risk_profile = self.risk_engine.assess_skill_gap_risk(project_id, phase, skill_id)

        gap_info = gap_analysis['gap_analysis']
        business_impact = gap_analysis['business_impact']
        best_option = financial_analysis['recommendation']['best_option']

        # Critical gap recommendation
        if gap_info['gap_severity'] == 'critical':
            rec = self._create_critical_gap_recommendation(
                gap_analysis, financial_analysis, risk_profile
            )
            recommendations.append(rec)

        # High NPV investment recommendation
        if financial_analysis['recommendation']['expected_npv'] > 100000:  # $100K+ NPV
            rec = self._create_high_npv_recommendation(
                gap_analysis, financial_analysis, risk_profile
            )
            recommendations.append(rec)

        # Risk mitigation recommendations
        critical_risks = [rf for rf in risk_profile.risk_factors
                         if rf.severity == RiskSeverity.CRITICAL]
        if critical_risks:
            for risk_factor in critical_risks:
                rec = self._create_risk_mitigation_recommendation(
                    risk_factor, gap_analysis, financial_analysis
                )
                recommendations.append(rec)

        # Alternative strategy recommendations
        if len(financial_analysis['intervention_options']) > 1:
            rec = self._create_alternative_strategy_recommendation(
                gap_analysis, financial_analysis, risk_profile
            )
            recommendations.append(rec)

        return recommendations

    def generate_project_recommendations(self, project_id: str) -> List[Recommendation]:
        """Generate comprehensive recommendations for entire project"""

        recommendations = []

        # Get project analysis
        project_analysis = self.gap_engine.analyze_project_gaps(project_id)
        financial_analysis = self.financial_engine.analyze_project_financial_risk(project_id)
        risk_profile = self.risk_engine.assess_project_risk(project_id)

        project_info = project_analysis['project_info']
        project_summary = project_analysis['project_summary']

        # Project-level strategic recommendation
        rec = self._create_project_strategy_recommendation(
            project_analysis, financial_analysis, risk_profile
        )
        recommendations.append(rec)

        # Critical gaps prioritization
        if project_summary['critical_gaps'] > 0:
            rec = self._create_gap_prioritization_recommendation(
                project_analysis, financial_analysis
            )
            recommendations.append(rec)

        # Portfolio optimization
        if financial_analysis['financial_summary']['total_investment_required'] > 500000:
            rec = self._create_portfolio_optimization_recommendation(
                project_analysis, financial_analysis, risk_profile
            )
            recommendations.append(rec)

        # Risk management recommendations
        if risk_profile.overall_risk_score > 7.0:
            rec = self._create_project_risk_management_recommendation(
                risk_profile, project_info
            )
            recommendations.append(rec)

        # Timeline optimization
        payback_weeks = financial_analysis['financial_summary']['payback_period_weeks']
        if payback_weeks > 26:  # > 6 months payback
            rec = self._create_timeline_optimization_recommendation(
                project_analysis, financial_analysis, payback_weeks
            )
            recommendations.append(rec)

        return recommendations

    def generate_organization_recommendations(self) -> List[Recommendation]:
        """Generate organization-wide strategic recommendations"""

        recommendations = []

        # Get organization-wide analysis
        financial_overview = self.financial_engine.get_organization_financial_overview()
        risk_profile = self.risk_engine.assess_organization_risk()

        org_overview = financial_overview['organization_overview']
        exec_recommendations = financial_overview['executive_recommendations']

        # Strategic investment recommendation
        if org_overview['total_expected_npv'] > org_overview['total_investment_required']:
            rec = self._create_strategic_investment_recommendation(
                financial_overview, risk_profile
            )
            recommendations.append(rec)

        # Capability building recommendations
        capability_priorities = exec_recommendations['capability_building_priorities']
        if capability_priorities:
            rec = self._create_capability_building_recommendation(
                capability_priorities, financial_overview
            )
            recommendations.append(rec)

        # Risk management framework
        if risk_profile.overall_risk_score > 6.0:
            rec = self._create_risk_framework_recommendation(
                risk_profile, financial_overview
            )
            recommendations.append(rec)

        # Budget optimization
        budget_allocation = financial_overview['budget_allocation']
        rec = self._create_budget_optimization_recommendation(
            budget_allocation, financial_overview
        )
        recommendations.append(rec)

        # Organizational transformation
        if org_overview['projects_requiring_investment'] > len(financial_overview['project_financials']) * 0.7:
            rec = self._create_transformation_recommendation(
                financial_overview, risk_profile
            )
            recommendations.append(rec)

        return recommendations

    def prioritize_recommendations(self, recommendations: List[Recommendation]) -> List[Recommendation]:
        """Prioritize recommendations using multi-criteria decision analysis"""

        def calculate_priority_score(rec: Recommendation) -> float:
            # Base priority scores
            priority_scores = {
                Priority.CRITICAL: 10.0,
                Priority.HIGH: 7.5,
                Priority.MEDIUM: 5.0,
                Priority.LOW: 2.5
            }

            base_score = priority_scores[rec.priority]

            # Adjust for expected impact
            impact_modifier = 0
            if 'npv' in rec.expected_impact:
                npv = rec.expected_impact['npv']
                impact_modifier += min(3.0, npv / 100000)  # $100K = +3 points

            if 'cost_reduction_weekly' in rec.expected_impact:
                weekly_savings = rec.expected_impact['cost_reduction_weekly']
                impact_modifier += min(2.0, weekly_savings / 10000)  # $10K/week = +2 points

            # Adjust for implementation difficulty
            difficulty_penalty = 0
            if rec.timeline_weeks > 26:
                difficulty_penalty += 1.0
            if rec.estimated_cost > 200000:
                difficulty_penalty += 1.0
            if rec.risk_level in ['high', 'critical']:
                difficulty_penalty += 0.5

            # Adjust for dependencies
            dependency_penalty = len(rec.dependencies) * 0.2

            final_score = base_score + impact_modifier - difficulty_penalty - dependency_penalty
            return max(0, final_score)

        # Sort by priority score
        prioritized = sorted(recommendations, key=calculate_priority_score, reverse=True)

        return prioritized

    def create_implementation_roadmap(self, recommendations: List[Recommendation]) -> Dict[str, Any]:
        """Create implementation roadmap from prioritized recommendations"""

        # Group by timeline and dependencies
        roadmap_phases = {
            'immediate': [],      # 0-4 weeks
            'short_term': [],     # 1-3 months
            'medium_term': [],    # 3-6 months
            'long_term': []       # 6+ months
        }

        for rec in recommendations:
            if rec.timeline_weeks <= 4:
                phase = 'immediate'
            elif rec.timeline_weeks <= 12:
                phase = 'short_term'
            elif rec.timeline_weeks <= 26:
                phase = 'medium_term'
            else:
                phase = 'long_term'

            roadmap_phases[phase].append({
                'recommendation_id': rec.id,
                'title': rec.title,
                'priority': rec.priority.value,
                'timeline_weeks': rec.timeline_weeks,
                'estimated_cost': rec.estimated_cost,
                'expected_impact': rec.expected_impact,
                'dependencies': rec.dependencies
            })

        # Calculate phase totals
        phase_summaries = {}
        for phase, recs in roadmap_phases.items():
            total_cost = sum(r['estimated_cost'] for r in recs)
            total_npv = sum(r['expected_impact'].get('npv', 0) for r in recs)
            phase_summaries[phase] = {
                'recommendation_count': len(recs),
                'total_investment': total_cost,
                'expected_npv': total_npv,
                'roi': total_npv / total_cost if total_cost > 0 else 0
            }

        # Identify critical path
        critical_path = self._identify_critical_path(recommendations)

        return {
            'roadmap_phases': roadmap_phases,
            'phase_summaries': phase_summaries,
            'critical_path': critical_path,
            'total_timeline_weeks': max(r.timeline_weeks for r in recommendations) if recommendations else 0,
            'total_investment': sum(r.estimated_cost for r in recommendations),
            'total_expected_npv': sum(r.expected_impact.get('npv', 0) for r in recommendations),
            'generated_at': datetime.now().isoformat()
        }

    # Internal recommendation generation methods

    def _create_critical_gap_recommendation(self, gap_analysis: Dict[str, Any],
                                          financial_analysis: Dict[str, Any],
                                          risk_profile) -> Recommendation:
        """Create recommendation for critical skill gap"""

        gap_info = gap_analysis['gap_analysis']
        skill_name = gap_analysis.get('skill_name', 'Unknown Skill')
        best_option = financial_analysis['recommendation']['best_option']
        intervention = financial_analysis['intervention_options'][best_option]

        implementation_plan = [
            {'phase': 'Assessment', 'duration_weeks': 1, 'activities': ['Validate gap severity', 'Confirm requirements']},
            {'phase': 'Strategy Selection', 'duration_weeks': 1, 'activities': [f'Implement {best_option} strategy']},
            {'phase': 'Execution', 'duration_weeks': intervention.get('lead_time_weeks', 8), 'activities': ['Execute chosen intervention']},
            {'phase': 'Validation', 'duration_weeks': 2, 'activities': ['Verify gap closure', 'Measure impact']}
        ]

        return Recommendation(
            id=f"critical_gap_{skill_name.lower().replace(' ', '_')}",
            title=f"CRITICAL: Address {skill_name} Gap",
            type=RecommendationType.IMMEDIATE_ACTION,
            priority=Priority.CRITICAL,
            description=f"Immediately address critical skill gap in {skill_name} with {gap_info['expected_gap_fte']:.1f} FTE deficit",
            rationale=f"Gap severity: {gap_info['gap_severity']}, Coverage: {gap_info['coverage_ratio']*100:.0f}%, Weekly impact: ${gap_analysis['business_impact']['cost_impact_weekly']:,.0f}",
            expected_impact={
                'npv': financial_analysis['recommendation']['expected_npv'],
                'cost_reduction_weekly': gap_analysis['business_impact']['cost_impact_weekly'],
                'risk_reduction_score': 8.0,
                'timeline_acceleration_weeks': 4
            },
            implementation_plan=implementation_plan,
            success_metrics=[
                f"Achieve >{gap_info.get('required_level', 3.0)} skill level",
                f"Reduce gap to <0.5 FTE",
                "Project gate status moves to GO",
                f"Weekly cost impact <${gap_analysis['business_impact']['cost_impact_weekly']*0.2:,.0f}"
            ],
            timeline_weeks=max(4, intervention.get('lead_time_weeks', 8)),
            estimated_cost=intervention.get('upfront_cost', 50000),  # Default cost if missing
            risk_level=self._assess_recommendation_risk(risk_profile),
            dependencies=[],
            alternatives=[opt for opt in financial_analysis['intervention_options'].keys() if opt != best_option]
        )

    def _create_high_npv_recommendation(self, gap_analysis: Dict[str, Any],
                                      financial_analysis: Dict[str, Any],
                                      risk_profile) -> Recommendation:
        """Create recommendation for high-NPV investment"""

        skill_name = gap_analysis.get('skill_name', 'Unknown Skill')
        best_option = financial_analysis['recommendation']['best_option']
        npv = financial_analysis['recommendation']['expected_npv']
        roi = financial_analysis['recommendation']['risk_adjusted_roi']

        return Recommendation(
            id=f"high_npv_{skill_name.lower().replace(' ', '_')}",
            title=f"High-ROI Investment: {skill_name}",
            type=RecommendationType.STRATEGIC_INVESTMENT,
            priority=Priority.HIGH,
            description=f"Strategic investment in {skill_name} with ${npv:,.0f} NPV and {roi*100:.0f}% ROI",
            rationale=f"Exceptional financial returns justify immediate investment. Payback: {financial_analysis['recommendation']['payback_period_weeks']} weeks",
            expected_impact={
                'npv': npv,
                'roi_percentage': roi * 100,
                'payback_weeks': financial_analysis['recommendation']['payback_period_weeks'],
                'strategic_value': 'high'
            },
            implementation_plan=self._create_investment_plan(best_option, financial_analysis),
            success_metrics=[
                f"Achieve NPV >${npv*0.8:,.0f}",
                f"Payback within {financial_analysis['recommendation']['payback_period_weeks']*1.2:.0f} weeks",
                f"ROI >{roi*0.8*100:.0f}%"
            ],
            timeline_weeks=16,
            estimated_cost=financial_analysis['intervention_options'][best_option]['total_investment'],
            risk_level=self._assess_recommendation_risk(risk_profile),
            dependencies=['budget_approval', 'stakeholder_alignment'],
            alternatives=list(financial_analysis['intervention_options'].keys())
        )

    def _create_risk_mitigation_recommendation(self, risk_factor, gap_analysis: Dict[str, Any],
                                             financial_analysis: Dict[str, Any]) -> Recommendation:
        """Create risk mitigation recommendation"""

        return Recommendation(
            id=f"risk_mitigation_{risk_factor.name.lower().replace(' ', '_')}",
            title=f"Mitigate Risk: {risk_factor.name}",
            type=RecommendationType.RISK_MITIGATION,
            priority=Priority.HIGH if risk_factor.severity == RiskSeverity.CRITICAL else Priority.MEDIUM,
            description=f"Address {risk_factor.severity.value} risk: {risk_factor.description}",
            rationale=f"Risk probability: {risk_factor.probability*100:.0f}%, Impact: {risk_factor.impact_score}/10, Weekly cost: ${risk_factor.cost_impact_weekly:,.0f}",
            expected_impact={
                'risk_reduction_score': risk_factor.impact_score,
                'cost_avoidance_weekly': risk_factor.cost_impact_weekly,
                'probability_reduction': risk_factor.probability * 0.7  # Assume 70% risk reduction
            },
            implementation_plan=[
                {'phase': 'Planning', 'duration_weeks': 2, 'activities': risk_factor.mitigation_strategies[:2]},
                {'phase': 'Implementation', 'duration_weeks': 4, 'activities': risk_factor.mitigation_strategies[2:4]},
                {'phase': 'Monitoring', 'duration_weeks': 2, 'activities': ['Monitor risk indicators', 'Adjust as needed']}
            ],
            success_metrics=[
                f"Reduce risk probability by 50%",
                f"Reduce impact score to <{risk_factor.impact_score*0.5}",
                "Implement all mitigation strategies"
            ],
            timeline_weeks=8,
            estimated_cost=risk_factor.cost_impact_weekly * 4,  # 4 weeks of impact as investment
            risk_level='medium',
            dependencies=[],
            alternatives=risk_factor.mitigation_strategies
        )

    def _create_project_strategy_recommendation(self, project_analysis: Dict[str, Any],
                                              financial_analysis: Dict[str, Any],
                                              risk_profile) -> Recommendation:
        """Create overall project strategy recommendation"""

        project_info = project_analysis['project_info']
        project_summary = project_analysis['project_summary']
        decision = financial_analysis['strategic_recommendation']['decision']

        # Map decision to recommendation
        strategy_map = {
            'proceed': {
                'title': f"Proceed with Project: {project_info['name']}",
                'priority': Priority.HIGH,
                'description': "Project shows strong business case and manageable risks"
            },
            'proceed_with_caution': {
                'title': f"Conditional Approval: {project_info['name']}",
                'priority': Priority.MEDIUM,
                'description': "Project viable but requires risk mitigation measures"
            },
            'conditional': {
                'title': f"Re-evaluate Project: {project_info['name']}",
                'priority': Priority.MEDIUM,
                'description': "Project requires scope or approach adjustments"
            },
            'defer': {
                'title': f"Defer Project: {project_info['name']}",
                'priority': Priority.LOW,
                'description': "Current business case does not justify investment"
            }
        }

        strategy = strategy_map.get(decision, strategy_map['conditional'])

        return Recommendation(
            id=f"project_strategy_{project_info['name'].lower().replace(' ', '_')}",
            title=strategy['title'],
            type=RecommendationType.STRATEGIC_INVESTMENT,
            priority=strategy['priority'],
            description=strategy['description'],
            rationale=financial_analysis['strategic_recommendation']['rationale'],
            expected_impact={
                'project_value': project_info.get('cost_of_delay_weekly', 0) * 52,
                'total_npv': financial_analysis['financial_summary']['total_expected_npv'],
                'investment_required': financial_analysis['financial_summary']['total_investment_required'],
                'roi': financial_analysis['financial_summary']['portfolio_roi']
            },
            implementation_plan=self._create_project_implementation_plan(decision, project_analysis),
            success_metrics=financial_analysis['strategic_recommendation']['key_success_factors'],
            timeline_weeks=self._calculate_project_timeline(project_info),
            estimated_cost=financial_analysis['financial_summary']['total_investment_required'],
            risk_level=risk_profile.overall_risk_score / 10.0,
            dependencies=['executive_approval', 'budget_allocation'],
            alternatives=financial_analysis['strategic_recommendation']['alternative_strategies']
        )

    def _create_capability_building_recommendation(self, capability_priorities: List[Dict[str, Any]],
                                                 financial_overview: Dict[str, Any]) -> Recommendation:
        """Create organizational capability building recommendation"""

        top_priority = capability_priorities[0] if capability_priorities else {}
        skill_name = top_priority.get('skill_name', 'Core Skills')

        return Recommendation(
            id=f"capability_building_{skill_name.lower().replace(' ', '_')}",
            title=f"Build Organizational Capability: {skill_name}",
            type=RecommendationType.LONG_TERM_PLANNING,
            priority=Priority.HIGH,
            description=f"Strategic capability building in {skill_name} affecting {top_priority.get('projects_affected', 0)} projects",
            rationale=f"Priority score: {top_priority.get('priority_score', 0):.1f}, Total NPV: ${top_priority.get('total_npv', 0):,.0f}",
            expected_impact={
                'capability_improvement': 'high',
                'projects_enabled': top_priority.get('projects_affected', 0),
                'total_npv': top_priority.get('total_npv', 0),
                'strategic_alignment': 'high'
            },
            implementation_plan=[
                {'phase': 'Assessment', 'duration_weeks': 4, 'activities': ['Current state analysis', 'Target state definition']},
                {'phase': 'Strategy Development', 'duration_weeks': 4, 'activities': ['Training program design', 'Hiring plan']},
                {'phase': 'Implementation', 'duration_weeks': 16, 'activities': ['Execute training', 'Strategic hiring']},
                {'phase': 'Evaluation', 'duration_weeks': 2, 'activities': ['Assess progress', 'Adjust approach']}
            ],
            success_metrics=[
                f"Increase average {skill_name} level by 1.0",
                f"Reduce gaps in {top_priority.get('projects_affected', 0)} projects",
                "Achieve organizational capability rating ≥ 3.5"
            ],
            timeline_weeks=26,
            estimated_cost=top_priority.get('total_investment', 200000),
            risk_level='medium',
            dependencies=['leadership_commitment', 'training_budget'],
            alternatives=['external_consulting', 'acquisition', 'partnership']
        )

    def _create_budget_optimization_recommendation(self, budget_allocation: Dict[str, Any],
                                                 financial_overview: Dict[str, Any]) -> Recommendation:
        """Create budget optimization recommendation"""

        scenarios = budget_allocation['budget_scenarios']
        optimal_scenario = budget_allocation.get('recommended_scenario', '100pct_budget')
        optimal_data = scenarios.get(optimal_scenario, {})

        return Recommendation(
            id="budget_optimization",
            title="Optimize Investment Portfolio",
            type=RecommendationType.OPTIMIZATION,
            priority=Priority.HIGH,
            description=f"Optimize budget allocation across {optimal_data.get('projects_count', 0)} projects",
            rationale=f"Recommended scenario achieves ${optimal_data.get('total_npv', 0):,.0f} NPV with {optimal_data.get('utilization_rate', 0)*100:.0f}% budget utilization",
            expected_impact={
                'npv_optimization': optimal_data.get('total_npv', 0),
                'budget_efficiency': optimal_data.get('utilization_rate', 0),
                'projects_enabled': optimal_data.get('projects_count', 0)
            },
            implementation_plan=[
                {'phase': 'Analysis', 'duration_weeks': 2, 'activities': ['Validate NPV calculations', 'Confirm project priorities']},
                {'phase': 'Budget Allocation', 'duration_weeks': 2, 'activities': ['Allocate funds per optimization', 'Secure approvals']},
                {'phase': 'Execution', 'duration_weeks': 8, 'activities': ['Launch prioritized projects', 'Monitor performance']}
            ],
            success_metrics=[
                f"Achieve ≥90% of optimal NPV",
                f"Fund top {optimal_data.get('projects_count', 0)} priority projects",
                "Maintain budget utilization >85%"
            ],
            timeline_weeks=12,
            estimated_cost=optimal_data.get('budget_allocated', 0),
            risk_level='low',
            dependencies=['budget_approval', 'project_readiness'],
            alternatives=list(scenarios.keys())
        )

    # Helper methods

    def _create_investment_plan(self, intervention_type: str, financial_analysis: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Create implementation plan for investment"""

        base_plans = {
            'hire_fte': [
                {'phase': 'Recruitment', 'duration_weeks': 8, 'activities': ['Job posting', 'Candidate screening', 'Interviews']},
                {'phase': 'Onboarding', 'duration_weeks': 4, 'activities': ['Orientation', 'Training', 'Integration']},
                {'phase': 'Ramp-up', 'duration_weeks': 4, 'activities': ['Initial projects', 'Mentoring', 'Skill validation']}
            ],
            'train_internal': [
                {'phase': 'Planning', 'duration_weeks': 2, 'activities': ['Training needs analysis', 'Program design']},
                {'phase': 'Training', 'duration_weeks': 8, 'activities': ['Course delivery', 'Hands-on practice', 'Assessments']},
                {'phase': 'Application', 'duration_weeks': 6, 'activities': ['Project application', 'Mentoring', 'Skill reinforcement']}
            ],
            'contract_specialists': [
                {'phase': 'Sourcing', 'duration_weeks': 2, 'activities': ['Vendor identification', 'RFP process', 'Selection']},
                {'phase': 'Onboarding', 'duration_weeks': 1, 'activities': ['Contract execution', 'Access setup', 'Briefing']},
                {'phase': 'Execution', 'duration_weeks': 12, 'activities': ['Project delivery', 'Knowledge transfer', 'Performance monitoring']}
            ]
        }

        return base_plans.get(intervention_type, [
            {'phase': 'Planning', 'duration_weeks': 2, 'activities': ['Strategy refinement']},
            {'phase': 'Execution', 'duration_weeks': 10, 'activities': ['Implementation']},
            {'phase': 'Validation', 'duration_weeks': 4, 'activities': ['Results assessment']}
        ])

    def _create_project_implementation_plan(self, decision: str, project_analysis: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Create project implementation plan based on decision"""

        if decision == 'proceed':
            return [
                {'phase': 'Launch Preparation', 'duration_weeks': 2, 'activities': ['Team assembly', 'Resource allocation']},
                {'phase': 'Gap Mitigation', 'duration_weeks': 8, 'activities': ['Address critical gaps', 'Build capabilities']},
                {'phase': 'Project Execution', 'duration_weeks': 20, 'activities': ['Deliver project phases', 'Monitor progress']},
                {'phase': 'Validation', 'duration_weeks': 2, 'activities': ['Results assessment', 'Lessons learned']}
            ]
        elif decision == 'conditional':
            return [
                {'phase': 'Re-assessment', 'duration_weeks': 4, 'activities': ['Scope refinement', 'Risk mitigation planning']},
                {'phase': 'Pilot Phase', 'duration_weeks': 8, 'activities': ['Limited scope delivery', 'Proof of concept']},
                {'phase': 'Decision Gate', 'duration_weeks': 1, 'activities': ['Go/no-go decision', 'Full project approval']}
            ]
        else:  # defer
            return [
                {'phase': 'Documentation', 'duration_weeks': 1, 'activities': ['Document decision rationale']},
                {'phase': 'Monitoring', 'duration_weeks': 12, 'activities': ['Monitor conditions', 'Reassess quarterly']}
            ]

    def _assess_recommendation_risk(self, risk_profile) -> str:
        """Assess risk level for recommendation"""
        if risk_profile.overall_risk_score >= 8.0:
            return 'critical'
        elif risk_profile.overall_risk_score >= 6.0:
            return 'high'
        elif risk_profile.overall_risk_score >= 4.0:
            return 'medium'
        else:
            return 'low'

    def _calculate_project_timeline(self, project_info: Dict[str, Any]) -> int:
        """Calculate project timeline in weeks"""
        try:
            start = datetime.fromisoformat(project_info['start_date'])
            end = datetime.fromisoformat(project_info['end_date'])
            return (end - start).days // 7
        except:
            return 26  # Default 6 months

    def _identify_critical_path(self, recommendations: List[Recommendation]) -> List[Dict[str, Any]]:
        """Identify critical path through recommendations"""

        # Build dependency graph
        dependency_map = {}
        for rec in recommendations:
            dependency_map[rec.id] = {
                'recommendation': rec,
                'dependencies': rec.dependencies,
                'timeline_weeks': rec.timeline_weeks
            }

        # Find critical path (simplified - assumes linear dependencies)
        critical_path = []
        remaining_recs = sorted(recommendations, key=lambda x: x.priority.value, reverse=True)

        for rec in remaining_recs:
            if not rec.dependencies or all(dep in [cp['id'] for cp in critical_path] for dep in rec.dependencies):
                critical_path.append({
                    'id': rec.id,
                    'title': rec.title,
                    'timeline_weeks': rec.timeline_weeks,
                    'dependencies': rec.dependencies
                })

        return critical_path

    def _create_gap_prioritization_recommendation(self, project_analysis: Dict[str, Any],
                                                financial_analysis: Dict[str, Any]) -> Recommendation:
        """Create recommendation for gap prioritization"""

        critical_gaps = project_analysis['project_summary']['critical_gaps']
        top_gaps = project_analysis['top_gaps'][:3]  # Top 3 gaps

        return Recommendation(
            id=f"gap_prioritization_{project_analysis['project_info']['name'].lower().replace(' ', '_')}",
            title=f"Prioritize {critical_gaps} Critical Gaps",
            type=RecommendationType.OPTIMIZATION,
            priority=Priority.HIGH,
            description=f"Systematic approach to addressing {critical_gaps} critical skill gaps",
            rationale="Focus resources on highest-impact gaps to maximize project success probability",
            expected_impact={
                'gaps_addressed': critical_gaps,
                'risk_reduction_score': 6.0,
                'project_success_probability': 0.85
            },
            implementation_plan=[
                {'phase': 'Gap Analysis', 'duration_weeks': 2, 'activities': ['Validate gap priorities', 'Resource requirements']},
                {'phase': 'Parallel Execution', 'duration_weeks': 8, 'activities': ['Address top 3 gaps simultaneously']},
                {'phase': 'Integration', 'duration_weeks': 2, 'activities': ['Validate gap closure', 'Team integration']}
            ],
            success_metrics=[
                f"Close {len(top_gaps)} top gaps",
                "Achieve project GO status",
                "Reduce overall project risk score by 50%"
            ],
            timeline_weeks=12,
            estimated_cost=sum(gap.get('cost_estimate', 50000) for gap in top_gaps),
            risk_level='medium',
            dependencies=['resource_allocation'],
            alternatives=['sequential_approach', 'outsourced_gap_filling']
        )

    def _create_portfolio_optimization_recommendation(self, project_analysis: Dict[str, Any],
                                                   financial_analysis: Dict[str, Any],
                                                   risk_profile) -> Recommendation:
        """Create portfolio optimization recommendation"""

        total_investment = financial_analysis['financial_summary']['total_investment_required']
        total_npv = financial_analysis['financial_summary']['total_expected_npv']

        return Recommendation(
            id=f"portfolio_optimization_{project_analysis['project_info']['name'].lower().replace(' ', '_')}",
            title="Optimize Project Portfolio Allocation",
            type=RecommendationType.OPTIMIZATION,
            priority=Priority.HIGH,
            description=f"Optimize ${total_investment:,.0f} investment for maximum NPV",
            rationale=f"Large investment requires optimization. Current NPV: ${total_npv:,.0f}",
            expected_impact={
                'investment_optimization': total_investment * 0.1,  # 10% savings potential
                'npv_improvement': total_npv * 0.15,  # 15% NPV improvement
                'risk_reduction_score': 3.0
            },
            implementation_plan=[
                {'phase': 'Portfolio Analysis', 'duration_weeks': 3, 'activities': ['Investment analysis', 'Risk-return optimization']},
                {'phase': 'Reallocation', 'duration_weeks': 2, 'activities': ['Budget reallocation', 'Timeline adjustment']},
                {'phase': 'Implementation', 'duration_weeks': 16, 'activities': ['Execute optimized plan', 'Monitor performance']}
            ],
            success_metrics=[
                f"Achieve NPV >${total_npv * 1.1:,.0f}",
                "Reduce total investment by 5-10%",
                "Improve risk-adjusted returns"
            ],
            timeline_weeks=21,
            estimated_cost=total_investment * 0.02,  # 2% optimization cost
            risk_level=self._assess_recommendation_risk(risk_profile),
            dependencies=['portfolio_review', 'stakeholder_alignment'],
            alternatives=['status_quo', 'project_deferral', 'scope_reduction']
        )

    def _create_project_risk_management_recommendation(self, risk_profile, project_info: Dict[str, Any]) -> Recommendation:
        """Create project risk management recommendation"""

        return Recommendation(
            id=f"risk_management_{project_info['name'].lower().replace(' ', '_')}",
            title=f"Implement Risk Management: {project_info['name']}",
            type=RecommendationType.RISK_MITIGATION,
            priority=Priority.HIGH,
            description=f"Comprehensive risk management for high-risk project (score: {risk_profile.overall_risk_score:.1f}/10)",
            rationale="High risk score requires systematic risk management approach",
            expected_impact={
                'risk_reduction_score': risk_profile.overall_risk_score * 0.4,  # 40% risk reduction
                'project_success_probability': 0.8,
                'cost_avoidance': project_info.get('cost_of_delay_weekly', 0) * 4
            },
            implementation_plan=[
                {'phase': 'Risk Assessment', 'duration_weeks': 2, 'activities': ['Detailed risk analysis', 'Mitigation planning']},
                {'phase': 'Risk Controls', 'duration_weeks': 4, 'activities': ['Implement controls', 'Monitoring systems']},
                {'phase': 'Ongoing Management', 'duration_weeks': 20, 'activities': ['Regular risk reviews', 'Adaptive management']}
            ],
            success_metrics=[
                f"Reduce risk score to <{risk_profile.overall_risk_score * 0.6:.1f}",
                "Zero critical risk events",
                "Maintain project timeline within 10%"
            ],
            timeline_weeks=26,
            estimated_cost=project_info.get('cost_of_delay_weekly', 0) * 2,  # 2 weeks CoD as investment
            risk_level='medium',  # Risk management reduces its own risk
            dependencies=['risk_management_expertise'],
            alternatives=risk_profile.recommendations[:3]
        )

    def _create_timeline_optimization_recommendation(self, project_analysis: Dict[str, Any],
                                                   financial_analysis: Dict[str, Any],
                                                   payback_weeks: float) -> Recommendation:
        """Create timeline optimization recommendation"""

        return Recommendation(
            id=f"timeline_optimization_{project_analysis['project_info']['name'].lower().replace(' ', '_')}",
            title="Optimize Project Timeline",
            type=RecommendationType.OPTIMIZATION,
            priority=Priority.MEDIUM,
            description=f"Accelerate payback from {payback_weeks:.0f} weeks to <26 weeks",
            rationale="Long payback period increases financial risk and reduces NPV",
            expected_impact={
                'timeline_acceleration_weeks': payback_weeks * 0.3,  # 30% acceleration
                'npv_improvement': financial_analysis['financial_summary']['total_expected_npv'] * 0.1,
                'risk_reduction_score': 2.0
            },
            implementation_plan=[
                {'phase': 'Timeline Analysis', 'duration_weeks': 2, 'activities': ['Critical path analysis', 'Acceleration opportunities']},
                {'phase': 'Resource Optimization', 'duration_weeks': 2, 'activities': ['Resource reallocation', 'Parallel execution']},
                {'phase': 'Execution', 'duration_weeks': 16, 'activities': ['Accelerated delivery', 'Progress monitoring']}
            ],
            success_metrics=[
                f"Achieve payback in <{payback_weeks * 0.8:.0f} weeks",
                "Maintain quality standards",
                "Stay within budget +10%"
            ],
            timeline_weeks=20,
            estimated_cost=financial_analysis['financial_summary']['total_investment_required'] * 0.05,  # 5% acceleration premium
            risk_level='medium',
            dependencies=['resource_availability', 'scope_flexibility'],
            alternatives=['scope_reduction', 'phased_delivery', 'additional_resources']
        )

    def _create_alternative_strategy_recommendation(self, gap_analysis: Dict[str, Any],
                                                  financial_analysis: Dict[str, Any],
                                                  risk_profile) -> Recommendation:
        """Create alternative strategy recommendation"""

        skill_name = gap_analysis.get('skill_name', 'Skill')
        interventions = financial_analysis['intervention_options']
        best_option = financial_analysis['recommendation']['best_option']

        # Find second-best option
        sorted_interventions = sorted(interventions.items(),
                                    key=lambda x: x[1]['expected_npv'],
                                    reverse=True)
        alternative_option = sorted_interventions[1][0] if len(sorted_interventions) > 1 else 'hybrid_approach'

        return Recommendation(
            id=f"alternative_strategy_{skill_name.lower().replace(' ', '_')}",
            title=f"Alternative Strategy: {skill_name}",
            type=RecommendationType.OPTIMIZATION,
            priority=Priority.MEDIUM,
            description=f"Consider {alternative_option} as alternative to {best_option} for {skill_name}",
            rationale=f"Multiple viable options available. Alternative may offer better risk profile or alignment.",
            expected_impact={
                'strategy_flexibility': 'high',
                'npv': interventions[alternative_option]['expected_npv'],
                'risk_diversification': 'medium'
            },
            implementation_plan=[
                {'phase': 'Strategy Comparison', 'duration_weeks': 1, 'activities': ['Detailed option analysis']},
                {'phase': 'Decision', 'duration_weeks': 1, 'activities': ['Stakeholder input', 'Final selection']},
                {'phase': 'Implementation', 'duration_weeks': 12, 'activities': ['Execute chosen strategy']}
            ],
            success_metrics=[
                "Select optimal strategy within 2 weeks",
                f"Achieve ≥80% of best-case NPV",
                "Maintain acceptable risk profile"
            ],
            timeline_weeks=14,
            estimated_cost=interventions[alternative_option]['total_investment'],
            risk_level=self._assess_recommendation_risk(risk_profile),
            dependencies=['decision_framework'],
            alternatives=list(interventions.keys())
        )

    def _create_strategic_investment_recommendation(self, financial_overview: Dict[str, Any],
                                                  risk_profile) -> Recommendation:
        """Create strategic investment recommendation for organization"""

        org_overview = financial_overview['organization_overview']
        total_npv = org_overview['total_expected_npv']
        total_investment = org_overview['total_investment_required']

        return Recommendation(
            id="strategic_investment_organization",
            title="Execute Strategic Investment Plan",
            type=RecommendationType.STRATEGIC_INVESTMENT,
            priority=Priority.CRITICAL,
            description=f"Organization-wide investment of ${total_investment:,.0f} for ${total_npv:,.0f} NPV",
            rationale=f"Exceptional ROI of {org_overview['organization_roi']*100:.0f}% justifies comprehensive investment",
            expected_impact={
                'total_npv': total_npv,
                'roi_percentage': org_overview['organization_roi'] * 100,
                'projects_enabled': org_overview['total_projects_analyzed'],
                'organizational_transformation': 'high'
            },
            implementation_plan=[
                {'phase': 'Investment Strategy', 'duration_weeks': 4, 'activities': ['Finalize investment plan', 'Secure funding']},
                {'phase': 'Portfolio Launch', 'duration_weeks': 8, 'activities': ['Launch priority projects', 'Establish governance']},
                {'phase': 'Execution', 'duration_weeks': 26, 'activities': ['Execute project portfolio', 'Monitor progress']},
                {'phase': 'Evaluation', 'duration_weeks': 4, 'activities': ['Assess results', 'Capture learnings']}
            ],
            success_metrics=[
                f"Achieve NPV ≥${total_npv * 0.8:,.0f}",
                f"Complete {org_overview['projects_requiring_investment']} projects",
                "Organizational capability improvement ≥30%"
            ],
            timeline_weeks=42,
            estimated_cost=total_investment,
            risk_level=self._assess_recommendation_risk(risk_profile),
            dependencies=['board_approval', 'capital_availability', 'organizational_readiness'],
            alternatives=['phased_approach', 'selective_investment', 'external_partnerships']
        )

    def _create_risk_framework_recommendation(self, risk_profile, financial_overview: Dict[str, Any]) -> Recommendation:
        """Create risk management framework recommendation"""

        return Recommendation(
            id="risk_framework_organization",
            title="Implement Enterprise Risk Management",
            type=RecommendationType.RISK_MITIGATION,
            priority=Priority.HIGH,
            description=f"Enterprise risk management for organization-wide risk score: {risk_profile.overall_risk_score:.1f}/10",
            rationale="High organizational risk requires systematic risk management framework",
            expected_impact={
                'risk_reduction_score': risk_profile.overall_risk_score * 0.5,
                'organizational_resilience': 'high',
                'cost_avoidance_annual': financial_overview['organization_overview']['annual_cost_of_inaction'] * 0.1
            },
            implementation_plan=[
                {'phase': 'Framework Design', 'duration_weeks': 6, 'activities': ['Risk governance', 'Policies & procedures']},
                {'phase': 'Implementation', 'duration_weeks': 12, 'activities': ['Risk systems', 'Training program']},
                {'phase': 'Operationalization', 'duration_weeks': 8, 'activities': ['Regular reporting', 'Continuous improvement']}
            ],
            success_metrics=[
                f"Reduce organizational risk to <{risk_profile.overall_risk_score * 0.7:.1f}",
                "100% project risk assessments",
                "Quarterly risk reviews implemented"
            ],
            timeline_weeks=26,
            estimated_cost=financial_overview['organization_overview']['total_investment_required'] * 0.02,  # 2% of total investment
            risk_level='medium',
            dependencies=['leadership_commitment', 'risk_management_expertise'],
            alternatives=risk_profile.recommendations[:5]
        )

    def _create_transformation_recommendation(self, financial_overview: Dict[str, Any],
                                            risk_profile) -> Recommendation:
        """Create organizational transformation recommendation"""

        org_overview = financial_overview['organization_overview']

        return Recommendation(
            id="organizational_transformation",
            title="Organizational Capability Transformation",
            type=RecommendationType.LONG_TERM_PLANNING,
            priority=Priority.HIGH,
            description=f"Comprehensive transformation to address capability gaps across {org_overview['projects_requiring_investment']} projects",
            rationale="Systemic capability gaps require organizational transformation approach",
            expected_impact={
                'organizational_maturity': 'significant_improvement',
                'capability_index': 2.0,  # 2-point improvement
                'projects_success_rate': 0.9,
                'long_term_npv': org_overview['total_expected_npv'] * 1.5  # 50% additional value
            },
            implementation_plan=[
                {'phase': 'Transformation Strategy', 'duration_weeks': 8, 'activities': ['Current state assessment', 'Future state design']},
                {'phase': 'Change Program', 'duration_weeks': 16, 'activities': ['Change management', 'Culture transformation']},
                {'phase': 'Capability Building', 'duration_weeks': 26, 'activities': ['Skills development', 'Process improvement']},
                {'phase': 'Sustainment', 'duration_weeks': 12, 'activities': ['Embed changes', 'Continuous improvement']}
            ],
            success_metrics=[
                "Average skill level improvement ≥1.5 points",
                "Organizational risk score reduction ≥40%",
                "Project success rate ≥85%"
            ],
            timeline_weeks=62,  # ~15 months
            estimated_cost=org_overview['total_investment_required'] * 0.3,  # 30% additional investment
            risk_level=self._assess_recommendation_risk(risk_profile),
            dependencies=['executive_sponsorship', 'change_management_capability', 'sustained_investment'],
            alternatives=['incremental_improvement', 'external_partnerships', 'selective_focus']
        )