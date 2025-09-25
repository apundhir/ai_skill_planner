#!/usr/bin/env python3
"""
Executive Decision Support API Endpoints
Provides high-level financial analysis, risk assessment, and strategic recommendations
Based on PRD specifications for Milestone 3
"""

import sys
import os
from typing import Dict, List, Any, Optional
from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel
import sqlite3

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from database.init_db import get_db_connection
from engines.financial import FinancialEngine
from engines.risk_assessment import RiskAssessmentEngine
from engines.recommendations import RecommendationEngine

# Create router for executive endpoints
exec_router = APIRouter(prefix="/executive", tags=["Executive Decision Support"])

# Initialize engines
financial_engine = FinancialEngine()
risk_engine = RiskAssessmentEngine()
recommendation_engine = RecommendationEngine()

# Pydantic models for responses
class FinancialSummary(BaseModel):
    total_investment_required: float
    total_expected_npv: float
    organization_roi: float
    payback_period_weeks: float
    projects_analyzed: int

class RiskSummary(BaseModel):
    overall_risk_score: float
    risk_distribution: Dict[str, float]
    high_risk_projects: int
    critical_risks: int
    confidence_level: float

class RecommendationSummary(BaseModel):
    id: str
    title: str
    priority: str
    type: str
    expected_impact: Dict[str, Any]
    timeline_weeks: int
    estimated_cost: float

class ExecutiveDashboard(BaseModel):
    financial_summary: FinancialSummary
    risk_summary: RiskSummary
    top_recommendations: List[RecommendationSummary]
    portfolio_status: Dict[str, Any]
    key_metrics: Dict[str, Any]

@exec_router.get("/dashboard", response_model=ExecutiveDashboard)
def get_executive_dashboard():
    """
    Executive dashboard with comprehensive overview of financial position,
    risks, and strategic recommendations
    """
    try:
        # Get organization-wide financial analysis
        financial_overview = financial_engine.get_organization_financial_overview()
        org_overview = financial_overview['organization_overview']

        # Get risk assessment
        risk_profile = risk_engine.assess_organization_risk()

        # Get strategic recommendations
        recommendations = recommendation_engine.generate_organization_recommendations()
        prioritized_recs = recommendation_engine.prioritize_recommendations(recommendations)

        # Build financial summary
        financial_summary = FinancialSummary(
            total_investment_required=org_overview['total_investment_required'],
            total_expected_npv=org_overview['total_expected_npv'],
            organization_roi=org_overview['organization_roi'],
            payback_period_weeks=financial_overview.get('average_payback_weeks', 26),
            projects_analyzed=org_overview['total_projects_analyzed']
        )

        # Build risk summary
        risk_summary = RiskSummary(
            overall_risk_score=risk_profile.overall_risk_score,
            risk_distribution=risk_profile.risk_distribution,
            high_risk_projects=len([rf for rf in risk_profile.risk_factors if 'high' in rf.name.lower()]),
            critical_risks=len([rf for rf in risk_profile.risk_factors if rf.severity.value == 'critical']),
            confidence_level=risk_profile.confidence_level
        )

        # Build recommendations summary
        top_recommendations = []
        for rec in prioritized_recs[:5]:  # Top 5 recommendations
            top_recommendations.append(RecommendationSummary(
                id=rec.id,
                title=rec.title,
                priority=rec.priority.value,
                type=rec.type.value,
                expected_impact=rec.expected_impact,
                timeline_weeks=rec.timeline_weeks,
                estimated_cost=rec.estimated_cost
            ))

        # Portfolio status
        portfolio_status = {
            'projects_at_risk': len([p for p in financial_overview['project_financials']
                                   if p['risk_assessment']['portfolio_risk_rating'] in ['high', 'critical']]),
            'total_projects': len(financial_overview['project_financials']),
            'avg_project_risk': sum(p['risk_assessment']['total_value_at_risk']
                                  for p in financial_overview['project_financials']) / len(financial_overview['project_financials']) if financial_overview['project_financials'] else 0,
            'budget_utilization': org_overview['total_investment_required'] / (org_overview['annual_cost_of_inaction'] * 0.5) if org_overview['annual_cost_of_inaction'] > 0 else 0
        }

        # Key metrics
        key_metrics = {
            'skill_gap_coverage': 0.7,  # Would calculate from actual data
            'talent_acquisition_rate': 0.8,
            'capability_maturity': 2.6,
            'project_success_probability': 0.75,
            'strategic_alignment_score': 8.2
        }

        return ExecutiveDashboard(
            financial_summary=financial_summary,
            risk_summary=risk_summary,
            top_recommendations=top_recommendations,
            portfolio_status=portfolio_status,
            key_metrics=key_metrics
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Dashboard generation failed: {str(e)}")

@exec_router.get("/financial/overview")
def get_financial_overview():
    """Comprehensive financial analysis for executive decision making"""
    try:
        return financial_engine.get_organization_financial_overview()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Financial analysis failed: {str(e)}")

@exec_router.get("/financial/project/{project_id}")
def get_project_financial_analysis(project_id: str):
    """Detailed financial analysis for specific project"""
    try:
        return financial_engine.analyze_project_financial_risk(project_id)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Project financial analysis failed: {str(e)}")

@exec_router.get("/financial/skill-gap")
def get_skill_gap_financial_analysis(project_id: str, phase: str, skill_id: str):
    """NPV analysis for addressing specific skill gap"""
    try:
        return financial_engine.calculate_skill_gap_npv(project_id, phase, skill_id)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Skill gap financial analysis failed: {str(e)}")

@exec_router.get("/risk/organization")
def get_organization_risk_assessment():
    """Organization-wide risk assessment"""
    try:
        risk_profile = risk_engine.assess_organization_risk()

        # Convert to JSON-serializable format
        return {
            'entity_type': risk_profile.entity_type,
            'entity_id': risk_profile.entity_id,
            'overall_risk_score': risk_profile.overall_risk_score,
            'risk_factors': [
                {
                    'name': rf.name,
                    'category': rf.category.value,
                    'severity': rf.severity.value,
                    'probability': rf.probability,
                    'impact_score': rf.impact_score,
                    'description': rf.description,
                    'mitigation_strategies': rf.mitigation_strategies,
                    'cost_impact_weekly': rf.cost_impact_weekly
                }
                for rf in risk_profile.risk_factors
            ],
            'risk_distribution': risk_profile.risk_distribution,
            'confidence_level': risk_profile.confidence_level,
            'recommendations': risk_profile.recommendations
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Risk assessment failed: {str(e)}")

@exec_router.get("/risk/project/{project_id}")
def get_project_risk_assessment(project_id: str):
    """Project-specific risk assessment"""
    try:
        risk_profile = risk_engine.assess_project_risk(project_id)

        return {
            'entity_type': risk_profile.entity_type,
            'entity_id': risk_profile.entity_id,
            'overall_risk_score': risk_profile.overall_risk_score,
            'risk_factors': [
                {
                    'name': rf.name,
                    'category': rf.category.value,
                    'severity': rf.severity.value,
                    'probability': rf.probability,
                    'impact_score': rf.impact_score,
                    'description': rf.description,
                    'mitigation_strategies': rf.mitigation_strategies,
                    'cost_impact_weekly': rf.cost_impact_weekly
                }
                for rf in risk_profile.risk_factors
            ],
            'risk_distribution': risk_profile.risk_distribution,
            'confidence_level': risk_profile.confidence_level,
            'recommendations': risk_profile.recommendations
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Project risk assessment failed: {str(e)}")

@exec_router.get("/recommendations/organization")
def get_organization_recommendations():
    """Strategic recommendations for organization-wide improvements"""
    try:
        recommendations = recommendation_engine.generate_organization_recommendations()
        prioritized = recommendation_engine.prioritize_recommendations(recommendations)

        # Convert to JSON-serializable format
        return {
            'recommendations': [
                {
                    'id': rec.id,
                    'title': rec.title,
                    'type': rec.type.value,
                    'priority': rec.priority.value,
                    'description': rec.description,
                    'rationale': rec.rationale,
                    'expected_impact': rec.expected_impact,
                    'implementation_plan': rec.implementation_plan,
                    'success_metrics': rec.success_metrics,
                    'timeline_weeks': rec.timeline_weeks,
                    'estimated_cost': rec.estimated_cost,
                    'risk_level': rec.risk_level,
                    'dependencies': rec.dependencies,
                    'alternatives': rec.alternatives
                }
                for rec in prioritized
            ],
            'total_recommendations': len(prioritized),
            'generated_at': recommendations[0].__dict__.get('generated_at', None) if recommendations else None
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Recommendations generation failed: {str(e)}")

@exec_router.get("/recommendations/project/{project_id}")
def get_project_recommendations(project_id: str):
    """Strategic recommendations for specific project"""
    try:
        recommendations = recommendation_engine.generate_project_recommendations(project_id)
        prioritized = recommendation_engine.prioritize_recommendations(recommendations)

        return {
            'project_id': project_id,
            'recommendations': [
                {
                    'id': rec.id,
                    'title': rec.title,
                    'type': rec.type.value,
                    'priority': rec.priority.value,
                    'description': rec.description,
                    'rationale': rec.rationale,
                    'expected_impact': rec.expected_impact,
                    'implementation_plan': rec.implementation_plan,
                    'success_metrics': rec.success_metrics,
                    'timeline_weeks': rec.timeline_weeks,
                    'estimated_cost': rec.estimated_cost,
                    'risk_level': rec.risk_level,
                    'dependencies': rec.dependencies,
                    'alternatives': rec.alternatives
                }
                for rec in prioritized
            ],
            'total_recommendations': len(prioritized)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Project recommendations failed: {str(e)}")

@exec_router.get("/recommendations/skill-gap")
def get_skill_gap_recommendations(project_id: str, phase: str, skill_id: str):
    """Strategic recommendations for addressing specific skill gap"""
    try:
        recommendations = recommendation_engine.generate_skill_gap_recommendations(
            project_id, phase, skill_id
        )
        prioritized = recommendation_engine.prioritize_recommendations(recommendations)

        return {
            'project_id': project_id,
            'phase': phase,
            'skill_id': skill_id,
            'recommendations': [
                {
                    'id': rec.id,
                    'title': rec.title,
                    'type': rec.type.value,
                    'priority': rec.priority.value,
                    'description': rec.description,
                    'rationale': rec.rationale,
                    'expected_impact': rec.expected_impact,
                    'implementation_plan': rec.implementation_plan,
                    'success_metrics': rec.success_metrics,
                    'timeline_weeks': rec.timeline_weeks,
                    'estimated_cost': rec.estimated_cost,
                    'risk_level': rec.risk_level,
                    'dependencies': rec.dependencies,
                    'alternatives': rec.alternatives
                }
                for rec in prioritized
            ],
            'total_recommendations': len(prioritized)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Skill gap recommendations failed: {str(e)}")

@exec_router.get("/roadmap")
def get_implementation_roadmap(scope: str = Query("organization", description="Scope: organization, project, or skill")):
    """Implementation roadmap for strategic recommendations"""
    try:
        if scope == "organization":
            recommendations = recommendation_engine.generate_organization_recommendations()
        else:
            # For now, default to organization-wide
            recommendations = recommendation_engine.generate_organization_recommendations()

        prioritized = recommendation_engine.prioritize_recommendations(recommendations)
        roadmap = recommendation_engine.create_implementation_roadmap(prioritized)

        return roadmap
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Roadmap generation failed: {str(e)}")

@exec_router.get("/scenario-analysis")
def get_scenario_analysis(scenarios: str = Query("base,optimistic,pessimistic", description="Comma-separated scenarios")):
    """What-if scenario analysis for executive decision making"""
    try:
        scenario_list = [s.strip() for s in scenarios.split(",")]

        # Get base financial overview
        base_financial = financial_engine.get_organization_financial_overview()

        results = {}

        for scenario in scenario_list:
            if scenario == "base":
                results[scenario] = {
                    'description': 'Current assumptions and projections',
                    'total_npv': base_financial['organization_overview']['total_expected_npv'],
                    'total_investment': base_financial['organization_overview']['total_investment_required'],
                    'roi': base_financial['organization_overview']['organization_roi'],
                    'risk_level': 'medium'
                }
            elif scenario == "optimistic":
                # Optimistic: 20% better outcomes, 15% lower costs
                base_npv = base_financial['organization_overview']['total_expected_npv']
                base_investment = base_financial['organization_overview']['total_investment_required']

                results[scenario] = {
                    'description': 'Best-case scenario: successful execution, favorable market conditions',
                    'total_npv': base_npv * 1.2,
                    'total_investment': base_investment * 0.85,
                    'roi': (base_npv * 1.2) / (base_investment * 0.85) if base_investment > 0 else 0,
                    'risk_level': 'low',
                    'probability': 0.25
                }
            elif scenario == "pessimistic":
                # Pessimistic: 30% worse outcomes, 25% higher costs
                base_npv = base_financial['organization_overview']['total_expected_npv']
                base_investment = base_financial['organization_overview']['total_investment_required']

                results[scenario] = {
                    'description': 'Worst-case scenario: execution challenges, adverse market conditions',
                    'total_npv': base_npv * 0.7,
                    'total_investment': base_investment * 1.25,
                    'roi': (base_npv * 0.7) / (base_investment * 1.25) if base_investment > 0 else 0,
                    'risk_level': 'high',
                    'probability': 0.15
                }

        # Calculate expected value
        expected_npv = 0
        expected_investment = 0

        scenario_weights = {'optimistic': 0.25, 'base': 0.6, 'pessimistic': 0.15}

        for scenario, weight in scenario_weights.items():
            if scenario in results:
                expected_npv += results[scenario]['total_npv'] * weight
                expected_investment += results[scenario]['total_investment'] * weight

        return {
            'scenarios': results,
            'expected_value_analysis': {
                'expected_npv': expected_npv,
                'expected_investment': expected_investment,
                'expected_roi': expected_npv / expected_investment if expected_investment > 0 else 0
            },
            'sensitivity_factors': {
                'market_conditions': 'High impact on NPV (±20%)',
                'execution_quality': 'High impact on costs (±25%)',
                'talent_availability': 'Medium impact on timeline',
                'regulatory_changes': 'Low probability, high impact'
            },
            'generated_at': base_financial['generated_at']
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Scenario analysis failed: {str(e)}")

@exec_router.get("/metrics/kpis")
def get_executive_kpis():
    """Key Performance Indicators for executive monitoring"""
    try:
        # Get data sources
        financial_overview = financial_engine.get_organization_financial_overview()
        risk_profile = risk_engine.assess_organization_risk()

        # Calculate KPIs
        org_overview = financial_overview['organization_overview']

        kpis = {
            'financial_kpis': {
                'portfolio_npv': {
                    'value': org_overview['total_expected_npv'],
                    'target': org_overview['total_investment_required'] * 1.5,
                    'status': 'green' if org_overview['total_expected_npv'] > org_overview['total_investment_required'] else 'red',
                    'trend': 'increasing',
                    'description': 'Total portfolio NPV'
                },
                'investment_roi': {
                    'value': org_overview['organization_roi'] * 100,
                    'target': 50.0,  # 50% target ROI
                    'status': 'green' if org_overview['organization_roi'] > 0.5 else 'yellow',
                    'trend': 'stable',
                    'description': 'Investment ROI percentage'
                },
                'capital_efficiency': {
                    'value': org_overview['total_expected_npv'] / org_overview['total_investment_required'] if org_overview['total_investment_required'] > 0 else 0,
                    'target': 2.0,  # $2 NPV per $1 invested
                    'status': 'green',
                    'trend': 'increasing',
                    'description': 'NPV per dollar invested'
                }
            },
            'risk_kpis': {
                'organizational_risk_score': {
                    'value': risk_profile.overall_risk_score,
                    'target': 5.0,  # Target risk score ≤ 5
                    'status': 'red' if risk_profile.overall_risk_score > 7 else 'yellow' if risk_profile.overall_risk_score > 5 else 'green',
                    'trend': 'stable',
                    'description': 'Overall organizational risk (0-10 scale)'
                },
                'projects_at_risk': {
                    'value': len([p for p in financial_overview['project_financials']
                               if p['risk_assessment']['portfolio_risk_rating'] in ['high', 'critical']]),
                    'target': 0,
                    'status': 'red',
                    'trend': 'stable',
                    'description': 'Number of high-risk projects'
                },
                'risk_confidence': {
                    'value': risk_profile.confidence_level * 100,
                    'target': 80.0,
                    'status': 'green' if risk_profile.confidence_level > 0.8 else 'yellow',
                    'trend': 'stable',
                    'description': 'Confidence in risk assessments'
                }
            },
            'capability_kpis': {
                'skill_gap_coverage': {
                    'value': 65.0,  # Would calculate from actual proficiency data
                    'target': 85.0,
                    'status': 'yellow',
                    'trend': 'improving',
                    'description': 'Percentage of skill requirements covered'
                },
                'capability_maturity': {
                    'value': 2.8,  # Average skill level across organization
                    'target': 3.5,
                    'status': 'yellow',
                    'trend': 'improving',
                    'description': 'Average organizational capability level'
                },
                'talent_acquisition_rate': {
                    'value': 78.0,
                    'target': 85.0,
                    'status': 'yellow',
                    'trend': 'stable',
                    'description': 'Successful talent acquisition rate'
                }
            },
            'strategic_kpis': {
                'project_success_rate': {
                    'value': 75.0,
                    'target': 90.0,
                    'status': 'yellow',
                    'trend': 'improving',
                    'description': 'Percentage of projects meeting goals'
                },
                'innovation_index': {
                    'value': 7.2,
                    'target': 8.0,
                    'status': 'green',
                    'trend': 'increasing',
                    'description': 'Innovation capability index (0-10)'
                },
                'strategic_alignment': {
                    'value': 82.0,
                    'target': 90.0,
                    'status': 'green',
                    'trend': 'stable',
                    'description': 'Strategic alignment percentage'
                }
            }
        }

        return {
            'kpis': kpis,
            'summary': {
                'total_kpis': sum(len(category) for category in kpis.values()),
                'green_kpis': sum(1 for category in kpis.values() for kpi in category.values() if kpi['status'] == 'green'),
                'yellow_kpis': sum(1 for category in kpis.values() for kpi in category.values() if kpi['status'] == 'yellow'),
                'red_kpis': sum(1 for category in kpis.values() for kpi in category.values() if kpi['status'] == 'red')
            },
            'generated_at': financial_overview['generated_at']
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"KPI generation failed: {str(e)}")

@exec_router.get("/decision-support/investment")
def get_investment_decision_support(budget_limit: Optional[float] = Query(None, description="Budget constraint")):
    """Decision support for investment prioritization"""
    try:
        financial_overview = financial_engine.get_organization_financial_overview()
        budget_allocation = financial_overview['budget_allocation']

        # Apply budget constraint if provided
        if budget_limit:
            # Filter scenarios within budget
            constrained_scenarios = {}
            for scenario_name, scenario_data in budget_allocation['budget_scenarios'].items():
                if scenario_data['budget_allocated'] <= budget_limit:
                    constrained_scenarios[scenario_name] = scenario_data

            if constrained_scenarios:
                # Find best scenario within budget
                best_scenario = max(constrained_scenarios.keys(),
                                  key=lambda x: constrained_scenarios[x]['total_npv'])

                recommendation = {
                    'recommended_scenario': best_scenario,
                    'budget_required': constrained_scenarios[best_scenario]['budget_allocated'],
                    'expected_npv': constrained_scenarios[best_scenario]['total_npv'],
                    'projects_funded': constrained_scenarios[best_scenario]['projects_funded'],
                    'rationale': f"Optimal allocation within ${budget_limit:,.0f} budget constraint"
                }
            else:
                recommendation = {
                    'recommended_scenario': 'insufficient_budget',
                    'budget_required': budget_limit,
                    'expected_npv': 0,
                    'projects_funded': [],
                    'rationale': f"Budget of ${budget_limit:,.0f} insufficient for any meaningful investment"
                }
        else:
            # No budget constraint - use optimal allocation
            recommendation = {
                'recommended_scenario': budget_allocation['recommended_scenario'],
                'budget_required': financial_overview['organization_overview']['total_investment_required'],
                'expected_npv': financial_overview['organization_overview']['total_expected_npv'],
                'projects_funded': [p['project_info']['project_name'] for p in financial_overview['project_financials']],
                'rationale': "Optimal allocation without budget constraints"
            }

        return {
            'investment_recommendation': recommendation,
            'budget_scenarios': budget_allocation['budget_scenarios'],
            'optimization_ranking': budget_allocation['optimization_ranking'][:10],  # Top 10
            'marginal_analysis': budget_allocation['marginal_analysis'],
            'decision_framework': {
                'criteria': ['NPV', 'ROI', 'Risk Level', 'Strategic Alignment'],
                'weights': [0.4, 0.3, 0.2, 0.1],
                'methodology': 'Multi-criteria decision analysis with NPV optimization'
            }
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Investment decision support failed: {str(e)}")

# Export the router
__all__ = ['exec_router']