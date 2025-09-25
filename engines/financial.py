#!/usr/bin/env python3
"""
Financial Analysis Engine for AI Skill Planner
Implements NPV-based recommendations, risk assessment, and Monte Carlo simulation
Based on PRD specifications for Milestone 3
"""

import sys
import os
import math
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime, date, timedelta
import sqlite3

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from database.init_db import get_db_connection
from engines.gap_analysis import GapAnalysisEngine

@dataclass
class FinancialAssumptions:
    """Financial modeling assumptions"""
    discount_rate: float = 0.12  # 12% WACC
    training_success_rate: float = 0.85  # 85% training success
    hiring_lead_time_weeks: float = 16  # 4 months average
    training_lead_time_weeks: float = 12  # 3 months skill development
    contractor_premium: float = 1.8  # 80% premium over FTE

    # Risk factors
    project_delay_cost_multiplier: float = 1.5  # Delay costs 50% more
    market_volatility: float = 0.2  # 20% cost volatility
    skill_decay_annual: float = 0.15  # 15% annual skill decay if unused

@dataclass
class FinancialScenario:
    """Financial scenario for Monte Carlo simulation"""
    hiring_cost: float
    training_cost: float
    contractor_cost: float
    timeline_impact_weeks: float
    success_probability: float
    business_value: float

class FinancialEngine:
    """
    Financial Analysis Engine implementing NPV calculations,
    risk assessment, and strategic recommendations
    """

    def __init__(self, assumptions: Optional[FinancialAssumptions] = None):
        self.assumptions = assumptions or FinancialAssumptions()
        self.gap_engine = GapAnalysisEngine()

        # Cost models from market data
        self.HIRING_COSTS = {
            'junior': 25000,      # Recruiting, onboarding
            'mid_level': 35000,
            'senior': 50000,
            'expert': 75000
        }

        self.TRAINING_COSTS = {
            'internal': 8000,     # Per skill level improvement
            'external': 15000,    # External courses/bootcamps
            'certification': 5000, # Professional certifications
            'mentoring': 12000    # 1:1 expert mentoring
        }

        self.CONTRACTOR_RATES = {
            'junior': 80,         # Hourly rates
            'mid_level': 120,
            'senior': 180,
            'expert': 250
        }

    def calculate_skill_gap_npv(self, project_id: str, phase: str, skill_id: str,
                               time_horizon_weeks: int = 52) -> Dict[str, Any]:
        """
        Calculate NPV analysis for addressing a specific skill gap
        """
        # Get gap analysis
        gap_analysis = self.gap_engine.detect_skill_gap(project_id, phase, skill_id)
        gap_info = gap_analysis['gap_analysis']
        business_impact = gap_analysis['business_impact']

        # Calculate intervention options
        interventions = self._calculate_intervention_options(gap_info, business_impact)

        # Run NPV analysis for each option
        npv_results = {}
        for intervention_name, intervention in interventions.items():
            npv_results[intervention_name] = self._calculate_npv(
                intervention, business_impact, time_horizon_weeks
            )

        # Find optimal recommendation
        best_option = max(npv_results.keys(),
                         key=lambda x: npv_results[x]['expected_npv'])

        # Run Monte Carlo for the best option only
        mc_confidence = self._monte_carlo_npv(
            interventions[best_option], business_impact, time_horizon_weeks, n_simulations=100
        )

        return {
            'skill_gap': {
                'skill_id': skill_id,
                'skill_name': gap_analysis.get('skill_name', 'Unknown'),
                'gap_severity': gap_info['gap_severity'],
                'expected_gap_fte': gap_info['expected_gap_fte'],
                'coverage_ratio': gap_info['coverage_ratio']
            },
            'business_impact': {
                'cost_impact_weekly': business_impact['cost_impact_weekly'],
                'project_delay_risk': business_impact.get('delay_probability', 0.5),
                'opportunity_cost_total': business_impact['cost_impact_weekly'] * time_horizon_weeks
            },
            'intervention_options': npv_results,
            'recommendation': {
                'best_option': best_option,
                'expected_npv': npv_results[best_option]['expected_npv'],
                'confidence_level': mc_confidence['confidence_95_high'] / mc_confidence['mean'] if mc_confidence['mean'] != 0 else 0.8,
                'payback_period_weeks': npv_results[best_option]['payback_period_weeks'],
                'risk_adjusted_roi': npv_results[best_option]['risk_adjusted_roi']
            },
            'sensitivity_analysis': self._run_sensitivity_analysis(
                npv_results[best_option], business_impact
            ),
            'monte_carlo_confidence': mc_confidence,
            'generated_at': datetime.now().isoformat()
        }

    def _calculate_intervention_options(self, gap_info: Dict[str, Any],
                                      business_impact: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
        """Calculate different intervention strategies"""
        gap_fte = gap_info['expected_gap_fte']
        skill_level_needed = gap_info.get('required_level', 3.0)

        interventions = {}

        # Option 1: Hire new FTE
        hire_level = self._determine_hire_level(skill_level_needed)
        interventions['hire_fte'] = {
            'strategy': 'hire_fte',
            'upfront_cost': self.HIRING_COSTS[hire_level] * gap_fte,
            'ongoing_cost_weekly': self._get_weekly_salary(hire_level) * gap_fte,
            'lead_time_weeks': self.assumptions.hiring_lead_time_weeks,
            'success_probability': 0.8,  # Hiring success rate
            'capacity_delivered': gap_fte,
            'permanence': True
        }

        # Option 2: Train existing team
        training_type = self._determine_training_type(skill_level_needed)
        interventions['train_internal'] = {
            'strategy': 'train_internal',
            'upfront_cost': self.TRAINING_COSTS[training_type] * gap_fte,
            'ongoing_cost_weekly': 0,  # Training is one-time
            'lead_time_weeks': self.assumptions.training_lead_time_weeks,
            'success_probability': self.assumptions.training_success_rate,
            'capacity_delivered': gap_fte * 0.8,  # Trained staff slightly less effective
            'permanence': True
        }

        # Option 3: Contract specialists
        contractor_level = self._determine_hire_level(skill_level_needed)
        interventions['contract_specialists'] = {
            'strategy': 'contract_specialists',
            'upfront_cost': 5000,  # Contracting setup costs
            'ongoing_cost_weekly': (self.CONTRACTOR_RATES[contractor_level] * 40) * gap_fte,
            'lead_time_weeks': 4,  # Faster contractor procurement
            'success_probability': 0.95,  # Contractors very reliable
            'capacity_delivered': gap_fte * 1.1,  # Specialists more effective
            'permanence': False  # Temporary solution
        }

        # Option 4: Hybrid approach (train + contract)
        interventions['hybrid_approach'] = {
            'strategy': 'hybrid_approach',
            'upfront_cost': (self.TRAINING_COSTS[training_type] * gap_fte * 0.6 +
                           5000),  # Train 60% of need, contract rest
            'ongoing_cost_weekly': (self.CONTRACTOR_RATES[contractor_level] * 40) * gap_fte * 0.4,
            'lead_time_weeks': 6,  # Parallel execution
            'success_probability': 0.9,  # Lower risk approach
            'capacity_delivered': gap_fte,
            'permanence': False  # Mixed permanence
        }

        return interventions

    def _calculate_npv(self, intervention: Dict[str, Any],
                      business_impact: Dict[str, Any],
                      time_horizon_weeks: int) -> Dict[str, Any]:
        """Calculate risk-adjusted NPV for an intervention"""

        # Cash flow components
        upfront_cost = intervention['upfront_cost']
        weekly_ongoing = intervention['ongoing_cost_weekly']
        lead_time = intervention['lead_time_weeks']
        success_prob = intervention['success_probability']

        # Business value components
        weekly_cost_avoided = business_impact['cost_impact_weekly']
        delay_risk = business_impact.get('delay_probability', 0.5)

        # Calculate cash flows
        cash_flows = []

        # Initial investment (negative)
        cash_flows.append(-upfront_cost)

        # Weekly cash flows
        for week in range(1, time_horizon_weeks + 1):
            if week <= lead_time:
                # During lead time: only costs, no benefits
                weekly_cf = -weekly_ongoing
            else:
                # After lead time: benefits minus ongoing costs
                weekly_benefits = weekly_cost_avoided * success_prob
                weekly_cf = weekly_benefits - weekly_ongoing

            # Add delay risk penalty
            if week <= lead_time:
                delay_penalty = weekly_cost_avoided * delay_risk
                weekly_cf -= delay_penalty

            cash_flows.append(weekly_cf)

        # Calculate NPV
        weekly_discount_rate = (1 + self.assumptions.discount_rate) ** (1/52) - 1

        npv = 0
        for week, cf in enumerate(cash_flows):
            npv += cf / ((1 + weekly_discount_rate) ** week)

        # Risk adjustment
        risk_adjustment = self._calculate_risk_adjustment(intervention, business_impact)
        risk_adjusted_npv = npv * (1 - risk_adjustment)

        # Calculate additional metrics
        payback_period = self._calculate_payback_period(cash_flows)
        roi = (sum(cash_flows) / upfront_cost) if upfront_cost > 0 else 0

        return {
            'expected_npv': risk_adjusted_npv,
            'raw_npv': npv,
            'risk_adjustment_factor': risk_adjustment,
            'total_investment': upfront_cost,
            'annual_benefits': weekly_cost_avoided * success_prob * 52,
            'payback_period_weeks': payback_period,
            'risk_adjusted_roi': roi * (1 - risk_adjustment),
            'cash_flow_profile': cash_flows[:13]  # First quarter for analysis
        }

    def _monte_carlo_npv(self, intervention: Dict[str, Any],
                        business_impact: Dict[str, Any],
                        time_horizon_weeks: int,
                        n_simulations: int = 1000) -> Dict[str, Any]:
        """Run Monte Carlo simulation for NPV confidence intervals"""
        npv_results = []

        for _ in range(n_simulations):
            # Vary key assumptions
            upfront_variance = np.random.normal(1.0, 0.15)  # ±15% cost variance
            success_variance = np.random.normal(1.0, 0.1)   # ±10% success rate variance
            benefit_variance = np.random.normal(1.0, self.assumptions.market_volatility)

            # Adjusted parameters
            sim_upfront = intervention['upfront_cost'] * upfront_variance
            sim_success = min(0.95, intervention['success_probability'] * success_variance)
            sim_benefits = business_impact['cost_impact_weekly'] * benefit_variance

            # Create simulation scenario
            sim_intervention = intervention.copy()
            sim_intervention['upfront_cost'] = sim_upfront
            sim_intervention['success_probability'] = sim_success

            sim_impact = business_impact.copy()
            sim_impact['cost_impact_weekly'] = sim_benefits

            # Calculate NPV for this simulation
            sim_result = self._calculate_npv(sim_intervention, sim_impact, time_horizon_weeks)
            npv_results.append(sim_result['raw_npv'])

        # Calculate confidence intervals
        npv_array = np.array(npv_results)

        return {
            'mean': float(np.mean(npv_array)),
            'std_dev': float(np.std(npv_array)),
            'confidence_95_low': float(np.percentile(npv_array, 2.5)),
            'confidence_95_high': float(np.percentile(npv_array, 97.5)),
            'confidence_80_low': float(np.percentile(npv_array, 10)),
            'confidence_80_high': float(np.percentile(npv_array, 90)),
            'confidence_level': 0.95,
            'probability_positive': float(np.mean(npv_array > 0)),
            'value_at_risk_5pct': float(np.percentile(npv_array, 5))
        }

    def analyze_project_financial_risk(self, project_id: str) -> Dict[str, Any]:
        """Comprehensive financial risk analysis for entire project"""

        # Get project gap analysis
        project_analysis = self.gap_engine.analyze_project_gaps(project_id)
        project_info = project_analysis['project_info']

        # Analyze each critical gap
        gap_financials = []
        total_investment_required = 0
        total_expected_npv = 0

        for gap in project_analysis['top_gaps']:
            if gap['requirement'] and gap['gap_analysis']['gap_severity'] in ['critical', 'high']:
                gap_npv = self.calculate_skill_gap_npv(
                    project_id, gap['phase'], gap['skill_id']
                )
                gap_financials.append(gap_npv)

                best_option = gap_npv['recommendation']['best_option']
                investment = gap_npv['intervention_options'][best_option]['total_investment']
                npv = gap_npv['recommendation']['expected_npv']

                total_investment_required += investment
                total_expected_npv += npv

        # Calculate portfolio-level risk metrics
        portfolio_risk = self._calculate_portfolio_risk(gap_financials, project_info)

        # Generate strategic recommendation
        strategic_rec = self._generate_strategic_recommendation(
            gap_financials, portfolio_risk, project_info
        )

        return {
            'project_info': {
                'project_id': project_id,
                'project_name': project_info['name'],
                'cost_of_delay_weekly': project_info['cost_of_delay_weekly'],
                'project_duration_weeks': self._calculate_project_duration(project_info),
                'total_project_value': project_info['cost_of_delay_weekly'] * 52  # Annualized
            },
            'financial_summary': {
                'total_investment_required': total_investment_required,
                'total_expected_npv': total_expected_npv,
                'portfolio_roi': total_expected_npv / total_investment_required if total_investment_required > 0 else 0,
                'payback_period_weeks': self._calculate_portfolio_payback(gap_financials),
                'break_even_probability': self._calculate_break_even_probability(gap_financials)
            },
            'risk_assessment': portfolio_risk,
            'gap_analyses': gap_financials,
            'strategic_recommendation': strategic_rec,
            'sensitivity_analysis': self._project_sensitivity_analysis(gap_financials, project_info),
            'generated_at': datetime.now().isoformat()
        }

    def get_organization_financial_overview(self) -> Dict[str, Any]:
        """Organization-wide financial analysis and recommendations"""

        # Get all projects
        conn = get_db_connection()
        cursor = conn.cursor()

        cursor.execute("""
            SELECT id, name, cost_of_delay_weekly, start_date, end_date
            FROM projects
            ORDER BY cost_of_delay_weekly DESC
        """)

        projects = [dict(row) for row in cursor.fetchall()]
        conn.close()

        # Analyze each project
        project_financials = []
        total_organization_investment = 0
        total_organization_npv = 0

        for project in projects:
            try:
                project_financial = self.analyze_project_financial_risk(project['id'])
                project_financials.append(project_financial)

                total_organization_investment += project_financial['financial_summary']['total_investment_required']
                total_organization_npv += project_financial['financial_summary']['total_expected_npv']

            except Exception as e:
                print(f"Warning: Could not analyze project {project['id']}: {e}")

        # Calculate organization-level metrics
        organization_metrics = self._calculate_organization_metrics(project_financials)

        # Generate executive recommendations
        executive_recommendations = self._generate_executive_recommendations(
            project_financials, organization_metrics
        )

        return {
            'organization_overview': {
                'total_projects_analyzed': len(project_financials),
                'total_investment_required': total_organization_investment,
                'total_expected_npv': total_organization_npv,
                'organization_roi': total_organization_npv / total_organization_investment if total_organization_investment > 0 else 0,
                'annual_cost_of_inaction': sum(p['project_info']['total_project_value'] for p in project_financials),
                'projects_requiring_investment': len([p for p in project_financials if p['financial_summary']['total_investment_required'] > 0])
            },
            'portfolio_risk': organization_metrics,
            'project_financials': project_financials,
            'executive_recommendations': executive_recommendations,
            'budget_allocation': self._optimize_budget_allocation(project_financials),
            'generated_at': datetime.now().isoformat()
        }

    # Helper methods

    def _determine_hire_level(self, required_level: float) -> str:
        """Determine hiring level based on required skill level"""
        if required_level >= 4.0:
            return 'expert'
        elif required_level >= 3.5:
            return 'senior'
        elif required_level >= 2.5:
            return 'mid_level'
        else:
            return 'junior'

    def _determine_training_type(self, required_level: float) -> str:
        """Determine training approach based on skill gap"""
        if required_level >= 4.0:
            return 'mentoring'  # Expert mentoring required
        elif required_level >= 3.0:
            return 'external'   # External courses
        else:
            return 'internal'   # Internal training

    def _get_weekly_salary(self, level: str) -> float:
        """Get weekly salary cost for different levels"""
        annual_salaries = {
            'junior': 90000,
            'mid_level': 130000,
            'senior': 180000,
            'expert': 250000
        }
        return annual_salaries[level] / 52

    def _calculate_risk_adjustment(self, intervention: Dict[str, Any],
                                 business_impact: Dict[str, Any]) -> float:
        """Calculate risk adjustment factor for NPV"""
        base_risk = 0.1  # 10% base risk

        # Strategy-specific risks
        strategy_risks = {
            'hire_fte': 0.15,      # Hiring risk
            'train_internal': 0.12, # Training risk
            'contract_specialists': 0.08,  # Lower risk
            'hybrid_approach': 0.10  # Balanced risk
        }

        strategy_risk = strategy_risks.get(intervention['strategy'], base_risk)

        # Market volatility risk
        market_risk = self.assumptions.market_volatility * 0.5

        # Timeline risk (longer projects = more risk)
        timeline_risk = min(0.2, intervention['lead_time_weeks'] / 100)

        return min(0.4, base_risk + strategy_risk + market_risk + timeline_risk)

    def _calculate_payback_period(self, cash_flows: List[float]) -> float:
        """Calculate payback period in weeks"""
        cumulative_cf = 0
        for week, cf in enumerate(cash_flows):
            cumulative_cf += cf
            if cumulative_cf > 0:
                return week
        return len(cash_flows)  # Never pays back

    def _run_sensitivity_analysis(self, npv_result: Dict[str, Any],
                                business_impact: Dict[str, Any]) -> Dict[str, Any]:
        """Run sensitivity analysis on key variables"""
        base_npv = npv_result['expected_npv']

        sensitivities = {}

        # Cost sensitivity (+/-20%)
        cost_impact_high = base_npv * 1.2
        cost_impact_low = base_npv * 0.8
        sensitivities['cost_sensitivity'] = {
            'high_20pct': cost_impact_high - base_npv,
            'low_20pct': base_npv - cost_impact_low,
            'elasticity': (cost_impact_high - cost_impact_low) / (0.4 * base_npv) if base_npv != 0 else 0
        }

        # Discount rate sensitivity
        sensitivities['discount_rate_sensitivity'] = {
            'rate_plus_2pct': base_npv * 0.85,  # Approximate impact
            'rate_minus_2pct': base_npv * 1.18,
            'duration_risk': 'high' if npv_result['payback_period_weeks'] > 26 else 'medium'
        }

        # Success probability sensitivity
        success_impact = base_npv * 0.3  # 30% NPV impact from success rates
        sensitivities['success_probability'] = {
            'high_confidence': success_impact,
            'low_confidence': -success_impact,
            'critical_success_factors': ['team_capacity', 'market_conditions', 'execution_quality']
        }

        return sensitivities

    def _calculate_portfolio_risk(self, gap_financials: List[Dict[str, Any]],
                                project_info: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate portfolio-level risk metrics"""

        if not gap_financials:
            return {'total_var': 0, 'correlation_risk': 0, 'concentration_risk': 0}

        # Value at Risk calculation
        npvs = [gf['recommendation']['expected_npv'] for gf in gap_financials]
        total_var = abs(min(npvs)) if npvs else 0

        # Correlation risk (skills often interdependent)
        correlation_risk = len(gap_financials) * 0.1  # Assume 10% correlation penalty per gap

        # Concentration risk
        investment_amounts = [
            gf['intervention_options'][gf['recommendation']['best_option']]['total_investment']
            for gf in gap_financials
        ]
        total_investment = sum(investment_amounts)

        if total_investment > 0:
            concentration_scores = [(inv / total_investment) ** 2 for inv in investment_amounts]
            concentration_risk = sum(concentration_scores)  # Herfindahl index
        else:
            concentration_risk = 0

        return {
            'total_value_at_risk': total_var,
            'correlation_risk_factor': min(1.0, correlation_risk),
            'concentration_risk_score': concentration_risk,
            'portfolio_risk_rating': self._rate_portfolio_risk(total_var, correlation_risk, concentration_risk),
            'risk_mitigation_suggestions': self._suggest_risk_mitigations(gap_financials)
        }

    def _generate_strategic_recommendation(self, gap_financials: List[Dict[str, Any]],
                                         portfolio_risk: Dict[str, Any],
                                         project_info: Dict[str, Any]) -> Dict[str, Any]:
        """Generate strategic recommendation for project"""

        if not gap_financials:
            return {'decision': 'proceed', 'rationale': 'No critical gaps identified'}

        total_npv = sum(gf['recommendation']['expected_npv'] for gf in gap_financials)
        total_investment = sum(
            gf['intervention_options'][gf['recommendation']['best_option']]['total_investment']
            for gf in gap_financials
        )

        # Decision logic
        if total_npv > total_investment * 0.5:  # 50% ROI threshold
            if portfolio_risk['portfolio_risk_rating'] in ['low', 'medium']:
                decision = 'proceed'
                rationale = f"Strong NPV (${total_npv:,.0f}) with manageable risk"
            else:
                decision = 'proceed_with_caution'
                rationale = f"Positive NPV but high portfolio risk. Consider risk mitigation."
        elif total_npv > 0:
            decision = 'conditional'
            rationale = f"Marginal NPV (${total_npv:,.0f}). Revisit assumptions and scope."
        else:
            decision = 'defer'
            rationale = f"Negative NPV (${total_npv:,.0f}). Consider alternative approaches."

        return {
            'decision': decision,
            'rationale': rationale,
            'confidence_level': self._calculate_decision_confidence(gap_financials),
            'key_success_factors': self._identify_success_factors(gap_financials),
            'alternative_strategies': self._suggest_alternatives(gap_financials, decision)
        }

    def _rate_portfolio_risk(self, var: float, correlation: float, concentration: float) -> str:
        """Rate overall portfolio risk"""
        risk_score = (var / 100000) + correlation + concentration  # Normalize VAR

        if risk_score < 0.5:
            return 'low'
        elif risk_score < 1.0:
            return 'medium'
        elif risk_score < 2.0:
            return 'high'
        else:
            return 'critical'

    def _suggest_risk_mitigations(self, gap_financials: List[Dict[str, Any]]) -> List[str]:
        """Suggest risk mitigation strategies"""
        suggestions = []

        # Check for high-risk interventions
        high_risk_gaps = [gf for gf in gap_financials
                         if gf['recommendation']['expected_npv'] < 0]

        if high_risk_gaps:
            suggestions.append("Consider hybrid approaches for high-risk skill gaps")

        if len(gap_financials) > 5:
            suggestions.append("Phase interventions to reduce execution risk")

        # Check for contractor dependency
        contractor_heavy = sum(1 for gf in gap_financials
                             if gf['recommendation']['best_option'] == 'contract_specialists')

        if contractor_heavy > len(gap_financials) * 0.5:
            suggestions.append("High contractor dependency - consider building internal capabilities")

        return suggestions

    def _calculate_project_duration(self, project_info: Dict[str, Any]) -> int:
        """Calculate project duration in weeks"""
        try:
            start = datetime.fromisoformat(project_info['start_date']).date()
            end = datetime.fromisoformat(project_info['end_date']).date()
            return (end - start).days // 7
        except:
            return 52  # Default 1 year

    def _calculate_portfolio_payback(self, gap_financials: List[Dict[str, Any]]) -> float:
        """Calculate weighted average payback period"""
        if not gap_financials:
            return 0

        total_investment = sum(
            gf['intervention_options'][gf['recommendation']['best_option']]['total_investment']
            for gf in gap_financials
        )

        if total_investment == 0:
            return 0

        weighted_payback = sum(
            gf['recommendation']['payback_period_weeks'] *
            gf['intervention_options'][gf['recommendation']['best_option']]['total_investment']
            for gf in gap_financials
        ) / total_investment

        return weighted_payback

    def _calculate_break_even_probability(self, gap_financials: List[Dict[str, Any]]) -> float:
        """Calculate probability of portfolio breaking even"""
        if not gap_financials:
            return 1.0

        # Assume independence and multiply probabilities
        break_even_prob = 1.0
        for gf in gap_financials:
            try:
                best_option = gf['recommendation']['best_option']
                intervention = gf['intervention_options'][best_option]

                # Safely access confidence_interval with fallback
                if 'confidence_interval' in intervention and 'probability_positive' in intervention['confidence_interval']:
                    prob_positive = intervention['confidence_interval']['probability_positive']
                elif 'monte_carlo' in intervention and 'probability_positive' in intervention['monte_carlo']:
                    prob_positive = intervention['monte_carlo']['probability_positive']
                else:
                    prob_positive = 0.75  # Default 75% success probability

                break_even_prob *= prob_positive
            except (KeyError, TypeError) as e:
                print(f"Warning: Could not analyze project {gf.get('project_info', {}).get('project_name', 'unknown')}: {str(e)}")
                # Use default probability
                break_even_prob *= 0.75

        return break_even_prob

    def _project_sensitivity_analysis(self, gap_financials: List[Dict[str, Any]],
                                    project_info: Dict[str, Any]) -> Dict[str, Any]:
        """Project-level sensitivity analysis"""
        return {
            'cost_of_delay_impact': {
                'baseline_weekly': project_info.get('cost_of_delay_weekly', 0),
                'sensitivity_high': 'NPV decreases significantly with project delays',
                'sensitivity_low': 'Strong NPV resilience to minor delays'
            },
            'market_conditions': {
                'bull_case': 'Higher talent costs but also higher project values',
                'bear_case': 'Lower costs but potentially reduced project ROI',
                'most_likely': 'Current assumptions hold steady'
            }
        }

    def _calculate_organization_metrics(self, project_financials: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate organization-wide risk and performance metrics"""

        if not project_financials:
            return {}

        # Aggregate risk metrics
        total_var = sum(pf['risk_assessment']['total_value_at_risk'] for pf in project_financials)
        avg_correlation = np.mean([pf['risk_assessment']['correlation_risk_factor'] for pf in project_financials])

        # Portfolio diversification
        strategy_distribution = {}
        for pf in project_financials:
            for gap in pf['gap_analyses']:
                strategy = gap['recommendation']['best_option']
                strategy_distribution[strategy] = strategy_distribution.get(strategy, 0) + 1

        return {
            'aggregate_value_at_risk': total_var,
            'portfolio_correlation': avg_correlation,
            'strategy_diversification': strategy_distribution,
            'high_risk_projects': len([pf for pf in project_financials
                                     if pf['risk_assessment']['portfolio_risk_rating'] in ['high', 'critical']]),
            'average_payback_weeks': np.mean([pf['financial_summary']['payback_period_weeks']
                                            for pf in project_financials])
        }

    def _generate_executive_recommendations(self, project_financials: List[Dict[str, Any]],
                                          organization_metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Generate executive-level strategic recommendations"""

        total_investment = sum(pf['financial_summary']['total_investment_required'] for pf in project_financials)
        total_npv = sum(pf['financial_summary']['total_expected_npv'] for pf in project_financials)

        # Strategic priorities
        high_npv_projects = sorted(project_financials,
                                 key=lambda x: x['financial_summary']['total_expected_npv'],
                                 reverse=True)[:3]

        # Budget allocation strategy
        if total_npv > total_investment * 0.5:
            strategy = "aggressive_investment"
            rationale = "Strong ROI across portfolio justifies significant investment"
        elif total_npv > 0:
            strategy = "selective_investment"
            rationale = "Focus on highest-NPV projects, defer marginal ones"
        else:
            strategy = "defensive"
            rationale = "Negative portfolio NPV suggests fundamental reassessment needed"

        return {
            'investment_strategy': strategy,
            'strategic_rationale': rationale,
            'priority_projects': [p['project_info']['project_name'] for p in high_npv_projects],
            'budget_recommendations': {
                'total_budget_needed': total_investment,
                'expected_return': total_npv,
                'payback_period': organization_metrics.get('average_payback_weeks', 0),
                'risk_level': self._assess_organization_risk_level(organization_metrics)
            },
            'capability_building_priorities': self._identify_capability_priorities(project_financials)
        }

    def _optimize_budget_allocation(self, project_financials: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Optimize budget allocation across projects using NPV ranking"""

        # Create investment opportunities list
        opportunities = []
        for pf in project_financials:
            opportunities.append({
                'project_name': pf['project_info']['project_name'],
                'investment_required': pf['financial_summary']['total_investment_required'],
                'expected_npv': pf['financial_summary']['total_expected_npv'],
                'roi': pf['financial_summary']['portfolio_roi'],
                'payback_weeks': pf['financial_summary']['payback_period_weeks'],
                'risk_rating': pf['risk_assessment']['portfolio_risk_rating']
            })

        # Sort by NPV/Investment ratio (efficiency)
        opportunities.sort(key=lambda x: x['expected_npv'] / max(x['investment_required'], 1), reverse=True)

        # Create budget scenarios
        budget_scenarios = {}
        for budget_pct in [0.5, 0.75, 1.0, 1.25]:
            total_available = sum(op['investment_required'] for op in opportunities) * budget_pct

            selected_projects = []
            remaining_budget = total_available
            total_npv = 0

            for opp in opportunities:
                if opp['investment_required'] <= remaining_budget:
                    selected_projects.append(opp['project_name'])
                    remaining_budget -= opp['investment_required']
                    total_npv += opp['expected_npv']

            budget_scenarios[f"{int(budget_pct*100)}pct_budget"] = {
                'budget_allocated': total_available - remaining_budget,
                'projects_funded': selected_projects,
                'total_npv': total_npv,
                'projects_count': len(selected_projects),
                'utilization_rate': (total_available - remaining_budget) / total_available if total_available > 0 else 0
            }

        return {
            'optimization_ranking': opportunities,
            'budget_scenarios': budget_scenarios,
            'recommended_scenario': '100pct_budget',  # Default recommendation
            'marginal_analysis': self._marginal_investment_analysis(opportunities)
        }

    def _assess_organization_risk_level(self, organization_metrics: Dict[str, Any]) -> str:
        """Assess overall organization risk level"""
        risk_factors = 0

        if organization_metrics.get('aggregate_value_at_risk', 0) > 500000:
            risk_factors += 1

        if organization_metrics.get('high_risk_projects', 0) > len(organization_metrics) * 0.3:
            risk_factors += 1

        if organization_metrics.get('average_payback_weeks', 0) > 26:
            risk_factors += 1

        if risk_factors >= 2:
            return 'high'
        elif risk_factors == 1:
            return 'medium'
        else:
            return 'low'

    def _identify_capability_priorities(self, project_financials: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Identify top capability building priorities across organization"""

        skill_investments = {}

        for pf in project_financials:
            for gap in pf['gap_analyses']:
                skill_name = gap['skill_gap']['skill_name']
                investment = gap['intervention_options'][gap['recommendation']['best_option']]['total_investment']
                npv = gap['recommendation']['expected_npv']

                if skill_name not in skill_investments:
                    skill_investments[skill_name] = {
                        'total_investment': 0,
                        'total_npv': 0,
                        'project_count': 0,
                        'avg_severity': []
                    }

                skill_investments[skill_name]['total_investment'] += investment
                skill_investments[skill_name]['total_npv'] += npv
                skill_investments[skill_name]['project_count'] += 1
                skill_investments[skill_name]['avg_severity'].append(gap['skill_gap']['gap_severity'])

        # Calculate priorities
        priorities = []
        for skill_name, data in skill_investments.items():
            avg_severity_score = len([s for s in data['avg_severity'] if s == 'critical']) / len(data['avg_severity'])

            priorities.append({
                'skill_name': skill_name,
                'priority_score': (data['total_npv'] / max(data['total_investment'], 1)) * data['project_count'] * (1 + avg_severity_score),
                'total_investment': data['total_investment'],
                'total_npv': data['total_npv'],
                'projects_affected': data['project_count'],
                'severity_score': avg_severity_score
            })

        return sorted(priorities, key=lambda x: x['priority_score'], reverse=True)[:10]

    def _marginal_investment_analysis(self, opportunities: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze marginal return on investment"""

        if len(opportunities) < 2:
            return {'marginal_npv': 0, 'marginal_projects': []}

        # Calculate marginal NPV for each additional project
        marginal_analysis = []
        cumulative_investment = 0
        cumulative_npv = 0

        for i, opp in enumerate(opportunities):
            cumulative_investment += opp['investment_required']
            cumulative_npv += opp['expected_npv']

            marginal_roi = opp['expected_npv'] / max(opp['investment_required'], 1)

            marginal_analysis.append({
                'project_rank': i + 1,
                'project_name': opp['project_name'],
                'marginal_investment': opp['investment_required'],
                'marginal_npv': opp['expected_npv'],
                'marginal_roi': marginal_roi,
                'cumulative_investment': cumulative_investment,
                'cumulative_npv': cumulative_npv
            })

        # Find optimal cutoff point (where marginal ROI drops significantly)
        optimal_cutoff = len(opportunities)  # Default: fund all
        for i in range(1, len(marginal_analysis)):
            if marginal_analysis[i]['marginal_roi'] < marginal_analysis[i-1]['marginal_roi'] * 0.5:
                optimal_cutoff = i
                break

        return {
            'marginal_projects': marginal_analysis,
            'optimal_portfolio_size': optimal_cutoff,
            'diminishing_returns_threshold': optimal_cutoff,
            'optimal_investment': marginal_analysis[optimal_cutoff-1]['cumulative_investment'] if optimal_cutoff > 0 else 0,
            'optimal_npv': marginal_analysis[optimal_cutoff-1]['cumulative_npv'] if optimal_cutoff > 0 else 0
        }

    def _calculate_decision_confidence(self, gap_financials: List[Dict[str, Any]]) -> float:
        """Calculate confidence level for strategic decision"""
        if not gap_financials:
            return 1.0

        # Average confidence across all gaps
        confidences = []
        for gf in gap_financials:
            best_option = gf['recommendation']['best_option']
            # Use Monte Carlo confidence if available, otherwise default confidence
            if 'monte_carlo_confidence' in gf and 'probability_positive' in gf['monte_carlo_confidence']:
                prob_positive = gf['monte_carlo_confidence']['probability_positive']
            else:
                prob_positive = 0.8  # Default confidence
            confidences.append(prob_positive)

        return np.mean(confidences)

    def _identify_success_factors(self, gap_financials: List[Dict[str, Any]]) -> List[str]:
        """Identify key success factors for project"""
        factors = set()

        # Common success factors
        for gf in gap_financials:
            strategy = gf['recommendation']['best_option']

            if strategy == 'hire_fte':
                factors.add('successful_recruitment')
                factors.add('effective_onboarding')
            elif strategy == 'train_internal':
                factors.add('training_program_quality')
                factors.add('employee_engagement')
            elif strategy == 'contract_specialists':
                factors.add('vendor_management')
                factors.add('knowledge_transfer')
            elif strategy == 'hybrid_approach':
                factors.add('coordination_effectiveness')
                factors.add('change_management')

        # Always important
        factors.add('stakeholder_alignment')
        factors.add('budget_availability')

        return list(factors)

    def _suggest_alternatives(self, gap_financials: List[Dict[str, Any]],
                            decision: str) -> List[str]:
        """Suggest alternative strategies based on decision"""
        alternatives = []

        if decision == 'defer':
            alternatives.append("Scope reduction - focus on highest-value features only")
            alternatives.append("Timeline extension - reduce pressure and costs")
            alternatives.append("Technology substitution - use lower-skill alternatives")

        elif decision == 'conditional':
            alternatives.append("Phased approach - implement in stages")
            alternatives.append("Partnership strategy - joint venture with skilled organization")
            alternatives.append("Market validation - confirm assumptions before full investment")

        elif decision in ['proceed', 'proceed_with_caution']:
            alternatives.append("Insurance/hedging - protect against key person risk")
            alternatives.append("Accelerated timeline - front-load critical capabilities")
            alternatives.append("Talent pipeline - build sustainable capabilities")

        return alternatives