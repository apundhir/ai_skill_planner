#!/usr/bin/env python3
"""
Risk Assessment Framework for AI Skill Planner
Implements comprehensive risk modeling for skill gaps, projects, and organizational capabilities
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
from engines.proficiency import ProficiencyCalculator

class RiskCategory(Enum):
    EXECUTION = "execution"
    MARKET = "market"
    TECHNICAL = "technical"
    ORGANIZATIONAL = "organizational"
    EXTERNAL = "external"

class RiskSeverity(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

@dataclass
class RiskFactor:
    """Individual risk factor with quantified impact"""
    name: str
    category: RiskCategory
    severity: RiskSeverity
    probability: float  # 0-1
    impact_score: float  # 0-10
    description: str
    mitigation_strategies: List[str]
    cost_impact_weekly: float = 0.0

@dataclass
class RiskProfile:
    """Comprehensive risk profile for entity"""
    entity_type: str  # 'skill', 'project', 'organization'
    entity_id: str
    overall_risk_score: float  # 0-10
    risk_factors: List[RiskFactor]
    risk_distribution: Dict[str, float]  # By category
    confidence_level: float
    recommendations: List[str]

class RiskAssessmentEngine:
    """
    Advanced risk assessment engine implementing multiple risk models
    and quantification frameworks from financial literature
    """

    def __init__(self):
        self.gap_engine = GapAnalysisEngine()
        self.proficiency_calc = ProficiencyCalculator()

        # Risk model parameters
        self.RISK_WEIGHTS = {
            RiskCategory.EXECUTION: 0.25,
            RiskCategory.MARKET: 0.20,
            RiskCategory.TECHNICAL: 0.25,
            RiskCategory.ORGANIZATIONAL: 0.20,
            RiskCategory.EXTERNAL: 0.10
        }

        # Severity multipliers for impact calculation
        self.SEVERITY_MULTIPLIERS = {
            RiskSeverity.LOW: 1.0,
            RiskSeverity.MEDIUM: 2.5,
            RiskSeverity.HIGH: 5.0,
            RiskSeverity.CRITICAL: 8.0
        }

        # Market risk factors (updated periodically)
        self.MARKET_CONDITIONS = {
            'talent_availability': 0.7,  # 30% talent shortage
            'wage_inflation': 0.15,      # 15% annual wage growth
            'technology_velocity': 0.8,   # High pace of change
            'regulatory_uncertainty': 0.3  # Moderate regulatory risk
        }

    def assess_skill_gap_risk(self, project_id: str, phase: str, skill_id: str) -> RiskProfile:
        """Comprehensive risk assessment for individual skill gap"""

        # Get gap analysis
        gap_analysis = self.gap_engine.detect_skill_gap(project_id, phase, skill_id)
        gap_info = gap_analysis['gap_analysis']
        business_impact = gap_analysis['business_impact']

        risk_factors = []

        # Execution Risk Factors
        execution_risks = self._assess_execution_risks(gap_info, business_impact)
        risk_factors.extend(execution_risks)

        # Market Risk Factors
        market_risks = self._assess_market_risks(skill_id, gap_info)
        risk_factors.extend(market_risks)

        # Technical Risk Factors
        technical_risks = self._assess_technical_risks(project_id, skill_id, gap_info)
        risk_factors.extend(technical_risks)

        # Organizational Risk Factors
        org_risks = self._assess_organizational_risks(gap_info, business_impact)
        risk_factors.extend(org_risks)

        # External Risk Factors
        external_risks = self._assess_external_risks(skill_id)
        risk_factors.extend(external_risks)

        # Calculate overall risk score
        overall_score = self._calculate_risk_score(risk_factors)

        # Generate risk distribution
        risk_distribution = self._calculate_risk_distribution(risk_factors)

        # Generate recommendations
        recommendations = self._generate_risk_recommendations(risk_factors, gap_info)

        return RiskProfile(
            entity_type='skill_gap',
            entity_id=f"{project_id}_{phase}_{skill_id}",
            overall_risk_score=overall_score,
            risk_factors=risk_factors,
            risk_distribution=risk_distribution,
            confidence_level=self._calculate_confidence_level(risk_factors),
            recommendations=recommendations
        )

    def assess_project_risk(self, project_id: str) -> RiskProfile:
        """Comprehensive risk assessment for entire project"""

        # Get project analysis
        project_analysis = self.gap_engine.analyze_project_gaps(project_id)
        project_info = project_analysis['project_info']

        risk_factors = []

        # Aggregate risks from individual skill gaps
        skill_gap_risks = []
        for gap in project_analysis['top_gaps']:
            if gap['requirement']:
                skill_risk = self.assess_skill_gap_risk(project_id, gap['phase'], gap['skill_id'])
                skill_gap_risks.append(skill_risk)

                # Add high-impact skill risks to project level
                for rf in skill_risk.risk_factors:
                    if rf.severity in [RiskSeverity.HIGH, RiskSeverity.CRITICAL]:
                        risk_factors.append(rf)

        # Project-specific risks
        project_risks = self._assess_project_specific_risks(project_info, project_analysis)
        risk_factors.extend(project_risks)

        # Portfolio risks (if project is part of larger initiative)
        portfolio_risks = self._assess_portfolio_risks(project_id, project_info)
        risk_factors.extend(portfolio_risks)

        # Timeline and dependency risks
        timeline_risks = self._assess_timeline_risks(project_info, project_analysis)
        risk_factors.extend(timeline_risks)

        # Calculate overall project risk
        overall_score = self._calculate_project_risk_score(risk_factors, skill_gap_risks)

        return RiskProfile(
            entity_type='project',
            entity_id=project_id,
            overall_risk_score=overall_score,
            risk_factors=risk_factors,
            risk_distribution=self._calculate_risk_distribution(risk_factors),
            confidence_level=self._calculate_confidence_level(risk_factors),
            recommendations=self._generate_project_risk_recommendations(risk_factors, project_info)
        )

    def assess_organization_risk(self) -> RiskProfile:
        """Organization-wide risk assessment"""

        # Get all projects
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT id FROM projects")
        project_ids = [row['id'] for row in cursor.fetchall()]
        conn.close()

        # Assess each project
        project_risks = []
        for project_id in project_ids:
            try:
                project_risk = self.assess_project_risk(project_id)
                project_risks.append(project_risk)
            except Exception as e:
                print(f"Warning: Could not assess risk for project {project_id}: {e}")

        # Organization-level risks
        org_risks = self._assess_organization_level_risks(project_risks)

        # Systemic risks
        systemic_risks = self._assess_systemic_risks(project_risks)

        # Capability risks
        capability_risks = self._assess_capability_risks()

        all_risks = org_risks + systemic_risks + capability_risks

        # Calculate organization risk score
        overall_score = self._calculate_organization_risk_score(project_risks, all_risks)

        return RiskProfile(
            entity_type='organization',
            entity_id='organization',
            overall_risk_score=overall_score,
            risk_factors=all_risks,
            risk_distribution=self._calculate_risk_distribution(all_risks),
            confidence_level=self._calculate_confidence_level(all_risks),
            recommendations=self._generate_organization_risk_recommendations(all_risks, project_risks)
        )

    def calculate_risk_adjusted_metrics(self, base_metrics: Dict[str, Any],
                                      risk_profile: RiskProfile) -> Dict[str, Any]:
        """Apply risk adjustments to financial and operational metrics"""

        risk_adjustment_factor = self._calculate_risk_adjustment_factor(risk_profile)

        adjusted_metrics = {}

        # Adjust NPV and financial metrics
        if 'npv' in base_metrics:
            adjusted_metrics['risk_adjusted_npv'] = base_metrics['npv'] * (1 - risk_adjustment_factor)

        if 'expected_value' in base_metrics:
            adjusted_metrics['risk_adjusted_expected_value'] = base_metrics['expected_value'] * (1 - risk_adjustment_factor)

        # Adjust timeline estimates
        if 'timeline_weeks' in base_metrics:
            timeline_risk = self._extract_timeline_risk(risk_profile)
            adjusted_metrics['risk_adjusted_timeline'] = base_metrics['timeline_weeks'] * (1 + timeline_risk)

        # Adjust cost estimates
        if 'cost_estimate' in base_metrics:
            cost_risk = self._extract_cost_risk(risk_profile)
            adjusted_metrics['risk_adjusted_cost'] = base_metrics['cost_estimate'] * (1 + cost_risk)

        # Calculate confidence intervals
        adjusted_metrics['confidence_intervals'] = self._calculate_confidence_intervals(
            base_metrics, risk_profile
        )

        # Add risk metrics
        adjusted_metrics['risk_metrics'] = {
            'overall_risk_score': risk_profile.overall_risk_score,
            'risk_adjustment_factor': risk_adjustment_factor,
            'primary_risk_categories': self._identify_primary_risks(risk_profile),
            'risk_mitigation_priority': self._prioritize_risk_mitigations(risk_profile)
        }

        return adjusted_metrics

    # Internal risk assessment methods

    def _assess_execution_risks(self, gap_info: Dict[str, Any],
                              business_impact: Dict[str, Any]) -> List[RiskFactor]:
        """Assess execution-related risks for skill gap"""

        risks = []

        # Gap severity risk
        gap_severity = gap_info.get('gap_severity', 'low')
        if gap_severity == 'critical':
            risks.append(RiskFactor(
                name="Critical Skill Gap",
                category=RiskCategory.EXECUTION,
                severity=RiskSeverity.CRITICAL,
                probability=0.95,
                impact_score=9.0,
                description=f"Critical gap with {gap_info.get('coverage_ratio', 0)*100:.0f}% coverage",
                mitigation_strategies=["Emergency hiring", "Contract specialists", "Scope reduction"],
                cost_impact_weekly=business_impact.get('cost_impact_weekly', 0)
            ))

        # Bus factor risk
        bus_factor = gap_info.get('bus_factor', 5)
        if bus_factor <= 2:
            risks.append(RiskFactor(
                name="Low Bus Factor",
                category=RiskCategory.EXECUTION,
                severity=RiskSeverity.HIGH,
                probability=0.6,
                impact_score=7.0,
                description=f"Only {bus_factor} people have this skill",
                mitigation_strategies=["Cross-training", "Documentation", "Knowledge sharing"],
                cost_impact_weekly=business_impact.get('cost_impact_weekly', 0) * 0.3
            ))

        # Capacity utilization risk
        coverage_ratio = gap_info.get('coverage_ratio', 1.0)
        if coverage_ratio < 0.5:
            risks.append(RiskFactor(
                name="Low Capacity Coverage",
                category=RiskCategory.EXECUTION,
                severity=RiskSeverity.HIGH,
                probability=0.8,
                impact_score=6.0,
                description=f"Only {coverage_ratio*100:.0f}% capacity coverage",
                mitigation_strategies=["Resource reallocation", "Timeline adjustment", "Priority revision"],
                cost_impact_weekly=business_impact.get('cost_impact_weekly', 0) * 0.4
            ))

        return risks

    def _assess_market_risks(self, skill_id: str, gap_info: Dict[str, Any]) -> List[RiskFactor]:
        """Assess market-related risks"""

        risks = []

        # Talent scarcity risk
        talent_availability = self.MARKET_CONDITIONS['talent_availability']
        if talent_availability < 0.5:
            risks.append(RiskFactor(
                name="Talent Scarcity",
                category=RiskCategory.MARKET,
                severity=RiskSeverity.HIGH,
                probability=0.7,
                impact_score=7.0,
                description=f"Limited talent pool for skill {skill_id}",
                mitigation_strategies=["Remote hiring", "Training programs", "Contractor networks"],
                cost_impact_weekly=gap_info.get('expected_gap_fte', 1) * 2000  # $2K/week per FTE
            ))

        # Wage inflation risk
        wage_inflation = self.MARKET_CONDITIONS['wage_inflation']
        if wage_inflation > 0.10:
            risks.append(RiskFactor(
                name="Wage Inflation",
                category=RiskCategory.MARKET,
                severity=RiskSeverity.MEDIUM,
                probability=0.8,
                impact_score=4.0,
                description=f"{wage_inflation*100:.0f}% annual wage growth",
                mitigation_strategies=["Long-term contracts", "Equity compensation", "Skill development"],
                cost_impact_weekly=gap_info.get('expected_gap_fte', 1) * 1000 * wage_inflation
            ))

        return risks

    def _assess_technical_risks(self, project_id: str, skill_id: str,
                              gap_info: Dict[str, Any]) -> List[RiskFactor]:
        """Assess technical and complexity risks"""

        risks = []

        # Get project complexity
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT complexity FROM projects WHERE id = ?", (project_id,))
        complexity = cursor.fetchone()
        conn.close()

        if complexity and complexity['complexity'] == 'high':
            risks.append(RiskFactor(
                name="High Technical Complexity",
                category=RiskCategory.TECHNICAL,
                severity=RiskSeverity.HIGH,
                probability=0.6,
                impact_score=6.0,
                description="High complexity project requires expert-level skills",
                mitigation_strategies=["Expert consultation", "Proof of concept", "Phased approach"],
                cost_impact_weekly=gap_info.get('expected_gap_fte', 1) * 1500
            ))

        # Technology velocity risk
        tech_velocity = self.MARKET_CONDITIONS['technology_velocity']
        if tech_velocity > 0.7:
            risks.append(RiskFactor(
                name="Rapid Technology Evolution",
                category=RiskCategory.TECHNICAL,
                severity=RiskSeverity.MEDIUM,
                probability=0.5,
                impact_score=5.0,
                description="Fast-changing technology landscape",
                mitigation_strategies=["Continuous learning", "Technology monitoring", "Flexible architecture"],
                cost_impact_weekly=500  # Constant monitoring cost
            ))

        return risks

    def _assess_organizational_risks(self, gap_info: Dict[str, Any],
                                   business_impact: Dict[str, Any]) -> List[RiskFactor]:
        """Assess organizational and cultural risks"""

        risks = []

        # Change management risk
        if gap_info.get('gap_severity') in ['critical', 'high']:
            risks.append(RiskFactor(
                name="Change Management Challenge",
                category=RiskCategory.ORGANIZATIONAL,
                severity=RiskSeverity.MEDIUM,
                probability=0.4,
                impact_score=5.0,
                description="Significant skill gaps require organizational changes",
                mitigation_strategies=["Change management program", "Leadership alignment", "Communication plan"],
                cost_impact_weekly=business_impact.get('cost_impact_weekly', 0) * 0.1
            ))

        # Knowledge retention risk
        risks.append(RiskFactor(
            name="Knowledge Retention",
            category=RiskCategory.ORGANIZATIONAL,
            severity=RiskSeverity.MEDIUM,
            probability=0.3,
            impact_score=4.0,
            description="Risk of losing institutional knowledge",
            mitigation_strategies=["Documentation", "Mentoring programs", "Knowledge sharing"],
            cost_impact_weekly=1000  # Documentation and retention efforts
        ))

        return risks

    def _assess_external_risks(self, skill_id: str) -> List[RiskFactor]:
        """Assess external environmental risks"""

        risks = []

        # Regulatory risk
        regulatory_uncertainty = self.MARKET_CONDITIONS['regulatory_uncertainty']
        if regulatory_uncertainty > 0.2:
            risks.append(RiskFactor(
                name="Regulatory Uncertainty",
                category=RiskCategory.EXTERNAL,
                severity=RiskSeverity.MEDIUM,
                probability=regulatory_uncertainty,
                impact_score=3.0,
                description="Potential regulatory changes affecting AI/tech skills",
                mitigation_strategies=["Regulatory monitoring", "Compliance preparation", "Legal consultation"],
                cost_impact_weekly=500
            ))

        # Economic downturn risk
        risks.append(RiskFactor(
            name="Economic Sensitivity",
            category=RiskCategory.EXTERNAL,
            severity=RiskSeverity.LOW,
            probability=0.2,
            impact_score=6.0,
            description="Economic conditions could impact project funding",
            mitigation_strategies=["Scenario planning", "Cost flexibility", "Priority adjustment"],
            cost_impact_weekly=0  # Contingent risk
        ))

        return risks

    def _assess_project_specific_risks(self, project_info: Dict[str, Any],
                                     project_analysis: Dict[str, Any]) -> List[RiskFactor]:
        """Project-specific risk factors"""

        risks = []

        # Project timeline risk
        try:
            start_date = datetime.fromisoformat(project_info['start_date']).date()
            end_date = datetime.fromisoformat(project_info['end_date']).date()
            duration_weeks = (end_date - start_date).days / 7
        except (KeyError, ValueError):
            # Use default duration if dates are missing
            duration_weeks = 26  # 6 months default

        if duration_weeks > 52:  # Projects longer than 1 year
            risks.append(RiskFactor(
                name="Extended Timeline Risk",
                category=RiskCategory.EXECUTION,
                severity=RiskSeverity.MEDIUM,
                probability=0.4,
                impact_score=5.0,
                description=f"Long project duration ({duration_weeks:.0f} weeks) increases risk",
                mitigation_strategies=["Milestone gating", "Agile approach", "Regular reviews"],
                cost_impact_weekly=project_info.get('cost_of_delay_weekly', 0) * 0.1
            ))

        # Regulatory intensity risk
        if project_info.get('regulatory_intensity') == 'high':
            risks.append(RiskFactor(
                name="High Regulatory Requirements",
                category=RiskCategory.EXTERNAL,
                severity=RiskSeverity.HIGH,
                probability=0.6,
                impact_score=6.0,
                description="High regulatory requirements increase complexity",
                mitigation_strategies=["Regulatory expertise", "Compliance framework", "Early validation"],
                cost_impact_weekly=project_info.get('cost_of_delay_weekly', 0) * 0.15
            ))

        return risks

    def _assess_portfolio_risks(self, project_id: str, project_info: Dict[str, Any]) -> List[RiskFactor]:
        """Portfolio-level risks affecting project"""

        risks = []

        # Resource contention risk (simplified - would need more project data)
        risks.append(RiskFactor(
            name="Resource Contention",
            category=RiskCategory.ORGANIZATIONAL,
            severity=RiskSeverity.MEDIUM,
            probability=0.3,
            impact_score=4.0,
            description="Competition for shared resources across projects",
            mitigation_strategies=["Resource planning", "Priority matrix", "Flexible resourcing"],
            cost_impact_weekly=project_info.get('cost_of_delay_weekly', 0) * 0.05
        ))

        return risks

    def _assess_timeline_risks(self, project_info: Dict[str, Any],
                             project_analysis: Dict[str, Any]) -> List[RiskFactor]:
        """Timeline and scheduling risks"""

        risks = []

        # Critical path risk
        critical_gaps = project_analysis['project_summary']['critical_gaps']
        if critical_gaps > 3:
            risks.append(RiskFactor(
                name="Critical Path Risk",
                category=RiskCategory.EXECUTION,
                severity=RiskSeverity.HIGH,
                probability=0.7,
                impact_score=7.0,
                description=f"{critical_gaps} critical gaps on timeline",
                mitigation_strategies=["Parallel execution", "Risk buffering", "Contingency planning"],
                cost_impact_weekly=project_info.get('cost_of_delay_weekly', 0) * 0.2
            ))

        return risks

    def _assess_organization_level_risks(self, project_risks: List[RiskProfile]) -> List[RiskFactor]:
        """Organization-level strategic risks"""

        risks = []

        # Portfolio concentration risk
        high_risk_projects = len([p for p in project_risks if p.overall_risk_score > 7.0])
        total_projects = len(project_risks)

        if total_projects > 0 and high_risk_projects / total_projects > 0.5:
            risks.append(RiskFactor(
                name="Portfolio Risk Concentration",
                category=RiskCategory.ORGANIZATIONAL,
                severity=RiskSeverity.HIGH,
                probability=0.8,
                impact_score=8.0,
                description=f"{high_risk_projects}/{total_projects} projects are high risk",
                mitigation_strategies=["Risk diversification", "Portfolio rebalancing", "Risk management program"],
                cost_impact_weekly=50000  # Organization-level impact
            ))

        return risks

    def _assess_systemic_risks(self, project_risks: List[RiskProfile]) -> List[RiskFactor]:
        """Systemic risks affecting multiple projects"""

        risks = []

        # Systemic skill shortage
        common_risk_categories = {}
        for project_risk in project_risks:
            for rf in project_risk.risk_factors:
                if rf.category not in common_risk_categories:
                    common_risk_categories[rf.category] = 0
                common_risk_categories[rf.category] += 1

        # If execution risks appear in many projects
        if common_risk_categories.get(RiskCategory.EXECUTION, 0) > len(project_risks) * 0.6:
            risks.append(RiskFactor(
                name="Systemic Execution Risk",
                category=RiskCategory.ORGANIZATIONAL,
                severity=RiskSeverity.HIGH,
                probability=0.6,
                impact_score=7.0,
                description="Execution risks span multiple projects",
                mitigation_strategies=["Capability building", "Process improvement", "Training programs"],
                cost_impact_weekly=25000
            ))

        return risks

    def _assess_capability_risks(self) -> List[RiskFactor]:
        """Organizational capability and maturity risks"""

        risks = []

        # Get organization proficiency summary
        proficiency_summary = self.proficiency_calc.get_skill_distribution_summary()
        overall_stats = proficiency_summary['overall_statistics']

        # Low average skill level risk
        avg_level = overall_stats.get('overall_avg_level', 3.0)
        if avg_level < 2.5:
            risks.append(RiskFactor(
                name="Low Organizational Skill Level",
                category=RiskCategory.ORGANIZATIONAL,
                severity=RiskSeverity.HIGH,
                probability=0.9,
                impact_score=6.0,
                description=f"Average skill level ({avg_level:.1f}) below market standards",
                mitigation_strategies=["Comprehensive training", "Strategic hiring", "Skill development programs"],
                cost_impact_weekly=15000
            ))

        # Lack of experts risk
        expert_count = overall_stats.get('expert_count', 0)
        if expert_count < 3:
            risks.append(RiskFactor(
                name="Expert Shortage",
                category=RiskCategory.ORGANIZATIONAL,
                severity=RiskSeverity.MEDIUM,
                probability=0.7,
                impact_score=5.0,
                description=f"Only {expert_count} experts in organization",
                mitigation_strategies=["Expert hiring", "External consulting", "Centers of excellence"],
                cost_impact_weekly=10000
            ))

        return risks

    def _calculate_risk_score(self, risk_factors: List[RiskFactor]) -> float:
        """Calculate overall risk score (0-10 scale)"""

        if not risk_factors:
            return 0.0

        # Weighted risk calculation
        total_weighted_risk = 0.0
        total_weight = 0.0

        for rf in risk_factors:
            # Weight by category, probability, and severity
            category_weight = self.RISK_WEIGHTS.get(rf.category, 0.2)
            severity_multiplier = self.SEVERITY_MULTIPLIERS[rf.severity]

            risk_contribution = (rf.probability * rf.impact_score * severity_multiplier * category_weight)
            total_weighted_risk += risk_contribution
            total_weight += category_weight

        if total_weight == 0:
            return 0.0

        # Normalize to 0-10 scale
        raw_score = total_weighted_risk / total_weight
        return min(10.0, raw_score)

    def _calculate_project_risk_score(self, project_factors: List[RiskFactor],
                                    skill_risks: List[RiskProfile]) -> float:
        """Calculate project-level risk score considering skill gap correlations"""

        base_score = self._calculate_risk_score(project_factors)

        # Add skill gap correlation penalty
        if len(skill_risks) > 1:
            avg_skill_risk = np.mean([sr.overall_risk_score for sr in skill_risks])
            correlation_penalty = min(2.0, len(skill_risks) * 0.2)  # Penalty for multiple high-risk skills
            base_score += correlation_penalty * (avg_skill_risk / 10.0)

        return min(10.0, base_score)

    def _calculate_organization_risk_score(self, project_risks: List[RiskProfile],
                                         org_factors: List[RiskFactor]) -> float:
        """Calculate organization-level risk score"""

        # Base organization risks
        base_score = self._calculate_risk_score(org_factors)

        # Add portfolio risk component
        if project_risks:
            avg_project_risk = np.mean([pr.overall_risk_score for pr in project_risks])
            portfolio_component = avg_project_risk * 0.6  # 60% weight to portfolio average

            # Add concentration penalty
            risk_variance = np.var([pr.overall_risk_score for pr in project_risks])
            concentration_penalty = min(1.0, risk_variance / 10.0)

            total_score = base_score + portfolio_component + concentration_penalty
        else:
            total_score = base_score

        return min(10.0, total_score)

    def _calculate_risk_distribution(self, risk_factors: List[RiskFactor]) -> Dict[str, float]:
        """Calculate risk distribution by category"""

        distribution = {category.value: 0.0 for category in RiskCategory}

        if not risk_factors:
            return distribution

        # Sum weighted risks by category
        for rf in risk_factors:
            category_key = rf.category.value
            severity_multiplier = self.SEVERITY_MULTIPLIERS[rf.severity]
            risk_contribution = rf.probability * rf.impact_score * severity_multiplier
            distribution[category_key] += risk_contribution

        # Normalize to percentages
        total_risk = sum(distribution.values())
        if total_risk > 0:
            for category in distribution:
                distribution[category] = (distribution[category] / total_risk) * 100

        return distribution

    def _calculate_confidence_level(self, risk_factors: List[RiskFactor]) -> float:
        """Calculate confidence level in risk assessment"""

        if not risk_factors:
            return 1.0

        # Factors affecting confidence
        factor_count = len(risk_factors)

        # More factors generally mean lower confidence (more uncertainty)
        count_factor = max(0.5, 1.0 - (factor_count - 5) * 0.05)

        # Severe risks are easier to assess (higher confidence)
        critical_count = len([rf for rf in risk_factors if rf.severity == RiskSeverity.CRITICAL])
        severity_factor = min(1.0, 0.8 + critical_count * 0.1)

        # External risks are harder to predict (lower confidence)
        external_count = len([rf for rf in risk_factors if rf.category == RiskCategory.EXTERNAL])
        external_factor = max(0.6, 1.0 - external_count * 0.1)

        confidence = count_factor * severity_factor * external_factor
        return max(0.3, min(1.0, confidence))

    def _generate_risk_recommendations(self, risk_factors: List[RiskFactor],
                                     gap_info: Dict[str, Any]) -> List[str]:
        """Generate risk mitigation recommendations"""

        recommendations = set()

        # Collect all mitigation strategies
        for rf in risk_factors:
            recommendations.update(rf.mitigation_strategies)

        # Add gap-specific recommendations
        if gap_info.get('gap_severity') == 'critical':
            recommendations.add("Implement emergency gap mitigation plan")

        if gap_info.get('coverage_ratio', 1.0) < 0.5:
            recommendations.add("Increase resource allocation for this skill")

        return list(recommendations)[:10]  # Top 10 recommendations

    def _generate_project_risk_recommendations(self, risk_factors: List[RiskFactor],
                                             project_info: Dict[str, Any]) -> List[str]:
        """Generate project-level risk recommendations"""

        recommendations = set()

        # High-level project recommendations
        high_severity_count = len([rf for rf in risk_factors if rf.severity == RiskSeverity.HIGH])
        critical_count = len([rf for rf in risk_factors if rf.severity == RiskSeverity.CRITICAL])

        if critical_count > 0:
            recommendations.add("Establish project risk management office")
            recommendations.add("Implement daily risk monitoring")

        if high_severity_count > 3:
            recommendations.add("Consider project scope reduction")
            recommendations.add("Implement risk-based milestone gating")

        # Category-specific recommendations
        category_counts = {}
        for rf in risk_factors:
            category_counts[rf.category] = category_counts.get(rf.category, 0) + 1

        dominant_category = max(category_counts.keys(), key=lambda x: category_counts[x]) if category_counts else None

        if dominant_category == RiskCategory.EXECUTION:
            recommendations.add("Strengthen project management capabilities")
        elif dominant_category == RiskCategory.MARKET:
            recommendations.add("Develop market risk hedging strategies")
        elif dominant_category == RiskCategory.TECHNICAL:
            recommendations.add("Increase technical review and validation")

        return list(recommendations)[:8]

    def _generate_organization_risk_recommendations(self, risk_factors: List[RiskFactor],
                                                  project_risks: List[RiskProfile]) -> List[str]:
        """Generate organization-level recommendations"""

        recommendations = set()

        # Systemic recommendations
        if len(project_risks) > 0:
            avg_project_risk = np.mean([pr.overall_risk_score for pr in project_risks])

            if avg_project_risk > 7.0:
                recommendations.add("Implement organization-wide risk management framework")
                recommendations.add("Establish risk governance committee")

            if avg_project_risk > 5.0:
                recommendations.add("Increase risk management training")
                recommendations.add("Develop risk appetite statement")

        # Capability-based recommendations
        org_risk_factors = [rf for rf in risk_factors if rf.category == RiskCategory.ORGANIZATIONAL]
        if len(org_risk_factors) > 2:
            recommendations.add("Launch organizational capability assessment")
            recommendations.add("Develop talent development strategy")

        return list(recommendations)[:6]

    def _calculate_risk_adjustment_factor(self, risk_profile: RiskProfile) -> float:
        """Calculate risk adjustment factor for financial metrics"""

        # Base adjustment from overall risk score
        base_adjustment = risk_profile.overall_risk_score / 10.0 * 0.3  # Max 30% adjustment

        # Additional adjustment for critical risks
        critical_risks = len([rf for rf in risk_profile.risk_factors
                            if rf.severity == RiskSeverity.CRITICAL])
        critical_adjustment = min(0.2, critical_risks * 0.05)  # Max 20% additional

        # Confidence adjustment (lower confidence = higher adjustment)
        confidence_adjustment = (1.0 - risk_profile.confidence_level) * 0.1  # Max 10%

        total_adjustment = base_adjustment + critical_adjustment + confidence_adjustment
        return min(0.5, total_adjustment)  # Cap at 50% adjustment

    def _extract_timeline_risk(self, risk_profile: RiskProfile) -> float:
        """Extract timeline risk factor from risk profile"""

        timeline_risks = [rf for rf in risk_profile.risk_factors
                         if 'timeline' in rf.name.lower() or 'delay' in rf.name.lower()]

        if not timeline_risks:
            return 0.0

        # Average timeline risk impact
        avg_impact = np.mean([rf.probability * rf.impact_score / 10.0 for rf in timeline_risks])
        return min(0.5, avg_impact)  # Max 50% timeline extension

    def _extract_cost_risk(self, risk_profile: RiskProfile) -> float:
        """Extract cost risk factor from risk profile"""

        cost_risks = [rf for rf in risk_profile.risk_factors
                     if 'cost' in rf.name.lower() or 'inflation' in rf.name.lower()]

        if not cost_risks:
            return 0.0

        # Average cost risk impact
        avg_impact = np.mean([rf.probability * rf.impact_score / 10.0 for rf in cost_risks])
        return min(0.4, avg_impact)  # Max 40% cost increase

    def _calculate_confidence_intervals(self, base_metrics: Dict[str, Any],
                                      risk_profile: RiskProfile) -> Dict[str, Any]:
        """Calculate confidence intervals for risk-adjusted metrics"""

        confidence_level = risk_profile.confidence_level

        intervals = {}

        # For each numeric metric, calculate confidence bounds
        for metric, value in base_metrics.items():
            if isinstance(value, (int, float)) and value != 0:
                # Use risk score to determine interval width
                risk_factor = risk_profile.overall_risk_score / 10.0

                # Wider intervals for higher risk and lower confidence
                interval_width = risk_factor * (1.0 - confidence_level) * 0.5  # Max 50% width

                intervals[f"{metric}_confidence"] = {
                    'lower_bound': value * (1 - interval_width),
                    'upper_bound': value * (1 + interval_width),
                    'confidence_level': confidence_level
                }

        return intervals

    def _identify_primary_risks(self, risk_profile: RiskProfile) -> List[str]:
        """Identify primary risk categories"""

        # Sort risks by impact
        sorted_risks = sorted(risk_profile.risk_factors,
                            key=lambda x: x.probability * x.impact_score * self.SEVERITY_MULTIPLIERS[x.severity],
                            reverse=True)

        return [rf.category.value for rf in sorted_risks[:3]]  # Top 3 risk categories

    def _prioritize_risk_mitigations(self, risk_profile: RiskProfile) -> List[Dict[str, Any]]:
        """Prioritize risk mitigation actions"""

        mitigations = []

        # Group by mitigation strategy
        strategy_impacts = {}
        for rf in risk_profile.risk_factors:
            for strategy in rf.mitigation_strategies:
                if strategy not in strategy_impacts:
                    strategy_impacts[strategy] = 0

                impact = rf.probability * rf.impact_score * self.SEVERITY_MULTIPLIERS[rf.severity]
                strategy_impacts[strategy] += impact

        # Sort by impact reduction potential
        for strategy, impact in sorted(strategy_impacts.items(), key=lambda x: x[1], reverse=True):
            mitigations.append({
                'strategy': strategy,
                'impact_reduction_potential': impact,
                'priority': 'high' if impact > 20 else 'medium' if impact > 10 else 'low'
            })

        return mitigations[:5]  # Top 5 mitigations