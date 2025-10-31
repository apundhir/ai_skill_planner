#!/usr/bin/env python3
"""
Enhanced Proficiency Calculator with Evidence Aggregation
Implements the sophisticated proficiency model from PRD v2.0 with:
- Safe evidence aggregation with bounded normalization
- Recency decay with skill-specific rates
- Inter-rater calibration using Cohen's kappa
- Confidence intervals based on evidence reliability
"""

import sys
import os
import math
from datetime import datetime, date, timedelta
from typing import Dict, List, Any, Tuple, Optional
import sqlite3

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from database.init_db import get_db_connection


class ProficiencyCalculator:
    """
    Advanced proficiency calculator that combines base skill levels with evidence,
    recency, and rater calibration to produce effective levels with confidence intervals
    """

    # Evidence type weights (from PRD)
    EVIDENCE_WEIGHTS = {
        'production_deployment': 1.0,
        'incident_response': 0.8,
        'code_review': 0.6,
        'certification': 0.5,
        'self_project': 0.3
    }

    # Default evidence weight for unknown types
    DEFAULT_EVIDENCE_WEIGHT = 0.2

    def __init__(self, database: Optional[str] = None):
        """Initialize the proficiency calculator"""
        self.database = database

    def _calculate_evidence_score_safe(self, evidence_list: List[Dict[str, Any]]) -> float:
        """
        Safe aggregation with diminishing returns using bounded normalization
        Maps to 0-1 range using 1 - exp(-k*x) for diminishing returns
        """
        if not evidence_list:
            return 0.0

        # Calculate raw weighted score
        raw_score = sum(
            self.EVIDENCE_WEIGHTS.get(evidence.get('evidence_type', ''), self.DEFAULT_EVIDENCE_WEIGHT)
            for evidence in evidence_list
        )

        # Bounded normalization using 1 - exp(-k*x) for diminishing returns
        # k=1.2 chosen to give good spread: 1 piece ≈ 0.7, 2 pieces ≈ 0.9, 5+ pieces ≈ 1.0
        return 1 - math.exp(-1.2 * raw_score)

    def _calculate_reliability(self, evidence_list: List[Dict[str, Any]]) -> float:
        """
        Calculate evidence reliability based on verification and diversity
        Higher reliability = more confidence = lower uncertainty (σ)
        """
        if not evidence_list:
            return 0.2  # Low reliability with no evidence

        # Count verified evidence
        verified_count = sum(1 for e in evidence_list if e.get('verified_by'))
        verification_ratio = verified_count / len(evidence_list)

        # Count evidence type diversity
        unique_types = len(set(e.get('evidence_type', '') for e in evidence_list))
        type_diversity = min(unique_types / 3.0, 1.0)  # Normalize to 0-1, max at 3 types

        # Count recent evidence (last 6 months)
        recent_cutoff = date.today() - timedelta(days=180)
        recent_count = sum(
            1 for e in evidence_list
            if e.get('date_achieved') and
            (isinstance(e['date_achieved'], date) and e['date_achieved'] >= recent_cutoff or
             isinstance(e['date_achieved'], str) and
             datetime.strptime(e['date_achieved'], '%Y-%m-%d').date() >= recent_cutoff)
        )
        recency_ratio = recent_count / len(evidence_list)

        # Combine factors: verification (40%), diversity (30%), recency (30%)
        reliability = (
            0.4 * verification_ratio +
            0.3 * type_diversity +
            0.3 * recency_ratio
        )

        # Ensure minimum reliability of 0.3 with some evidence
        return max(0.3, reliability)

    def _get_rater_calibration(self, rater_id: Optional[str], skill_category: str,
                              historical_kappa: Optional[float] = None,
                              conn: Optional[sqlite3.Connection] = None) -> float:
        """
        Adjust for rater bias based on Cohen's kappa
        Returns calibration factor to multiply the base assessment
        """
        if not rater_id:
            return 0.85  # Self-assessments get reduced confidence

        # Use provided kappa or look it up in database
        if historical_kappa is None and conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT cohens_kappa FROM rater_reliability
                WHERE rater_id = ? AND skill_category = ?
            """, (rater_id, skill_category))

            result = cursor.fetchone()
            historical_kappa = result['cohens_kappa'] if result else None

        # If no historical kappa, assume moderate reliability
        if historical_kappa is None:
            return 0.90

        # Adjust based on inter-rater agreement quality
        if historical_kappa < 0.4:  # Poor agreement
            return 0.85  # Reduce confidence
        elif historical_kappa < 0.6:  # Moderate agreement
            return 0.95
        else:  # Good agreement (>= 0.6)
            return 1.0

    def calculate_effective_level(self, person_id: str, skill_id: str,
                                conn: Optional[sqlite3.Connection] = None) -> Dict[str, float]:
        """
        Calculate effective skill level with confidence intervals and calibration

        Returns:
            Dict with keys: level, confidence_low, confidence_high, reliability
        """
        # Use provided connection or create new one
        should_close_conn = conn is None
        if conn is None:
            conn = get_db_connection(self.database)

        try:
            cursor = conn.cursor()

            # Get base skill information
            cursor.execute("""
                SELECT ps.base_level, ps.last_used, ps.rater_id, ps.rating_date,
                       s.decay_rate, s.category
                FROM person_skills ps
                JOIN skills s ON ps.skill_id = s.id
                WHERE ps.person_id = ? AND ps.skill_id = ?
            """, (person_id, skill_id))

            skill_info = cursor.fetchone()
            if not skill_info:
                # Person doesn't have this skill
                return {
                    'level': 0.0,
                    'confidence_low': 0.0,
                    'confidence_high': 0.0,
                    'reliability': 0.0
                }

            skill_info = dict(skill_info)
            base_level = skill_info['base_level']
            last_used = skill_info['last_used']
            rater_id = skill_info['rater_id']
            decay_rate = skill_info['decay_rate']
            category = skill_info['category']

            # Get evidence for this person-skill combination
            cursor.execute("""
                SELECT evidence_type, date_achieved, verified_by, description
                FROM evidence
                WHERE person_id = ? AND skill_id = ?
                ORDER BY date_achieved DESC
            """, (person_id, skill_id))

            evidence_list = [dict(row) for row in cursor.fetchall()]

            # Calculate evidence score (0-1 range)
            evidence_score = self._calculate_evidence_score_safe(evidence_list)

            # Calculate recency factor with proper decay
            if last_used:
                if isinstance(last_used, str):
                    last_used_date = datetime.strptime(last_used, '%Y-%m-%d').date()
                else:
                    last_used_date = last_used

                months_since = (date.today() - last_used_date).days / 30.0
                recency_factor = math.exp(-decay_rate * months_since)
            else:
                # No usage date - assume some decay
                recency_factor = 0.8

            # Calculate mean estimate (μ) - weighted combination
            # Base level (55%) + recency-adjusted level (25%) + evidence boost (20%)
            mu = (
                0.55 * base_level +
                0.25 * (base_level * recency_factor) +
                0.20 * (2.0 * evidence_score)  # Map 0-1 evidence to 0-2 boost
            )

            # Calculate uncertainty (σ) based on evidence reliability
            reliability = self._calculate_reliability(evidence_list)
            sigma = (1.0 - reliability) * 1.0  # Uncertainty band from 0 to 1.0

            # Apply inter-rater calibration
            calibration_factor = self._get_rater_calibration(rater_id, category, conn=conn)

            # Final effective level with bounds (0-5 scale)
            effective_level = max(0, min(5, mu * calibration_factor))

            # Calculate confidence intervals
            confidence_low = max(0, effective_level - sigma)
            confidence_high = min(5, effective_level + sigma)

            return {
                'level': round(effective_level, 2),
                'confidence_low': round(confidence_low, 2),
                'confidence_high': round(confidence_high, 2),
                'reliability': round(reliability, 2)
            }

        finally:
            if should_close_conn:
                conn.close()

    def update_person_skill_levels(self, person_id: str,
                                 conn: Optional[sqlite3.Connection] = None) -> Dict[str, Dict[str, float]]:
        """
        Update all effective skill levels for a person

        Returns:
            Dict mapping skill_id to calculated levels
        """
        should_close_conn = conn is None
        if conn is None:
            conn = get_db_connection(self.database)

        try:
            cursor = conn.cursor()

            # Get all skills for this person
            cursor.execute("""
                SELECT skill_id FROM person_skills
                WHERE person_id = ?
            """, (person_id,))

            skill_ids = [row['skill_id'] for row in cursor.fetchall()]
            updated_levels = {}

            for skill_id in skill_ids:
                # Calculate new effective level
                calculated = self.calculate_effective_level(person_id, skill_id, conn)

                # Update database
                cursor.execute("""
                    UPDATE person_skills
                    SET effective_level = ?,
                        confidence_low = ?,
                        confidence_high = ?
                    WHERE person_id = ? AND skill_id = ?
                """, (calculated['level'], calculated['confidence_low'],
                      calculated['confidence_high'], person_id, skill_id))

                updated_levels[skill_id] = calculated

            conn.commit()
            return updated_levels

        finally:
            if should_close_conn:
                conn.close()

    def update_all_skill_levels(self) -> Dict[str, int]:
        """
        Update effective skill levels for all people in the database

        Returns:
            Dict with update statistics
        """
        conn = get_db_connection(self.database)

        try:
            cursor = conn.cursor()

            # Get all people
            cursor.execute("SELECT id FROM people ORDER BY id")
            person_ids = [row['id'] for row in cursor.fetchall()]

            stats = {
                'people_updated': 0,
                'skills_updated': 0,
                'average_reliability': 0.0
            }

            total_reliability = 0.0
            total_skills = 0

            print(f"🧠 Updating skill proficiency levels for {len(person_ids)} people...")

            for person_id in person_ids:
                updated_levels = self.update_person_skill_levels(person_id, conn)

                stats['people_updated'] += 1
                stats['skills_updated'] += len(updated_levels)

                # Track reliability
                for skill_data in updated_levels.values():
                    total_reliability += skill_data['reliability']
                    total_skills += 1

                print(f"  ✅ {person_id}: {len(updated_levels)} skills updated")

            # Calculate average reliability
            if total_skills > 0:
                stats['average_reliability'] = round(total_reliability / total_skills, 2)

            print(f"\n📊 Update Summary:")
            print(f"  People updated: {stats['people_updated']}")
            print(f"  Skills updated: {stats['skills_updated']}")
            print(f"  Average reliability: {stats['average_reliability']}")

            return stats

        finally:
            conn.close()

    def get_skill_distribution_summary(self) -> Dict[str, Any]:
        """
        Get summary statistics of skill levels across the organization
        """
        conn = get_db_connection(self.database)

        try:
            cursor = conn.cursor()

            # Get skill level distribution
            cursor.execute("""
                SELECT s.category,
                       COUNT(*) as skill_count,
                       ROUND(AVG(ps.base_level), 2) as avg_base_level,
                       ROUND(AVG(ps.effective_level), 2) as avg_effective_level,
                       ROUND(AVG(ps.confidence_high - ps.confidence_low), 2) as avg_uncertainty,
                       ROUND(AVG((ps.confidence_high - ps.confidence_low) / ps.effective_level * 100), 1) as avg_uncertainty_pct
                FROM person_skills ps
                JOIN skills s ON ps.skill_id = s.id
                WHERE ps.effective_level > 0
                GROUP BY s.category
                ORDER BY avg_effective_level DESC
            """)

            category_stats = [dict(row) for row in cursor.fetchall()]

            # Get overall statistics
            cursor.execute("""
                SELECT COUNT(*) as total_assessments,
                       ROUND(AVG(effective_level), 2) as overall_avg_level,
                       ROUND(AVG(confidence_high - confidence_low), 2) as overall_avg_uncertainty,
                       COUNT(*) FILTER (WHERE effective_level >= 4.0) as expert_count,
                       COUNT(*) FILTER (WHERE effective_level >= 3.0) as proficient_count,
                       COUNT(*) FILTER (WHERE effective_level < 2.0) as novice_count
                FROM person_skills
                WHERE effective_level > 0
            """)

            overall_stats = dict(cursor.fetchone())

            # Get top skills by average level
            cursor.execute("""
                SELECT s.name, s.category,
                       COUNT(*) as people_count,
                       ROUND(AVG(ps.effective_level), 2) as avg_level,
                       ROUND(AVG(ps.confidence_high - ps.confidence_low), 2) as avg_uncertainty
                FROM person_skills ps
                JOIN skills s ON ps.skill_id = s.id
                WHERE ps.effective_level > 0
                GROUP BY s.id
                HAVING people_count >= 3
                ORDER BY avg_level DESC
                LIMIT 10
            """)

            top_skills = [dict(row) for row in cursor.fetchall()]

            return {
                'category_breakdown': category_stats,
                'overall_statistics': overall_stats,
                'top_skills': top_skills,
                'generated_at': datetime.now().isoformat()
            }

        finally:
            conn.close()


if __name__ == "__main__":
    # Test the proficiency calculator
    calc = ProficiencyCalculator()

    print("🧠 Testing Proficiency Calculator...")

    # Update all skill levels
    stats = calc.update_all_skill_levels()

    # Get distribution summary
    print("\n📈 Getting skill distribution summary...")
    summary = calc.get_skill_distribution_summary()

    print(f"\n📊 Skill Distribution by Category:")
    for category in summary['category_breakdown']:
        print(f"  {category['category']}: {category['avg_effective_level']} avg level "
              f"(±{category['avg_uncertainty']}, {category['skill_count']} skills)")

    print(f"\n🎯 Overall Statistics:")
    overall = summary['overall_statistics']
    print(f"  Total assessments: {overall['total_assessments']}")
    print(f"  Average level: {overall['overall_avg_level']}")
    print(f"  Average uncertainty: {overall['overall_avg_uncertainty']}")
    print(f"  Expert level (≥4.0): {overall['expert_count']}")
    print(f"  Proficient (≥3.0): {overall['proficient_count']}")
    print(f"  Novice (<2.0): {overall['novice_count']}")

    print(f"\n🏆 Top Skills by Average Level:")
    for skill in summary['top_skills'][:5]:
        print(f"  {skill['name']}: {skill['avg_level']} "
              f"({skill['people_count']} people, ±{skill['avg_uncertainty']})")

    print(f"\n✅ Proficiency calculation complete!")