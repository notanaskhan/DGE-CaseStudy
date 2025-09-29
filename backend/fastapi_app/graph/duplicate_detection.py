"""
Duplicate detection algorithms for Phase 5 Features 3 & 4.
Detects duplicate applications based on person, address, and financial matching.
"""

import os
import json
from typing import Dict, Any, List, Optional, Tuple, Set
from dataclasses import dataclass
from datetime import datetime, timedelta

from .entity_extraction import ExtractedEntity
from .entity_normalization import EntityNormalizer, EntityMatch, get_entity_normalizer


@dataclass
class DuplicateMatch:
    """Represents a potential duplicate application."""
    application_id: str
    duplicate_application_id: str
    match_type: str  # 'person', 'address', 'financial', 'composite'
    similarity_score: float
    confidence: float
    evidence: List[Dict[str, Any]]
    risk_level: str  # 'high', 'medium', 'low'


@dataclass
class ConflictDetection:
    """Represents a data conflict within or across applications."""
    conflict_type: str  # 'income_discrepancy', 'address_mismatch', 'timeline_conflict'
    severity: str       # 'critical', 'warning', 'info'
    description: str
    evidence: Dict[str, Any]
    suggestions: List[str]


class DuplicateDetector:
    """Service for detecting duplicate applications and data conflicts."""

    def __init__(self):
        self.normalizer = get_entity_normalizer()

        # Duplicate detection thresholds
        self.person_match_threshold = 0.85
        self.address_match_threshold = 0.80
        self.financial_match_threshold = 0.90
        self.composite_match_threshold = 0.75

        # Time window for duplicate detection (days)
        self.duplicate_time_window = 90

        # Conflict detection thresholds
        self.income_variance_threshold = 0.25  # 25% variance allowed
        self.timeline_gap_threshold = 30      # 30 days for employment gaps

    def find_duplicate_applications(self, current_app_id: str,
                                  historical_applications: List[Dict[str, Any]]) -> List[DuplicateMatch]:
        """Find potential duplicate applications for the current application."""
        duplicates = []

        # Get current application entities
        current_entities = self._load_application_entities(current_app_id)
        if not current_entities:
            return duplicates

        # Check each historical application
        for hist_app in historical_applications:
            hist_app_id = hist_app.get('application_id') or hist_app.get('id')
            if not hist_app_id or hist_app_id == current_app_id:
                continue

            # Check if within time window
            if not self._within_time_window(hist_app):
                continue

            hist_entities = self._load_application_entities(hist_app_id)
            if not hist_entities:
                continue

            # Perform different types of matching
            person_match = self._detect_person_duplicate(current_entities, hist_entities)
            address_match = self._detect_address_duplicate(current_entities, hist_entities)
            financial_match = self._detect_financial_duplicate(current_entities, hist_entities)

            # Evaluate overall match
            duplicate_match = self._evaluate_composite_match(
                current_app_id, hist_app_id, person_match, address_match, financial_match
            )

            if duplicate_match:
                duplicates.append(duplicate_match)

        return sorted(duplicates, key=lambda x: x.confidence, reverse=True)

    def _detect_person_duplicate(self, current_entities: List[ExtractedEntity],
                               hist_entities: List[ExtractedEntity]) -> Optional[Dict[str, Any]]:
        """Detect if the same person applied before."""
        current_persons = [e for e in current_entities if e.entity_type == 'person']
        current_emirates_ids = [e for e in current_entities if e.entity_type == 'emirates_id']

        hist_persons = [e for e in hist_entities if e.entity_type == 'person']
        hist_emirates_ids = [e for e in hist_entities if e.entity_type == 'emirates_id']

        evidence = []
        max_similarity = 0.0

        # Check Emirates ID matches (highest confidence)
        for curr_id in current_emirates_ids:
            for hist_id in hist_emirates_ids:
                similarity = self.normalizer.calculate_similarity(curr_id, hist_id)
                if similarity >= self.person_match_threshold:
                    evidence.append({
                        'type': 'emirates_id_match',
                        'current': curr_id.normalized,
                        'historical': hist_id.normalized,
                        'similarity': similarity
                    })
                    max_similarity = max(max_similarity, similarity)

        # Check name matches
        for curr_person in current_persons:
            for hist_person in hist_persons:
                similarity = self.normalizer.calculate_similarity(curr_person, hist_person)
                if similarity >= self.person_match_threshold:
                    evidence.append({
                        'type': 'name_match',
                        'current': curr_person.normalized,
                        'historical': hist_person.normalized,
                        'similarity': similarity
                    })
                    max_similarity = max(max_similarity, similarity)

        if evidence and max_similarity >= self.person_match_threshold:
            return {
                'similarity': max_similarity,
                'evidence': evidence,
                'confidence': min(0.95, max_similarity * 1.1)  # Boost confidence for person matches
            }

        return None

    def _detect_address_duplicate(self, current_entities: List[ExtractedEntity],
                                hist_entities: List[ExtractedEntity]) -> Optional[Dict[str, Any]]:
        """Detect if applications are from the same address."""
        current_addresses = [e for e in current_entities if e.entity_type == 'address']
        hist_addresses = [e for e in hist_entities if e.entity_type == 'address']

        evidence = []
        max_similarity = 0.0

        for curr_addr in current_addresses:
            for hist_addr in hist_addresses:
                similarity = self.normalizer.calculate_similarity(curr_addr, hist_addr)
                if similarity >= self.address_match_threshold:
                    evidence.append({
                        'type': 'address_match',
                        'current': curr_addr.normalized,
                        'historical': hist_addr.normalized,
                        'similarity': similarity
                    })
                    max_similarity = max(max_similarity, similarity)

        if evidence and max_similarity >= self.address_match_threshold:
            return {
                'similarity': max_similarity,
                'evidence': evidence,
                'confidence': max_similarity * 0.8  # Address matches have medium confidence
            }

        return None

    def _detect_financial_duplicate(self, current_entities: List[ExtractedEntity],
                                  hist_entities: List[ExtractedEntity]) -> Optional[Dict[str, Any]]:
        """Detect if applications share financial accounts."""
        current_financial = [e for e in current_entities if e.entity_type in ['iban', 'bank_account']]
        hist_financial = [e for e in hist_entities if e.entity_type in ['iban', 'bank_account']]

        evidence = []
        max_similarity = 0.0

        for curr_fin in current_financial:
            for hist_fin in hist_financial:
                if curr_fin.entity_type == hist_fin.entity_type:  # Same financial type
                    similarity = self.normalizer.calculate_similarity(curr_fin, hist_fin)
                    if similarity >= self.financial_match_threshold:
                        evidence.append({
                            'type': f'{curr_fin.entity_type}_match',
                            'current': curr_fin.normalized,
                            'historical': hist_fin.normalized,
                            'similarity': similarity
                        })
                        max_similarity = max(max_similarity, similarity)

        if evidence and max_similarity >= self.financial_match_threshold:
            return {
                'similarity': max_similarity,
                'evidence': evidence,
                'confidence': max_similarity * 0.9  # Financial matches have high confidence
            }

        return None

    def _evaluate_composite_match(self, current_app_id: str, hist_app_id: str,
                                person_match: Optional[Dict], address_match: Optional[Dict],
                                financial_match: Optional[Dict]) -> Optional[DuplicateMatch]:
        """Evaluate overall duplicate match based on multiple signals."""

        matches = [m for m in [person_match, address_match, financial_match] if m is not None]
        if not matches:
            return None

        # Calculate composite score
        evidence = []
        total_weight = 0
        weighted_score = 0

        # Person matches have highest weight
        if person_match:
            weight = 0.6
            weighted_score += person_match['similarity'] * weight
            total_weight += weight
            evidence.extend(person_match['evidence'])

        # Financial matches have medium-high weight
        if financial_match:
            weight = 0.3
            weighted_score += financial_match['similarity'] * weight
            total_weight += weight
            evidence.extend(financial_match['evidence'])

        # Address matches have lower weight
        if address_match:
            weight = 0.1
            weighted_score += address_match['similarity'] * weight
            total_weight += weight
            evidence.extend(address_match['evidence'])

        if total_weight == 0:
            return None

        composite_score = weighted_score / total_weight

        if composite_score < self.composite_match_threshold:
            return None

        # Determine match type and risk level
        if person_match and person_match['similarity'] > 0.95:
            match_type = 'person'
            risk_level = 'high'
        elif financial_match and financial_match['similarity'] > 0.95:
            match_type = 'financial'
            risk_level = 'high'
        elif len(matches) >= 2:
            match_type = 'composite'
            risk_level = 'medium' if composite_score > 0.85 else 'low'
        else:
            match_type = list(matches[0]['evidence'][0]['type'].split('_'))[0]
            risk_level = 'medium' if composite_score > 0.85 else 'low'

        # Calculate final confidence
        confidence = min(0.99, composite_score * (len(matches) / 3) * 1.2)

        return DuplicateMatch(
            application_id=current_app_id,
            duplicate_application_id=hist_app_id,
            match_type=match_type,
            similarity_score=composite_score,
            confidence=confidence,
            evidence=evidence,
            risk_level=risk_level
        )

    def detect_data_conflicts(self, application_id: str,
                            application_data: Dict[str, Any]) -> List[ConflictDetection]:
        """Detect data conflicts within an application."""
        conflicts = []

        app_row = application_data.get('app_row', {})
        validation = application_data.get('validation', {})

        # Income discrepancy detection
        income_conflicts = self._detect_income_conflicts(app_row, validation)
        conflicts.extend(income_conflicts)

        # Timeline conflict detection
        timeline_conflicts = self._detect_timeline_conflicts(application_data)
        conflicts.extend(timeline_conflicts)

        # Address consistency detection
        address_conflicts = self._detect_address_conflicts(application_data)
        conflicts.extend(address_conflicts)

        return conflicts

    def _detect_income_conflicts(self, app_row: Dict[str, Any],
                               validation: Dict[str, Any]) -> List[ConflictDetection]:
        """Detect income-related conflicts."""
        conflicts = []

        declared_income = float(app_row.get('declared_monthly_income', 0))
        extracted_income = float(validation.get('monthly_inferred_income', 0))

        if declared_income > 0 and extracted_income > 0:
            # Calculate variance
            variance = abs(declared_income - extracted_income) / declared_income

            if variance > self.income_variance_threshold:
                severity = 'critical' if variance > 0.5 else 'warning'

                conflicts.append(ConflictDetection(
                    conflict_type='income_discrepancy',
                    severity=severity,
                    description=f"Declared income ({declared_income:.0f} AED) differs significantly from extracted income ({extracted_income:.0f} AED) - {variance:.1%} variance",
                    evidence={
                        'declared_income': declared_income,
                        'extracted_income': extracted_income,
                        'variance_percentage': variance * 100,
                        'threshold': self.income_variance_threshold * 100
                    },
                    suggestions=[
                        "Request additional income documentation",
                        "Verify bank statement authenticity",
                        "Schedule applicant interview for clarification"
                    ]
                ))

        return conflicts

    def _detect_timeline_conflicts(self, application_data: Dict[str, Any]) -> List[ConflictDetection]:
        """Detect timeline-related conflicts."""
        conflicts = []

        # Check for employment history gaps or inconsistencies
        validation = application_data.get('validation', {})
        flags = validation.get('flags', [])

        if isinstance(flags, list):
            employment_flags = [f for f in flags if 'employment' in f.lower()]
        else:
            employment_flags = [k for k, v in flags.items() if 'employment' in k.lower() and not v]

        if employment_flags:
            conflicts.append(ConflictDetection(
                conflict_type='timeline_conflict',
                severity='warning',
                description="Employment history inconsistencies detected in provided documents",
                evidence={
                    'employment_flags': employment_flags,
                    'validation_details': validation
                },
                suggestions=[
                    "Request employment certificate",
                    "Verify employment dates with HR department",
                    "Cross-check with Emirates ID employment history"
                ]
            ))

        return conflicts

    def _detect_address_conflicts(self, application_data: Dict[str, Any]) -> List[ConflictDetection]:
        """Detect address-related conflicts."""
        conflicts = []

        # Load extracted entities to check for address inconsistencies
        entities = self._load_application_entities(application_data.get('application_id', ''))
        addresses = [e for e in entities if e.entity_type == 'address']

        if len(addresses) > 1:
            # Check if all addresses are similar
            for i, addr1 in enumerate(addresses):
                for addr2 in addresses[i+1:]:
                    similarity = self.normalizer.calculate_similarity(addr1, addr2)
                    if similarity < 0.6:  # Addresses are quite different
                        conflicts.append(ConflictDetection(
                            conflict_type='address_mismatch',
                            severity='info',
                            description=f"Multiple different addresses found in documents: '{addr1.value}' vs '{addr2.value}'",
                            evidence={
                                'address1': addr1.normalized,
                                'address2': addr2.normalized,
                                'similarity': similarity,
                                'sources': [addr1.source, addr2.source]
                            },
                            suggestions=[
                                "Verify current residential address",
                                "Check if addresses represent different purposes (home vs work)",
                                "Request utility bill for address confirmation"
                            ]
                        ))

        return conflicts

    def _load_application_entities(self, application_id: str) -> List[ExtractedEntity]:
        """Load extracted entities for an application."""
        try:
            from backend.fastapi_app.main import APPS_DIR

            analysis_path = os.path.join(APPS_DIR, application_id, "graph_analysis.json")
            if not os.path.exists(analysis_path):
                return []

            with open(analysis_path, "r") as f:
                analysis_data = json.load(f)

            entities = []
            for entity_data in analysis_data.get('entities', []):
                entity = ExtractedEntity(
                    entity_type=entity_data['type'],
                    value=entity_data['value'],
                    normalized=entity_data['normalized'],
                    confidence=entity_data['confidence'],
                    source=entity_data['source'],
                    context=entity_data.get('context', '')
                )
                entities.append(entity)

            return entities

        except Exception:
            return []

    def _within_time_window(self, historical_app: Dict[str, Any]) -> bool:
        """Check if historical application is within the duplicate detection time window."""
        try:
            submitted_at = historical_app.get('submitted_at')
            if not submitted_at:
                return True  # Include if no timestamp available

            if isinstance(submitted_at, str):
                from datetime import datetime
                submitted_date = datetime.fromisoformat(submitted_at.replace('Z', '+00:00'))
            else:
                submitted_date = submitted_at

            time_diff = datetime.utcnow() - submitted_date.replace(tzinfo=None)
            return time_diff.days <= self.duplicate_time_window

        except Exception:
            return True  # Include if parsing fails

    def get_duplicate_statistics(self, application_id: str,
                               duplicates: List[DuplicateMatch]) -> Dict[str, Any]:
        """Generate statistics about duplicate matches."""
        if not duplicates:
            return {
                'total_duplicates': 0,
                'risk_distribution': {'high': 0, 'medium': 0, 'low': 0},
                'match_types': {},
                'avg_confidence': 0.0
            }

        risk_counts = {'high': 0, 'medium': 0, 'low': 0}
        match_type_counts = {}

        for dup in duplicates:
            risk_counts[dup.risk_level] += 1
            match_type_counts[dup.match_type] = match_type_counts.get(dup.match_type, 0) + 1

        avg_confidence = sum(d.confidence for d in duplicates) / len(duplicates)

        return {
            'total_duplicates': len(duplicates),
            'risk_distribution': risk_counts,
            'match_types': match_type_counts,
            'avg_confidence': avg_confidence,
            'highest_risk': max(duplicates, key=lambda x: x.confidence).risk_level
        }


# Global detector instance
_detector_instance = None

def get_duplicate_detector() -> DuplicateDetector:
    """Get or create the global duplicate detector instance."""
    global _detector_instance
    if _detector_instance is None:
        _detector_instance = DuplicateDetector()
    return _detector_instance