"""
Entity normalization utilities with fuzzy matching for Phase 5.
Handles entity deduplication and similarity matching.
"""

import re
from typing import Dict, List, Tuple, Optional, Set
from dataclasses import dataclass
from difflib import SequenceMatcher

try:
    from fuzzywuzzy import fuzz
    FUZZYWUZZY_AVAILABLE = True
except ImportError:
    FUZZYWUZZY_AVAILABLE = False

try:
    import Levenshtein
    LEVENSHTEIN_AVAILABLE = True
except ImportError:
    LEVENSHTEIN_AVAILABLE = False

from .entity_extraction import ExtractedEntity


@dataclass
class EntityMatch:
    """Represents a match between two entities."""
    entity1: ExtractedEntity
    entity2: ExtractedEntity
    similarity_score: float
    match_type: str  # 'exact', 'fuzzy', 'phonetic', 'partial'
    confidence: float


class EntityNormalizer:
    """Normalizes and matches entities for deduplication."""

    def __init__(self):
        # Thresholds for different entity types
        self.similarity_thresholds = {
            'person': 0.85,      # Names need high similarity
            'emirates_id': 0.95,  # IDs need very high similarity
            'address': 0.80,     # Addresses can have more variation
            'phone': 0.90,       # Phone numbers need high similarity
            'iban': 0.95,        # IBANs need very high similarity
            'bank': 0.85,        # Bank names need high similarity
        }

        # Common name variations for phonetic matching
        self.name_variations = {
            'mohammad': ['mohammed', 'muhammed', 'mohamed'],
            'ahmad': ['ahmed', 'ahmeed'],
            'abdullah': ['abdallah', 'abd allah'],
            'abdul': ['abd ul', 'abd al'],
            'ali': ['aly'],
            'hassan': ['hasan'],
            'hussain': ['hussein', 'husain'],
            'fatima': ['fatma', 'fatmah'],
            'aisha': ['aysha', 'aishah'],
            'mariam': ['maryam', 'mary'],
        }

        # Build reverse lookup
        self.name_variations_reverse = {}
        for canonical, variants in self.name_variations.items():
            self.name_variations_reverse[canonical] = canonical
            for variant in variants:
                self.name_variations_reverse[variant] = canonical

    def calculate_similarity(self, entity1: ExtractedEntity, entity2: ExtractedEntity) -> float:
        """Calculate similarity score between two entities."""
        if entity1.entity_type != entity2.entity_type:
            return 0.0

        # Use different similarity methods based on entity type
        if entity1.entity_type == 'person':
            return self._calculate_name_similarity(entity1.normalized, entity2.normalized)
        elif entity1.entity_type == 'emirates_id':
            return self._calculate_exact_similarity(entity1.normalized, entity2.normalized)
        elif entity1.entity_type == 'address':
            return self._calculate_address_similarity(entity1.normalized, entity2.normalized)
        elif entity1.entity_type in ['phone', 'iban']:
            return self._calculate_exact_similarity(entity1.normalized, entity2.normalized)
        else:
            return self._calculate_fuzzy_similarity(entity1.normalized, entity2.normalized)

    def _calculate_name_similarity(self, name1: str, name2: str) -> float:
        """Calculate similarity for person names with phonetic matching."""
        name1 = name1.lower().strip()
        name2 = name2.lower().strip()

        # Exact match
        if name1 == name2:
            return 1.0

        # Check phonetic variations
        phonetic_score = self._calculate_phonetic_similarity(name1, name2)
        if phonetic_score > 0.8:
            return phonetic_score

        # Fuzzy string matching
        fuzzy_score = self._calculate_fuzzy_similarity(name1, name2)

        # Token-based matching (for names with different word orders)
        token_score = self._calculate_token_similarity(name1, name2)

        # Return highest score
        return max(phonetic_score, fuzzy_score, token_score)

    def _calculate_phonetic_similarity(self, name1: str, name2: str) -> float:
        """Calculate phonetic similarity using name variations."""
        # Normalize names to canonical forms
        canonical1 = self._get_canonical_name(name1)
        canonical2 = self._get_canonical_name(name2)

        if canonical1 == canonical2:
            return 0.95  # High but not perfect since it's phonetic

        # Check if names are variations of each other
        tokens1 = set(canonical1.split())
        tokens2 = set(canonical2.split())

        common_tokens = tokens1.intersection(tokens2)
        total_tokens = tokens1.union(tokens2)

        if total_tokens:
            return len(common_tokens) / len(total_tokens)

        return 0.0

    def _get_canonical_name(self, name: str) -> str:
        """Convert name to canonical form using variations map."""
        tokens = name.lower().split()
        canonical_tokens = []

        for token in tokens:
            canonical = self.name_variations_reverse.get(token, token)
            canonical_tokens.append(canonical)

        return ' '.join(canonical_tokens)

    def _calculate_address_similarity(self, addr1: str, addr2: str) -> float:
        """Calculate similarity for addresses."""
        addr1 = addr1.lower().strip()
        addr2 = addr2.lower().strip()

        if addr1 == addr2:
            return 1.0

        # Token-based matching for addresses
        token_score = self._calculate_token_similarity(addr1, addr2)

        # Fuzzy matching
        fuzzy_score = self._calculate_fuzzy_similarity(addr1, addr2)

        # Check for substring matches (building number + street)
        substring_score = self._calculate_substring_similarity(addr1, addr2)

        return max(token_score, fuzzy_score, substring_score)

    def _calculate_exact_similarity(self, val1: str, val2: str) -> float:
        """Calculate exact similarity (for IDs, phone numbers)."""
        if val1 == val2:
            return 1.0

        # For emirates IDs, allow minor formatting differences
        if len(val1) == len(val2) and len(val1) > 10:
            # Remove all non-alphanumeric and compare
            clean1 = re.sub(r'[^0-9A-Za-z]', '', val1)
            clean2 = re.sub(r'[^0-9A-Za-z]', '', val2)
            if clean1 == clean2:
                return 0.98

        return 0.0

    def _calculate_fuzzy_similarity(self, val1: str, val2: str) -> float:
        """Calculate fuzzy similarity using available libraries."""
        if not val1 or not val2:
            return 0.0

        # Use fuzzywuzzy if available
        if FUZZYWUZZY_AVAILABLE:
            return fuzz.ratio(val1, val2) / 100.0

        # Use Levenshtein if available
        if LEVENSHTEIN_AVAILABLE:
            max_len = max(len(val1), len(val2))
            if max_len == 0:
                return 1.0
            distance = Levenshtein.distance(val1, val2)
            return 1.0 - (distance / max_len)

        # Fallback to SequenceMatcher
        return SequenceMatcher(None, val1, val2).ratio()

    def _calculate_token_similarity(self, val1: str, val2: str) -> float:
        """Calculate similarity based on common tokens."""
        tokens1 = set(val1.split())
        tokens2 = set(val2.split())

        if not tokens1 and not tokens2:
            return 1.0
        if not tokens1 or not tokens2:
            return 0.0

        common = tokens1.intersection(tokens2)
        total = tokens1.union(tokens2)

        return len(common) / len(total)

    def _calculate_substring_similarity(self, val1: str, val2: str) -> float:
        """Calculate similarity based on longest common substring."""
        if not val1 or not val2:
            return 0.0

        # Find longest common substring
        max_len = 0
        m, n = len(val1), len(val2)

        # Create a table to store lengths of common substrings
        dp = [[0] * (n + 1) for _ in range(m + 1)]

        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if val1[i-1] == val2[j-1]:
                    dp[i][j] = dp[i-1][j-1] + 1
                    max_len = max(max_len, dp[i][j])

        # Return ratio of longest common substring to average length
        avg_len = (len(val1) + len(val2)) / 2
        return max_len / avg_len if avg_len > 0 else 0.0

    def find_matches(self, entities: List[ExtractedEntity]) -> List[EntityMatch]:
        """Find matching entities in a list."""
        matches = []

        # Group entities by type for efficiency
        by_type = {}
        for entity in entities:
            if entity.entity_type not in by_type:
                by_type[entity.entity_type] = []
            by_type[entity.entity_type].append(entity)

        # Find matches within each type
        for entity_type, type_entities in by_type.items():
            threshold = self.similarity_thresholds.get(entity_type, 0.8)

            for i in range(len(type_entities)):
                for j in range(i + 1, len(type_entities)):
                    entity1 = type_entities[i]
                    entity2 = type_entities[j]

                    similarity = self.calculate_similarity(entity1, entity2)

                    if similarity >= threshold:
                        match_type = self._determine_match_type(similarity)
                        confidence = min(entity1.confidence, entity2.confidence) * similarity

                        matches.append(EntityMatch(
                            entity1=entity1,
                            entity2=entity2,
                            similarity_score=similarity,
                            match_type=match_type,
                            confidence=confidence
                        ))

        return matches

    def _determine_match_type(self, similarity: float) -> str:
        """Determine the type of match based on similarity score."""
        if similarity >= 0.99:
            return 'exact'
        elif similarity >= 0.9:
            return 'fuzzy'
        elif similarity >= 0.8:
            return 'phonetic'
        else:
            return 'partial'

    def deduplicate_entities(self, entities: List[ExtractedEntity]) -> List[ExtractedEntity]:
        """Remove duplicate entities, keeping the highest confidence one."""
        if not entities:
            return []

        matches = self.find_matches(entities)

        # Create groups of matching entities
        entity_groups = {}
        entity_to_group = {}

        # Initialize each entity in its own group
        for i, entity in enumerate(entities):
            entity_groups[i] = [entity]
            entity_to_group[id(entity)] = i

        # Merge groups based on matches
        for match in matches:
            group1 = entity_to_group[id(match.entity1)]
            group2 = entity_to_group[id(match.entity2)]

            if group1 != group2:
                # Merge group2 into group1
                entity_groups[group1].extend(entity_groups[group2])

                # Update mappings
                for entity in entity_groups[group2]:
                    entity_to_group[id(entity)] = group1

                # Remove the merged group
                del entity_groups[group2]

        # Select best entity from each group
        deduplicated = []
        for group_entities in entity_groups.values():
            if group_entities:
                # Choose entity with highest confidence
                best_entity = max(group_entities, key=lambda e: e.confidence)
                deduplicated.append(best_entity)

        return deduplicated

    def normalize_entity_value(self, entity: ExtractedEntity) -> str:
        """Get the normalized value for an entity."""
        return entity.normalized


# Global normalizer instance
_normalizer_instance = None

def get_entity_normalizer() -> EntityNormalizer:
    """Get or create the global entity normalizer instance."""
    global _normalizer_instance
    if _normalizer_instance is None:
        _normalizer_instance = EntityNormalizer()
    return _normalizer_instance