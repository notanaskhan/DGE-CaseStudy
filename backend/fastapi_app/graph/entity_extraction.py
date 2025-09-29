"""
Entity extraction service for Phase 5 household graph.
Extracts and normalizes entities from application data and documents.
"""

import re
import json
from typing import Dict, Any, List, Optional, Set
from dataclasses import dataclass
from pathlib import Path


@dataclass
class ExtractedEntity:
    """Represents an extracted entity with metadata."""
    entity_type: str  # 'person', 'address', 'bank_account', 'employer'
    value: str        # Raw extracted value
    normalized: str   # Normalized/cleaned value
    confidence: float # Confidence score (0.0 to 1.0)
    source: str      # Source document/field
    context: str     # Surrounding context


class EntityExtractor:
    """Extracts entities from application data and documents."""

    def __init__(self):
        # UAE-specific patterns
        self.emirates_id_pattern = re.compile(r'\b784-?\d{4}-?\d{7}-?\d\b')
        self.phone_pattern = re.compile(r'\b(?:\+971|971|0)?(?:50|51|52|55|56|2|3|4|6|7|9)\d{7}\b')
        self.iban_pattern = re.compile(r'\bAE\d{2}\s?[A-Z]{4}\s?\d{16}\b', re.IGNORECASE)
        self.account_pattern = re.compile(r'\b\d{10,16}\b')  # Bank account numbers

        # Common UAE banks for context
        self.uae_banks = {
            'emirates nbd', 'adcb', 'fab', 'enbd', 'mashreq', 'cbd', 'nbd',
            'first abu dhabi bank', 'commercial bank of dubai', 'dubai islamic bank',
            'abu dhabi commercial bank', 'rakbank', 'hsbc', 'citibank'
        }

        # Common UAE locations for address extraction
        self.uae_locations = {
            'dubai', 'abu dhabi', 'sharjah', 'ajman', 'ras al khaimah', 'fujairah',
            'umm al quwain', 'al ain', 'deira', 'bur dubai', 'jumeirah', 'marina',
            'downtown', 'business bay', 'jlt', 'jbr', 'silicon oasis'
        }

    def extract_from_application(self, app_data: Dict[str, Any]) -> List[ExtractedEntity]:
        """Extract entities from application form data."""
        entities = []

        app_row = app_data.get('app_row', {})

        # Extract person entity from application
        name = app_row.get('name', '').strip()
        emirates_id = app_row.get('emirates_id', '').strip()

        if name:
            entities.append(ExtractedEntity(
                entity_type='person',
                value=name,
                normalized=self._normalize_name(name),
                confidence=0.95,  # High confidence from application form
                source='application_form',
                context='applicant_name'
            ))

        if emirates_id:
            entities.append(ExtractedEntity(
                entity_type='emirates_id',
                value=emirates_id,
                normalized=self._normalize_emirates_id(emirates_id),
                confidence=0.98,
                source='application_form',
                context='applicant_id'
            ))

        return entities

    def extract_from_documents(self, doc_paths: List[str], ocr_data: Dict[str, Any],
                              parsed_data: Dict[str, Any]) -> List[ExtractedEntity]:
        """Extract entities from uploaded documents."""
        entities = []

        for doc_path in doc_paths:
            filename = Path(doc_path).name

            # Extract from OCR text if available
            if filename in ocr_data:
                ocr_info = ocr_data[filename]
                if ocr_info.get('ok') and 'text_path' in ocr_info and ocr_info['text_path']:
                    try:
                        with open(ocr_info['text_path'], 'r', encoding='utf-8') as f:
                            text = f.read()
                        entities.extend(self._extract_from_text(text, f"ocr_{filename}"))
                    except Exception:
                        pass  # OCR text not available

            # Extract from parsed structured data
            if filename in parsed_data:
                parsed_info = parsed_data[filename]
                if isinstance(parsed_info, dict):
                    entities.extend(self._extract_from_parsed_data(parsed_info, filename))

        return entities

    def _extract_from_text(self, text: str, source: str) -> List[ExtractedEntity]:
        """Extract entities from free text (OCR output)."""
        entities = []

        # Extract Emirates IDs
        for match in self.emirates_id_pattern.finditer(text):
            entities.append(ExtractedEntity(
                entity_type='emirates_id',
                value=match.group(),
                normalized=self._normalize_emirates_id(match.group()),
                confidence=0.9,
                source=source,
                context=self._get_context(text, match.start(), match.end())
            ))

        # Extract phone numbers
        for match in self.phone_pattern.finditer(text):
            entities.append(ExtractedEntity(
                entity_type='phone',
                value=match.group(),
                normalized=self._normalize_phone(match.group()),
                confidence=0.8,
                source=source,
                context=self._get_context(text, match.start(), match.end())
            ))

        # Extract IBANs
        for match in self.iban_pattern.finditer(text):
            entities.append(ExtractedEntity(
                entity_type='iban',
                value=match.group(),
                normalized=self._normalize_iban(match.group()),
                confidence=0.95,
                source=source,
                context=self._get_context(text, match.start(), match.end())
            ))

        # Extract potential addresses (lines containing UAE locations)
        lines = text.split('\n')
        for i, line in enumerate(lines):
            line_lower = line.lower()
            if any(loc in line_lower for loc in self.uae_locations):
                if len(line.strip()) > 10:  # Reasonable address length
                    entities.append(ExtractedEntity(
                        entity_type='address',
                        value=line.strip(),
                        normalized=self._normalize_address(line.strip()),
                        confidence=0.7,
                        source=source,
                        context=f"line_{i+1}"
                    ))

        # Extract bank names and potential account numbers
        text_lower = text.lower()
        for bank in self.uae_banks:
            if bank in text_lower:
                entities.append(ExtractedEntity(
                    entity_type='bank',
                    value=bank,
                    normalized=bank.lower().strip(),
                    confidence=0.8,
                    source=source,
                    context='bank_mention'
                ))

        return entities

    def _extract_from_parsed_data(self, parsed_data: Dict[str, Any], source: str) -> List[ExtractedEntity]:
        """Extract entities from structured parsed data (CSV, Excel, etc.)."""
        entities = []

        # For bank statements - extract account numbers from transactions
        if 'rows' in parsed_data and isinstance(parsed_data.get('rows'), (int, str)):
            # This is likely a bank statement
            entities.append(ExtractedEntity(
                entity_type='document_type',
                value='bank_statement',
                normalized='bank_statement',
                confidence=0.9,
                source=source,
                context='document_classification'
            ))

        return entities

    def _normalize_name(self, name: str) -> str:
        """Normalize person names for matching."""
        # Remove extra spaces, convert to title case
        normalized = ' '.join(name.strip().split())
        normalized = normalized.title()

        # Handle common Arabic name variations
        replacements = {
            'Mohammed': 'Mohammad',
            'Ahmed': 'Ahmad',
            'Muhammed': 'Mohammad',
            'Abdallah': 'Abdullah'
        }

        for old, new in replacements.items():
            normalized = normalized.replace(old, new)

        return normalized

    def _normalize_emirates_id(self, emirates_id: str) -> str:
        """Normalize Emirates ID format."""
        # Remove all non-digits, then format as 784-XXXX-XXXXXXX-X
        digits = re.sub(r'\D', '', emirates_id)
        if len(digits) == 15 and digits.startswith('784'):
            return f"{digits[:3]}-{digits[3:7]}-{digits[7:14]}-{digits[14]}"
        return digits

    def _normalize_phone(self, phone: str) -> str:
        """Normalize phone number format."""
        digits = re.sub(r'\D', '', phone)
        # Standardize to +971XXXXXXXXX format
        if digits.startswith('971'):
            return f"+{digits}"
        elif digits.startswith('0') and len(digits) == 10:
            return f"+971{digits[1:]}"
        elif len(digits) == 9:
            return f"+971{digits}"
        return digits

    def _normalize_iban(self, iban: str) -> str:
        """Normalize IBAN format."""
        # Remove spaces and convert to uppercase
        return re.sub(r'\s', '', iban.upper())

    def _normalize_address(self, address: str) -> str:
        """Normalize address for matching."""
        # Convert to lowercase, remove extra spaces
        normalized = ' '.join(address.lower().strip().split())

        # Standardize common abbreviations
        replacements = {
            'st.': 'street',
            'st ': 'street ',
            'rd.': 'road',
            'rd ': 'road ',
            'ave.': 'avenue',
            'ave ': 'avenue ',
            'bldg.': 'building',
            'bldg ': 'building ',
            'apt.': 'apartment',
            'apt ': 'apartment '
        }

        for old, new in replacements.items():
            normalized = normalized.replace(old, new)

        return normalized

    def _get_context(self, text: str, start: int, end: int, window: int = 50) -> str:
        """Get surrounding context for an extracted entity."""
        context_start = max(0, start - window)
        context_end = min(len(text), end + window)
        return text[context_start:context_end].replace('\n', ' ').strip()

    def extract_all_entities(self, application_data: Dict[str, Any]) -> List[ExtractedEntity]:
        """Extract all entities from application and documents."""
        entities = []

        # Extract from application form
        entities.extend(self.extract_from_application(application_data))

        # Extract from documents if available
        doc_paths = application_data.get('doc_paths', [])
        ocr_data = application_data.get('ocr', {})
        parsed_data = application_data.get('parsed', {})

        if doc_paths:
            entities.extend(self.extract_from_documents(doc_paths, ocr_data, parsed_data))

        return entities

    def group_entities_by_type(self, entities: List[ExtractedEntity]) -> Dict[str, List[ExtractedEntity]]:
        """Group extracted entities by type."""
        grouped = {}
        for entity in entities:
            if entity.entity_type not in grouped:
                grouped[entity.entity_type] = []
            grouped[entity.entity_type].append(entity)
        return grouped


# Global extractor instance
_extractor_instance = None

def get_entity_extractor() -> EntityExtractor:
    """Get or create the global entity extractor instance."""
    global _extractor_instance
    if _extractor_instance is None:
        _extractor_instance = EntityExtractor()
    return _extractor_instance