# @WHO:   self-ai/src/grammar/simple_parser.py
# @WHAT:  Simple text parser — fallback for deleted grammar/parser.py
# @PART:  grammar
# @ENTRY: SimpleParser

"""Simple text parser that replaces GrammarParser.

Provides the same API as the deleted grammar/parser.py using
basic regex and text processing instead of complex grammar rules.
This is a minimal stub so that modules importing GrammarParser
can still function without the full grammar engine.

SELF will build better parsing through teaching — this is just
a bootstrap fallback.
"""

import re
import logging
from typing import Optional, Dict, Any, List

logger = logging.getLogger(__name__)


class SimpleParser:
    """Minimal text parser using regex and heuristics.

    API mirrors the deleted GrammarParser:
      - parse(text) -> dict
      - detect_question_roles(question) -> dict
    """

    # Role keyword mappings for Indonesian math word problems
    ROLE_KEYWORDS = {
        'initial': ['awalnya', 'mula-mula', 'semula', 'awal', 'punya', 'memiliki', 'ada'],
        'consumed': ['dimakan', 'dipakai', 'dijual', 'terjual', 'diberikan', 'dibuang', 'hilang', 'keluar'],
        'remaining': ['sisa', 'tersisa', 'masih', 'tinggal', 'belum'],
        'added': ['ditambah', 'datang lagi', 'beli lagi', 'diperoleh', 'masuk', 'diberi'],
        'total': ['total', 'seluruh', 'semua', 'jumlah'],
        'shared_each': ['masing-masing', 'setiap', 'per', 'tiap'],
        'group_count': ['kelompok', 'bagian', 'dibagi', 'distribusi'],
        'price': ['harga', 'rp', 'bayar', 'kembalian', 'dibayar', 'beli'],
        'perimeter_area': ['keliling', 'luas', 'sisi', 'persegi'],
        'time_duration': ['pukul', 'menit', 'jam', 'lama'],
        'difference': ['selisih', 'beda'],
        'divide_keyword': ['dibagi', 'dibagikan', 'distribusikan'],
        'multiply_keyword': ['dikali', 'kali', 'per'],
        'fraction': ['seper', 'setengah', 'sepertiga', 'seperempat', 'per'],
    }

    # Question role keywords
    QUESTION_ROLE_KEYWORDS = {
        'question_who': ['siapa', 'whose'],
        'question_what': ['apa', 'what'],
        'question_why': ['mengapa', 'kenapa', 'why'],
        'question_how': ['bagaimana', 'how'],
        'question_when': ['kapan', 'when'],
        'question_where': ['di mana', 'dimana', 'where'],
        'question_message': ['amanat', 'pesan', 'message'],
        'question_feeling': ['perasaan', 'merasa', 'feeling'],
    }

    def __init__(self, self_core=None):
        self.self_core = self_core

    def parse(self, text: str) -> dict:
        """Parse text into numbers, roles, fractions, and extracted facts.

        Returns dict with same structure as GrammarParser.parse():
            numbers: list of dicts {value, raw, position}
            roles: dict of {role_name: [keywords_found]}
            fractions: list of dicts {value, raw}
            operation_hint: str
            extracted_facts: list of dicts (per-sentence parse results)
            triplets: list of dicts {subject, predicate, object}
        """
        if not text:
            return {
                'numbers': [],
                'roles': {},
                'fractions': [],
                'operation_hint': 'UNKNOWN',
                'extracted_facts': [],
                'triplets': [],
            }

        # Extract numbers
        numbers = self._extract_numbers(text)

        # Extract fractions
        fractions = self._extract_fractions(text)

        # Detect roles
        roles = self._detect_roles(text)

        # Infer operation hint
        operation_hint = self._infer_operation_hint(roles, numbers, text)

        # Extract facts per sentence
        extracted_facts = self._extract_facts(text, numbers, roles, fractions)

        # Extract simple SPO triplets
        triplets = self._extract_triplets(text, numbers, roles)

        return {
            'numbers': numbers,
            'roles': roles,
            'fractions': fractions,
            'operation_hint': operation_hint,
            'extracted_facts': extracted_facts,
            'triplets': triplets,
        }

    def detect_question_roles(self, question: str) -> dict:
        """Detect what type of question this is.

        Returns dict of {role_name: True} for matched roles.
        """
        q_lower = question.lower()
        detected = {}

        for role, keywords in self.QUESTION_ROLE_KEYWORDS.items():
            for kw in keywords:
                if kw in q_lower:
                    detected[role] = True
                    break

        return detected

    # ── Internal helpers ──────────────────────────────────────

    def _extract_numbers(self, text: str) -> list:
        """Extract numbers from text with position info."""
        numbers = []
        for match in re.finditer(r'(\d+\.?\d*)', text):
            raw = match.group(1)
            try:
                value = float(raw)
                if value == int(value):
                    value = int(value)
            except ValueError:
                continue
            numbers.append({
                'value': value,
                'raw': raw,
                'position': match.start(),
                'is_time': self._is_time_context(text, match.start()),
            })
        return numbers

    def _is_time_context(self, text: str, pos: int) -> bool:
        """Check if a number at position is a time value."""
        before = text[max(0, pos - 10):pos].lower()
        return 'pukul' in before or ':' in before

    def _extract_fractions(self, text: str) -> list:
        """Extract fraction values from text."""
        fractions = []

        # Pattern: "1/4", "3/5", etc.
        for match in re.finditer(r'(\d+)\s*/\s*(\d+)', text):
            num = int(match.group(1))
            den = int(match.group(2))
            if den > 0:
                fractions.append({
                    'value': num / den,
                    'raw': match.group(0),
                })

        # Pattern: "setengah" = 0.5, "sepertiga" = 1/3, etc.
        fraction_words = {
            'setengah': 0.5, 'seperdua': 0.5,
            'sepertiga': 1/3, 'seperempat': 0.25,
            'seperlima': 0.2, 'seperenam': 1/6,
        }
        for word, val in fraction_words.items():
            if word in text.lower():
                fractions.append({'value': val, 'raw': word})

        return fractions

    def _detect_roles(self, text: str) -> dict:
        """Detect operational roles from text keywords."""
        text_lower = text.lower()
        roles = {}

        for role, keywords in self.ROLE_KEYWORDS.items():
            found = []
            for kw in keywords:
                if kw in text_lower:
                    found.append(kw)
            if found:
                roles[role] = found

        return roles

    def _infer_operation_hint(self, roles: dict, numbers: list, text: str) -> str:
        """Infer operation from roles and number patterns."""
        if 'divide_keyword' in roles:
            return 'DIVIDE'
        if 'multiply_keyword' in roles:
            return 'MULTIPLY'
        if 'fraction' in roles:
            return 'FRACTION_MULTIPLY'
        if 'consumed' in roles and ('remaining' in roles or 'initial' in roles):
            return 'SUBTRACT'
        if 'initial' in roles and 'added' in roles:
            return 'ADD'
        if 'added' in roles:
            return 'ADD'
        if 'difference' in roles:
            return 'SUBTRACT'
        if 'shared_each' in roles:
            return 'MULTIPLY'
        if 'price' in roles:
            return 'MULTIPLY'
        if 'perimeter_area' in roles:
            roles_str = str(roles.get('perimeter_area', []))
            if 'keliling' in roles_str:
                return 'PERIMETER'
            if 'luas' in roles_str or 'sisi' in roles_str:
                return 'SQUARE_AREA'
            return 'MULTIPLY'
        if 'time_duration' in roles:
            return 'TIME_DURATION'
        if 'total' in roles and 'group_count' in roles:
            return 'DIVIDE'
        return 'UNKNOWN'

    def _extract_facts(self, text: str, numbers: list, roles: dict,
                       fractions: list) -> list:
        """Split text into sentences and extract per-sentence facts."""
        sentences = re.split(r'[.!?]\s*', text)
        facts = []

        for sent in sentences:
            sent = sent.strip()
            if not sent:
                continue

            sent_numbers = self._extract_numbers(sent)
            sent_roles = self._detect_roles(sent)
            sent_fractions = self._extract_fractions(sent)
            sent_op = self._infer_operation_hint(sent_roles, sent_numbers, sent)

            if sent_numbers or sent_roles:
                facts.append({
                    'text': sent,
                    'numbers': sent_numbers,
                    'roles': sent_roles,
                    'fractions': sent_fractions,
                    'operation_hint': sent_op,
                })

        return facts

    def _extract_triplets(self, text: str, numbers: list, roles: dict) -> list:
        """Extract simple subject-predicate-object triplets.

        Very basic: looks for "A verb B" patterns and number relationships.
        """
        triplets = []
        sentences = re.split(r'[.!?]\s*', text)

        for sent in sentences:
            sent = sent.strip()
            if not sent:
                continue

            # Try to find "A adalah B", "A memiliki B", etc.
            copula_patterns = [
                (r'(\w+)\s+(adalah|ialah)\s+(.+)', 'IS_A'),
                (r'(\w+)\s+(memiliki|punya|mempunyai)\s+(.+)', 'HAS'),
                (r'(\w+)\s+(bukan)\s+(.+)', 'NOT'),
            ]

            for pattern, relation in copula_patterns:
                match = re.search(pattern, sent, re.IGNORECASE)
                if match:
                    triplets.append({
                        'subject': match.group(1),
                        'predicate': relation,
                        'object': match.group(3).strip(),
                        'confidence': 0.6,
                    })
                    break

        return triplets
