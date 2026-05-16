"""
AAM Layer 0 — Text Perceptual Abstractor

Input : raw text string
Output: PerceptualObservation (list of PerceptualTuple)

Proses:
  1. Kirim teks ke LLM (Mother model) dengan prompt ekstraksi terstruktur
  2. Parse JSON response → list PerceptualTuple
  3. Return PerceptualObservation (raw teks TIDAK disimpan)

LLM di sini bukan untuk menjawab — hanya untuk menghasilkan structured tuples.
Ini adalah "live perception" yang langsung dikompres ke relasi.

L0-04: Added retry with exponential backoff, improved noun-phrase fallback,
       simple caching for repeated text.
"""

import json
import hashlib
import datetime
import re
import time

from .base import (
    BasePerceptualAbstractor, PerceptualObservation, PerceptualTuple,
    PerceptualTupleMeta, ModalityType, RelationType,
)

EXTRACTION_PROMPT = """You are a perceptual abstractor. Extract structured relational tuples from text.

Output ONLY a JSON array. No explanation. No markdown fences.

Each object must have:
  subject       : string — the main entity
  relation_type : one of [categorical, differential, functional, spatial, temporal, causal]
  predicate     : string — what the subject relates to
  dimension     : string or null — (differential only) what dimension is compared
  direction     : string or null — (differential only) comparison direction
  confidence    : float 0.0-1.0

Example input: "An apple is a fruit. It is rounder than a pear. Apples can be eaten."
Example output:
[
  {{"subject":"apple","relation_type":"categorical","predicate":"fruit","confidence":0.99,"dimension":null,"direction":null}},
  {{"subject":"apple","relation_type":"differential","predicate":"pear","dimension":"shape","direction":"rounder","confidence":0.90}},
  {{"subject":"apple","relation_type":"functional","predicate":"edible","confidence":0.99,"dimension":null,"direction":null}}
]

Text to extract from:
{text}"""

# L0-04: Retry configuration
MAX_RETRIES = 3
RETRY_DELAYS = [1.0, 2.0, 4.0]  # Exponential backoff in seconds

# L0-04: Noun phrase extraction pattern for improved fallback
# Matches: optional determiner + optional adjectives + noun (2+ chars)
_NOUN_PHRASE_RE = re.compile(
    r'\b(?:the|a|an)?\s*(?:\w+(?:ing|ed|al|ous|ive|able|ible)\s+)*'
    r'([A-Z]?[a-z]{2,}(?:s|es|ion|ment|ness|ity|ence|ance)?)\b',
    re.IGNORECASE,
)


class TextAbstractor(BasePerceptualAbstractor):
    modality = ModalityType.TEXT

    def __init__(self, llm_bridge=None):
        """
        llm_bridge: optional — instance dengan method generate(prompt: str) -> str.
        Jika None, gunakan fallback (sangat terbatas, hanya untuk smoke test).
        """
        self.llm_bridge = llm_bridge
        # L0-04: Simple in-memory cache for repeated text
        self._cache: dict[str, list[PerceptualTuple]] = {}

    def abstract(self, raw_input: str, context: dict = {}) -> PerceptualObservation:
        input_ref = "text:" + hashlib.sha256(raw_input.encode()).hexdigest()[:16]

        # L0-04: Check cache first
        cache_key = input_ref
        if cache_key in self._cache:
            tuples = list(self._cache[cache_key])  # Return a copy
        elif self.llm_bridge:
            tuples = self._extract_via_llm(raw_input)
        else:
            tuples = self._extract_fallback(raw_input)

        # L0-04: Cache successful extractions
        if tuples and cache_key not in self._cache:
            self._cache[cache_key] = tuples

        # L0-06: Set metadata on each tuple
        meta = PerceptualTupleMeta(
            source_url=input_ref,
            extraction_model="llm" if self.llm_bridge else "fallback",
            extraction_timestamp=time.time(),
        )
        for t in tuples:
            if isinstance(t.metadata, dict) and not t.metadata:
                t.metadata = meta

        return PerceptualObservation(
            modality=self.modality,
            raw_input_ref=input_ref,
            tuples=tuples,
            context=context,
            timestamp=datetime.datetime.utcnow().isoformat(),
        )

    def _extract_via_llm(self, text: str) -> list[PerceptualTuple]:
        """L0-04: Extract via LLM with retry + exponential backoff."""
        prompt = EXTRACTION_PROMPT.format(text=text)

        last_error = None
        for attempt in range(MAX_RETRIES):
            try:
                raw = self.llm_bridge.generate(prompt)
                data = json.loads(raw)
                if not isinstance(data, list):
                    raise ValueError(f"Expected list, got {type(data).__name__}")
                tuples = [self._dict_to_tuple(d) for d in data if isinstance(d, dict)]
                if tuples:
                    return tuples
                # Empty list from valid JSON — treat as LLM failure
                last_error = ValueError("LLM returned empty tuple list")
            except (json.JSONDecodeError, ValueError, TypeError) as exc:
                last_error = exc
            except Exception as exc:
                last_error = exc

            # L0-04: Exponential backoff before retry
            if attempt < MAX_RETRIES - 1:
                time.sleep(RETRY_DELAYS[attempt])

        # All retries failed — use improved fallback instead of returning []
        return self._extract_fallback(text)

    def _extract_fallback(self, text: str) -> list[PerceptualTuple]:
        """
        L0-04: Improved fallback using noun phrase extraction.
        Previously: words > 4 chars → meaningless "entity" categoricals.
        Now: extract noun phrases, create categorical + functional tuples.
        """
        tuples: list[PerceptualTuple] = []

        # Extract noun phrases using regex
        phrases = _NOUN_PHRASE_RE.findall(text)
        seen: set[str] = set()

        for phrase in phrases:
            phrase = phrase.strip()
            if not phrase or len(phrase) < 2 or phrase.lower() in seen:
                continue
            seen.add(phrase.lower())

            # Create a categorical tuple: "X is an entity"
            tuples.append(self._make_categorical(phrase, "entity", 0.3))

        # Also extract simple "X can Y" / "X is Y" patterns
        is_pattern = re.findall(r'(\b[A-Z]?[a-z]{2,}\b)\s+is\s+(?:a\s+|an\s+)?(\b[A-Z]?[a-z]{2,}\b)', text)
        for subject, category in is_pattern:
            if subject.lower() not in seen:
                tuples.append(self._make_categorical(subject, category, 0.5))

        can_pattern = re.findall(r'(\b[A-Z]?[a-z]{2,}\b)\s+can\s+(?:be\s+)?(\b[A-Z]?[a-z]{2,}\b)', text)
        for subject, function in can_pattern:
            tuples.append(self._make_functional(subject, function, 0.5))

        # Limit to reasonable number
        return tuples[:10]

    def _dict_to_tuple(self, d: dict) -> PerceptualTuple:
        return PerceptualTuple(
            subject=d.get("subject", ""),
            relation_type=RelationType(d.get("relation_type", "categorical")),
            predicate=d.get("predicate", ""),
            dimension=d.get("dimension"),
            direction=d.get("direction"),
            confidence=float(d.get("confidence", 1.0)),
            source_modality=self.modality,
        )
