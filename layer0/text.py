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
"""

import json
import hashlib
import datetime

from .base import (
    BasePerceptualAbstractor, PerceptualObservation, PerceptualTuple,
    ModalityType, RelationType,
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


class TextAbstractor(BasePerceptualAbstractor):
    modality = ModalityType.TEXT

    def __init__(self, llm_bridge=None):
        """
        llm_bridge: optional — instance dengan method generate(prompt: str) -> str.
        Jika None, gunakan fallback (sangat terbatas, hanya untuk smoke test).
        """
        self.llm_bridge = llm_bridge

    def abstract(self, raw_input: str, context: dict = {}) -> PerceptualObservation:
        input_ref = "text:" + hashlib.sha256(raw_input.encode()).hexdigest()[:16]
        tuples = self._extract_via_llm(raw_input) if self.llm_bridge else self._extract_fallback(raw_input)
        return PerceptualObservation(
            modality=self.modality,
            raw_input_ref=input_ref,
            tuples=tuples,
            context=context,
            timestamp=datetime.datetime.utcnow().isoformat(),
        )

    def _extract_via_llm(self, text: str) -> list[PerceptualTuple]:
        prompt = EXTRACTION_PROMPT.format(text=text)
        try:
            raw = self.llm_bridge.generate(prompt)
            data = json.loads(raw)
            return [self._dict_to_tuple(d) for d in data if isinstance(d, dict)]
        except Exception:
            return []

    def _extract_fallback(self, text: str) -> list[PerceptualTuple]:
        """Fallback sangat sederhana — hanya untuk smoke test tanpa LLM."""
        words = [w.strip(".,!?;:") for w in text.split() if len(w) > 4]
        return [self._make_categorical(w, "entity", 0.3) for w in words[:5]]

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
