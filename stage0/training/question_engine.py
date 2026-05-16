"""
Question Engine — Generates natural language questions from detected gaps.

This is the core of the AAM feedback loop. When the system detects a gap,
it generates a question that the user (parent) can answer to teach the system.

The question engine generates MULTIPLE variations of questions for each gap,
so the system can ask in different ways and learn from different angles.
"""

from __future__ import annotations

import uuid
from typing import Dict, List, Optional

from .types import KnowledgeGap, GeneratedQuestion


# ────────────────────────────────────────────────────────────────────
# Role → Question Templates (Indonesian)
# ────────────────────────────────────────────────────────────────────

ROLE_QUESTION_TEMPLATES = {
    "Arg0Agent": {
        "primary": "Siapa yang {predicate}?",
        "variations": [
            "Siapa pelaku dari '{predicate}'?",
            "Siapa yang melakukan {predicate}?",
            "Agen dari aksi {predicate} adalah siapa?",
        ],
        "context": "Arg0Agent adalah pelaku/aktor dari aksi.",
    },
    "Arg1Patient": {
        "primary": "Apa yang di-{predicate}?",
        "variations": [
            "Objek dari '{predicate}' adalah apa?",
            "Apa yang dikenai aksi {predicate}?",
            "Pasien/target dari {predicate} adalah apa?",
        ],
        "context": "Arg1Patient adalah objek yang dikenai aksi.",
    },
    "Arg2Recipient": {
        "primary": "Kepada siapa {predicate} itu?",
        "variations": [
            "Siapa penerima dari '{predicate}'?",
            "Ditujukan kepada siapa?",
            "Setelah '{preposition}', siapa/apa yang dimaksud?",
        ],
        "context": "Arg2Recipient adalah penerima/benefisier. Biasanya setelah 'ke', 'kepada', 'untuk'.",
    },
    "Cause": {
        "primary": "Mengapa/karena apa {predicate} itu?",
        "variations": [
            "Apa penyebab dari '{predicate}'?",
            "Kenapa hal itu terjadi?",
            "Apa alasan di balik aksi tersebut?",
        ],
        "context": "Cause adalah alasan/sebab. Biasanya setelah 'karena', 'sebab', 'akibat'.",
    },
    "Purpose": {
        "primary": "Untuk apa {predicate} itu?",
        "variations": [
            "Apa tujuan dari '{predicate}'?",
            "Mengapa hal itu dilakukan? (tujuan)",
            "Supaya/agar apa?",
        ],
        "context": "Purpose adalah tujuan. Biasanya setelah 'untuk', 'agar', 'supaya'.",
    },
    "Location": {
        "primary": "Di mana {predicate} itu terjadi?",
        "variations": [
            "Lokasi dari '{predicate}'?",
            "Tempat terjadinya aksi?",
            "Di mana kejadian itu berlangsung?",
        ],
        "context": "Location adalah tempat. Biasanya setelah 'di'.",
    },
    "Instrument": {
        "primary": "Dengan apa {predicate} itu?",
        "variations": [
            "Alat yang digunakan untuk '{predicate}'?",
            "Memakai apa?",
        ],
        "context": "Instrument adalah alat/cara. Biasanya setelah 'dengan'.",
    },
    "Time": {
        "primary": "Kapan {predicate} itu terjadi?",
        "variations": [
            "Waktu terjadinya '{predicate}'?",
            "Pada saat apa?",
        ],
        "context": "Time adalah waktu kejadian.",
    },
    "Beneficiary": {
        "primary": "Siapa yang diuntungkan dari {predicate}?",
        "variations": [
            "Untuk siapa?",
            "Siapa yang mendapat manfaat?",
        ],
        "context": "Beneficiary adalah pihak yang diuntungkan.",
    },
}

AMBIGUOUS_TOKEN_TEMPLATES = {
    "primary": "Apa yang dimaksud dengan '{token}' dalam konteks ini?",
    "variations": [
        "Siapa/apa yang dirujuk oleh '{token}'?",
        "'{token}' merujuk ke siapa/apa?",
        "Jelaskan maksud '{token}' dalam kalimat ini.",
    ],
}


class QuestionEngine:
    """
    Generates natural language questions from detected gaps.

    The engine generates multiple variations of questions for each gap,
    allowing the system to ask in different ways and learn from different angles.

    Key innovation: Questions are NOT just "fill in the blank". They include:
    - Context hints so the parent knows what the system is confused about
    - Multiple variations so the system can probe from different angles
    - Pattern-based questions when similar patterns exist
    """

    def __init__(self, trainer: AAMTrainer):
        self.trainer = trainer

    def generate_questions(self, max_questions: int = 20) -> List[GeneratedQuestion]:
        """Generate questions from all unresolved gaps."""
        questions = []

        for gap in self.trainer.gaps.values():
            if gap.addressed:
                continue

            # Check if we've already asked about this gap
            gap_key = f"{gap.source_composition_id}:{gap.missing_role}"
            if gap_key in self.trainer.inquiry_memory:
                continue

            q = self._generate_question_for_gap(gap)
            if q:
                questions.append(q)

            if len(questions) >= max_questions:
                break

        return questions

    def _generate_question_for_gap(self, gap: KnowledgeGap) -> Optional[GeneratedQuestion]:
        """Generate a question for a specific gap."""
        if gap.gap_type in ("MissingRole", "MissingCause", "MissingPurpose"):
            return self._generate_missing_role_question(gap)
        elif gap.gap_type == "AmbiguousToken":
            return self._generate_ambiguous_token_question(gap)
        elif gap.gap_type == "IncompleteHiddenMeaning":
            return self._generate_hm_question(gap)
        else:
            return self._generate_generic_question(gap)

    def _generate_missing_role_question(self, gap: KnowledgeGap) -> Optional[GeneratedQuestion]:
        """Generate a question for a missing role gap."""
        role = gap.missing_role
        if not role:
            return None

        # Get the source composition
        comp = None
        if gap.source_composition_id:
            comp = self.trainer.compositions.get(gap.source_composition_id)

        # Get the predicate from the composition
        predicate_label = ""
        source_text = None
        if comp:
            pred = comp.member_with_role("Predicate")
            if pred:
                predicate_label = pred.label
            source_text = comp.source_text

        # Get templates for this role
        templates = ROLE_QUESTION_TEMPLATES.get(role, {})
        if not templates:
            # Fallback
            primary = f"Apa {role} dari kejadian ini?"
            variations = [f"Isi {role}?", f"{role} = ?"]
            context = f"{role} belum terisi."
        else:
            primary = templates["primary"].format(predicate=predicate_label or "aksi")
            variations = [
                v.format(predicate=predicate_label or "aksi", preposition="ke/karena/untuk")
                for v in templates.get("variations", [])
            ]
            context = templates.get("context", "")

        # Check if we have a pattern suggestion
        pattern_hint = self._find_pattern_hint(role, predicate_label)
        if pattern_hint:
            context += f" | Pola yang mungkin: {pattern_hint}"

        return GeneratedQuestion(
            question_id=f"q_{uuid.uuid4().hex[:8]}",
            question_text=primary,
            gap_id=gap.gap_id,
            target_role=role,
            target_composition_id=gap.source_composition_id,
            source_text=source_text,
            question_type="MissingRole",
            variations=variations,
            context_hint=context,
        )

    def _generate_ambiguous_token_question(self, gap: KnowledgeGap) -> Optional[GeneratedQuestion]:
        """Generate a question for an ambiguous token gap."""
        # Find the ambiguous token
        token_label = ""
        comp = None
        if gap.source_composition_id:
            comp = self.trainer.compositions.get(gap.source_composition_id)
            if comp and gap.missing_role:
                member = comp.member_with_role(gap.missing_role)
                if member:
                    token_label = member.label

        if not token_label:
            token_label = "token ini"

        primary = AMBIGUOUS_TOKEN_TEMPLATES["primary"].format(token=token_label)
        variations = [
            v.format(token=token_label)
            for v in AMBIGUOUS_TOKEN_TEMPLATES["variations"]
        ]

        # Find potential referents from recent compositions
        referent_hint = ""
        if comp:
            for member in comp.members:
                if member.role in ("Arg0Agent", "Arg1Patient") and member.label != token_label:
                    referent_hint += f"Mungkin merujuk ke: {member.label}. "

        return GeneratedQuestion(
            question_id=f"q_{uuid.uuid4().hex[:8]}",
            question_text=primary,
            gap_id=gap.gap_id,
            target_role=gap.missing_role or "Arg0Agent",
            target_composition_id=gap.source_composition_id,
            source_text=comp.source_text if comp else None,
            question_type="AmbiguousToken",
            variations=variations,
            context_hint=referent_hint or "Token ambigu membutuhkan klarifikasi.",
        )

    def _generate_hm_question(self, gap: KnowledgeGap) -> Optional[GeneratedQuestion]:
        """Generate a question for an incomplete hidden meaning gap."""
        role = gap.missing_role or "Problem"
        comp = None
        if gap.source_composition_id:
            comp = self.trainer.compositions.get(gap.source_composition_id)

        if role == "Problem":
            primary = "Apa masalah yang tersembunyi di balik kejadian ini?"
            variations = ["Apa pain point-nya?", "Masalah apa yang ingin diselesaikan?"]
        else:
            primary = "Apa solusi dari masalah ini?"
            variations = ["Bagaimana cara menyelesaikannya?", "Solusi apa yang diusulkan?"]

        return GeneratedQuestion(
            question_id=f"q_{uuid.uuid4().hex[:8]}",
            question_text=primary,
            gap_id=gap.gap_id,
            target_role=role,
            target_composition_id=gap.source_composition_id,
            source_text=comp.source_text if comp else None,
            question_type="IncompleteHiddenMeaning",
            variations=variations,
            context_hint="HiddenMeaning memerlukan Problem dan Solution.",
        )

    def _generate_generic_question(self, gap: KnowledgeGap) -> Optional[GeneratedQuestion]:
        """Generate a generic question for uncategorized gaps."""
        return GeneratedQuestion(
            question_id=f"q_{uuid.uuid4().hex[:8]}",
            question_text=gap.description,
            gap_id=gap.gap_id,
            target_role=gap.missing_role or "Unknown",
            target_composition_id=gap.source_composition_id,
            source_text=None,
            question_type="Generic",
            variations=[],
            context_hint="Gap umum yang memerlukan klarifikasi.",
        )

    def _find_pattern_hint(self, role: str, predicate: str) -> str:
        """Find a pattern hint from existing patterns for the given role and predicate."""
        if not predicate:
            return ""

        hints = []
        for pattern in self.trainer.patterns.values():
            if pattern.predicate == predicate and pattern.role == role:
                if pattern.observation_count >= 2:
                    hints.append(f"{pattern.filler} (×{pattern.observation_count})")

        return " atau ".join(hints[:3]) if hints else ""

    def generate_pattern_questions(self) -> List[GeneratedQuestion]:
        """
        Generate questions that verify existing patterns.

        These are NOT gap-fill questions — they verify that patterns the system
        has learned are correct. This is how the "parent" can correct wrong patterns.
        """
        questions = []

        for pattern in self.trainer.patterns.values():
            if pattern.observation_count >= 2 and pattern.epistemic in ("Inferred", "Grounded"):
                q = GeneratedQuestion(
                    question_id=f"q_verify_{uuid.uuid4().hex[:8]}",
                    question_text=f"Apakah benar bahwa setelah '{pattern.predicate}', {pattern.role} biasanya '{pattern.filler}'?",
                    gap_id=f"verify_{pattern.pattern_id}",
                    target_role=pattern.role,
                    target_composition_id=None,
                    source_text=None,
                    question_type="PatternCheck",
                    variations=[
                        f"Konfirmasi: {pattern.predicate} → {pattern.role} = {pattern.filler}?",
                        f"Benarkah pola '{pattern.predicate} + {pattern.role} = {pattern.filler}'?",
                    ],
                    context_hint=f"Pola teramati {pattern.observation_count} kali. [{pattern.lifecycle}/{pattern.epistemic}]",
                )
                questions.append(q)

        return questions
