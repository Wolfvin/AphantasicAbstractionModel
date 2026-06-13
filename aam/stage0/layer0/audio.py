"""
AAM Layer 0 — Audio Perceptual Abstractor

L0-03: Implemented as Whisper STT → TextAbstractor pipeline.

Strategy:
  - Speech audio: Transcribe via Whisper (or STT bridge), then pipe
    transcription through TextAbstractor for tuple extraction.
  - Non-speech audio: Extract basic temporal + causal tuples from
    audio metadata (duration, format, etc.).
  - No raw audio stored — only PerceptualTuple results.

If no STT bridge is available, falls back to metadata extraction
from the audio reference (path/filename cues).
"""

import hashlib
import datetime
import os

from .base import (
    BasePerceptualAbstractor, PerceptualObservation, PerceptualTuple,
    PerceptualTupleMeta, ModalityType, RelationType,
)
from .text import TextAbstractor


# Prompt for audio description extraction via LLM
AUDIO_DESCRIPTION_PROMPT = """You are an audio content analyzer. Given a description of audio content, extract structured relational tuples.

Output ONLY a JSON array. No explanation. No markdown fences.

Each object must have:
  subject       : string — the main entity or concept
  relation_type : one of [categorical, differential, functional, spatial, temporal, causal]
  predicate     : string — what the subject relates to
  dimension     : string or null
  direction     : string or null
  confidence    : float 0.0-1.0

Focus on temporal and causal relations that are common in audio:
- "speech occurs before music" (temporal)
- "thunder causes alarm" (causal)
- "dialogue mentions weather" (categorical)

Audio description:
{text}"""


class AudioAbstractor(BasePerceptualAbstractor):
    modality = ModalityType.AUDIO

    def __init__(self, stt_bridge=None, llm_bridge=None):
        """
        stt_bridge: optional — instance with method transcribe(audio_input) -> str.
                    If None, metadata-only extraction is used.
        llm_bridge: optional — instance with method generate(prompt: str) -> str.
                    Used for structured extraction from transcriptions.
        """
        self.stt_bridge = stt_bridge
        self._text_abstractor = TextAbstractor(llm_bridge=llm_bridge)

    def abstract(self, raw_input, context: dict = {}) -> PerceptualObservation:
        """
        Extract PerceptualTuples from audio input.

        Args:
            raw_input: Audio data (bytes, file path, or audio reference).
                       If bytes/path, stt_bridge handles transcription.
                       If str, treated as an audio description.
            context: Optional context dict with metadata.

        Returns:
            PerceptualObservation with extracted tuples.
        """
        # Compute reference hash
        if isinstance(raw_input, bytes):
            input_ref = "audio:" + hashlib.sha256(raw_input).hexdigest()[:16]
        elif isinstance(raw_input, str):
            input_ref = "audio:" + hashlib.sha256(raw_input.encode()).hexdigest()[:16]
        else:
            input_ref = "audio:unknown"

        tuples: list[PerceptualTuple] = []

        # Strategy 1: STT bridge → TextAbstractor pipeline
        if self.stt_bridge and isinstance(raw_input, (bytes, str)):
            transcription = self._transcribe(raw_input)
            if transcription:
                # Pipe transcription through TextAbstractor
                text_obs = self._text_abstractor.abstract(transcription, context)
                # Remap tuples to AUDIO modality
                for t in text_obs.tuples:
                    t.source_modality = ModalityType.AUDIO
                    tuples.append(t)

                # Add temporal tuple for the transcription event
                tuples.append(self._make_temporal(
                    "transcription", f"derived from audio {input_ref}", 0.95
                ))

        # Strategy 2: LLM-driven structured extraction from description
        elif self._text_abstractor.llm_bridge and isinstance(raw_input, str):
            # Treat string input as audio description
            desc_obs = self._text_abstractor.abstract(raw_input, context)
            for t in desc_obs.tuples:
                t.source_modality = ModalityType.AUDIO
                tuples.append(t)

        # Strategy 3: Fallback — metadata extraction
        if not tuples:
            tuples = self._extract_metadata_fallback(raw_input, input_ref)

        # Set metadata
        meta = PerceptualTupleMeta(
            source_url=input_ref,
            extraction_model="whisper+text" if self.stt_bridge else "fallback",
            extraction_timestamp=datetime.datetime.utcnow().timestamp(),
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

    def _transcribe(self, audio_input) -> str | None:
        """Transcribe audio using the STT bridge.

        # STUB:REQUIRES_EXTERNAL_BRIDGE — Needs an ASR (Automatic Speech
        # Recognition) engine like Whisper to transcribe audio input.
        # Without it, only metadata-level extraction is available.
        """
        try:
            return self.stt_bridge.transcribe(audio_input)
        except Exception:
            return None

    def _extract_metadata_fallback(self, raw_input, input_ref: str) -> list[PerceptualTuple]:
        """Fallback: extract basic tuples from audio metadata/filename cues.

        # STUB:REQUIRES_EXTERNAL_BRIDGE — This fallback produces minimal tuples
        # because no ASR/STT bridge is available. Real audio understanding
        # requires connecting an external speech-to-text model (Whisper, etc.).
        """
        tuples: list[PerceptualTuple] = []

        # At minimum, create a categorical tuple for the audio itself
        tuples.append(self._make_categorical(input_ref, "audio", 0.3))

        # Try to extract cues from the reference string
        if isinstance(raw_input, str):
            # Extract filename-like segments
            name = os.path.basename(raw_input).split('.')[0] if '.' in raw_input else raw_input
            if name and len(name) > 2:
                tuples.append(self._make_categorical(name, "audio_subject", 0.3))

        # Add a functional tuple for the modality
        tuples.append(self._make_functional("audio", "perceivable", 0.5))

        return tuples[:5]
