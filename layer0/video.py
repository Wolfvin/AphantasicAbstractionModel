"""
AAM Layer 0 — Video Perceptual Abstractor

L0-03: Implemented using frame sampling + description + temporal tuple extraction.

Strategy:
  - If a frame bridge is available: sample key frames, extract descriptions
    for each frame, then create spatial/categorical tuples per frame and
    temporal tuples between frames.
  - If only an LLM bridge is available: treat input as a video description
    and extract tuples (including temporal ones) via TextAbstractor-style parsing.
  - Focus on temporal tuples ("object X appears before Y", "scene changes
    from A to B"), plus per-frame spatial and categorical tuples.
  - No video/frame data stored — only PerceptualTuple results.

If no bridge is available, falls back to metadata extraction from the
video reference.
"""

import hashlib
import datetime

from .base import (
    BasePerceptualAbstractor, PerceptualObservation, PerceptualTuple,
    PerceptualTupleMeta, ModalityType, RelationType,
)
from .text import TextAbstractor


# Prompt for video description extraction via LLM
VIDEO_DESCRIPTION_PROMPT = """You are a video content analyzer. Given a description of video content, extract structured relational tuples.

Output ONLY a JSON array. No explanation. No markdown fences.

Each object must have:
  subject       : string — the main entity or event
  relation_type : one of [categorical, differential, functional, spatial, temporal, causal]
  predicate     : string — what the subject relates to
  dimension     : string or null
  direction     : string or null
  confidence    : float 0.0-1.0

Focus on temporal and causal relations that are common in video:
- "car enters scene before pedestrian" (temporal)
- "rain causes flood" (causal)
- "person is in the kitchen" (spatial)
- "dog is a mammal" (categorical)

Video description:
{text}"""


class VideoAbstractor(BasePerceptualAbstractor):
    modality = ModalityType.VIDEO

    def __init__(self, llm_bridge=None, frame_bridge=None, audio_abstractor=None):
        """
        llm_bridge: optional — instance with method generate(prompt: str) -> str.
        frame_bridge: optional — instance with method sample_frames(video_input, n) -> list[str].
                      Returns descriptions for sampled frames.
        audio_abstractor: optional — AudioAbstractor for processing audio tracks.
        """
        self.frame_bridge = frame_bridge
        self.audio_abstractor = audio_abstractor
        self._text_abstractor = TextAbstractor(llm_bridge=llm_bridge)

    def abstract(self, raw_input, context: dict = {}) -> PerceptualObservation:
        """
        Extract PerceptualTuples from video input.

        Args:
            raw_input: Video data (bytes, file path, or video description string).
            context: Optional context dict with metadata.

        Returns:
            PerceptualObservation with extracted tuples.
        """
        # Compute reference hash
        if isinstance(raw_input, bytes):
            input_ref = "video:" + hashlib.sha256(raw_input).hexdigest()[:16]
        elif isinstance(raw_input, str):
            input_ref = "video:" + hashlib.sha256(raw_input.encode()).hexdigest()[:16]
        else:
            input_ref = "video:unknown"

        tuples: list[PerceptualTuple] = []

        # Strategy 1: Frame bridge → per-frame descriptions → tuples + temporal links
        if self.frame_bridge and isinstance(raw_input, (bytes, str)):
            frame_descriptions = self._sample_frames(raw_input)
            if frame_descriptions:
                tuples = self._extract_from_frames(frame_descriptions)

        # Strategy 2: LLM-driven extraction from video description
        if not tuples and self._text_abstractor.llm_bridge and isinstance(raw_input, str):
            text_obs = self._text_abstractor.abstract(raw_input, context)
            for t in text_obs.tuples:
                t.source_modality = ModalityType.VIDEO
                tuples.append(t)

        # Strategy 3: Process audio track if audio abstractor available
        if self.audio_abstractor and isinstance(raw_input, (bytes, str)):
            try:
                audio_obs = self.audio_abstractor.abstract(raw_input, context)
                # Add audio-derived tuples, remapped to VIDEO modality
                for t in audio_obs.tuples:
                    t.source_modality = ModalityType.VIDEO
                    tuples.append(t)
            except Exception:
                pass

        # Strategy 4: Fallback — metadata extraction
        if not tuples:
            tuples = self._extract_metadata_fallback(raw_input, input_ref)

        # Set metadata
        meta = PerceptualTupleMeta(
            source_url=input_ref,
            extraction_model="frames+llm" if self.frame_bridge else "fallback",
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

    def _sample_frames(self, video_input, n: int = 3) -> list[str]:
        """Sample key frames from video and return their descriptions."""
        try:
            return self.frame_bridge.sample_frames(video_input, n)
        except Exception:
            return []

    def _extract_from_frames(self, frame_descriptions: list[str]) -> list[PerceptualTuple]:
        """
        Extract tuples from sampled frame descriptions.

        For each frame: extract spatial + categorical tuples.
        Between frames: create temporal tuples linking frame events.
        """
        tuples: list[PerceptualTuple] = []
        frame_entities: list[str] = []

        for i, desc in enumerate(frame_descriptions):
            # Extract tuples from each frame's description
            text_obs = self._text_abstractor.abstract(desc)
            for t in text_obs.tuples:
                t.source_modality = ModalityType.VIDEO
                tuples.append(t)
                # Track main entities for temporal linking
                if t.subject and t.subject not in frame_entities:
                    frame_entities.append(t.subject)

            # Create temporal tuples between consecutive frames
            if i > 0:
                # "frame_N occurs after frame_N-1"
                tuples.append(self._make_temporal(
                    f"frame_{i}", f"after frame_{i - 1}", 0.7
                ))

                # If we have entities from consecutive frames, link them
                if frame_entities:
                    # Link the most recent entity to a temporal sequence
                    for entity in frame_entities[-2:]:
                        tuples.append(self._make_temporal(
                            entity, f"appears in frame_{i}", 0.5
                        ))

        return tuples

    def _extract_metadata_fallback(self, raw_input, input_ref: str) -> list[PerceptualTuple]:
        """
        Fallback: extract basic tuples from video metadata/filename cues.
        """
        tuples: list[PerceptualTuple] = []

        # At minimum, create a categorical tuple for the video itself
        tuples.append(self._make_categorical(input_ref, "video", 0.3))

        # Try to extract cues from the reference string
        if isinstance(raw_input, str):
            import os
            name = os.path.basename(raw_input).split('.')[0] if '.' in raw_input else raw_input
            if name and len(name) > 2:
                tuples.append(self._make_categorical(name, "video_subject", 0.3))

        # Add functional tuples
        tuples.append(self._make_functional("video", "perceivable", 0.5))
        tuples.append(self._make_temporal("video", "has duration", 0.4))

        return tuples[:5]
