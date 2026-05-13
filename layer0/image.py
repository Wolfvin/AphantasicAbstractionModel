"""
AAM Layer 0 — Image Perceptual Abstractor

L0-03: Implemented using structured description extraction with LLM bridge.

Strategy:
  - If an LLM bridge is available: use it to generate a structured description
    of the image content, then extract tuples via TextAbstractor-style parsing.
  - If a vision bridge is available: use it to get image description, then
    extract tuples.
  - Focus on categorical, spatial, and differential relations:
    "cat is an animal", "cat is on the table", "cat is larger than mouse".
  - No pixel data stored — only PerceptualTuple results.

If no bridge is available, falls back to metadata extraction from the
image reference (filename/path cues).
"""

import hashlib
import datetime
import os

from .base import (
    BasePerceptualAbstractor, PerceptualObservation, PerceptualTuple,
    PerceptualTupleMeta, ModalityType, RelationType,
)
from .text import TextAbstractor


# Prompt for image description extraction via LLM
IMAGE_DESCRIPTION_PROMPT = """You are a visual content analyzer. Given a description of an image, extract structured relational tuples.

Output ONLY a JSON array. No explanation. No markdown fences.

Each object must have:
  subject       : string — the main entity or object
  relation_type : one of [categorical, differential, functional, spatial, temporal, causal]
  predicate     : string — what the subject relates to
  dimension     : string or null — (differential only) what dimension is compared
  direction     : string or null — (differential only) comparison direction
  confidence    : float 0.0-1.0

Focus on spatial and categorical relations that are common in images:
- "cat is on the table" (spatial)
- "cat is a mammal" (categorical)
- "cat is larger than mouse" (differential, dimension=size, direction=larger)
- "car is red" (categorical)

Image description:
{text}"""


class ImageAbstractor(BasePerceptualAbstractor):
    modality = ModalityType.IMAGE

    def __init__(self, llm_bridge=None, vision_bridge=None):
        """
        llm_bridge: optional — instance with method generate(prompt: str) -> str.
                    Used for structured extraction from image descriptions.
        vision_bridge: optional — instance with method describe(image_input) -> str.
                       Used to generate image descriptions. Can be CLIP/LLaVA.
        """
        self.vision_bridge = vision_bridge
        self._text_abstractor = TextAbstractor(llm_bridge=llm_bridge)

    def abstract(self, raw_input, context: dict = {}) -> PerceptualObservation:
        """
        Extract PerceptualTuples from image input.

        Args:
            raw_input: Image data (bytes, file path, or image description string).
            context: Optional context dict with metadata.

        Returns:
            PerceptualObservation with extracted tuples.
        """
        # Compute reference hash
        if isinstance(raw_input, bytes):
            input_ref = "image:" + hashlib.sha256(raw_input).hexdigest()[:16]
        elif isinstance(raw_input, str):
            input_ref = "image:" + hashlib.sha256(raw_input.encode()).hexdigest()[:16]
        else:
            input_ref = "image:unknown"

        tuples: list[PerceptualTuple] = []

        # Strategy 1: Vision bridge → description → TextAbstractor
        if self.vision_bridge and isinstance(raw_input, (bytes, str)):
            description = self._describe(raw_input)
            if description:
                text_obs = self._text_abstractor.abstract(description, context)
                for t in text_obs.tuples:
                    t.source_modality = ModalityType.IMAGE
                    tuples.append(t)

        # Strategy 2: LLM-driven extraction from description string
        elif self._text_abstractor.llm_bridge and isinstance(raw_input, str):
            # Treat string input as image description
            text_obs = self._text_abstractor.abstract(raw_input, context)
            for t in text_obs.tuples:
                t.source_modality = ModalityType.IMAGE
                tuples.append(t)

        # Strategy 3: Fallback — metadata extraction
        if not tuples:
            tuples = self._extract_metadata_fallback(raw_input, input_ref)

        # Set metadata
        meta = PerceptualTupleMeta(
            source_url=input_ref,
            extraction_model="vision+llm" if self.vision_bridge else "fallback",
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

    def _describe(self, image_input) -> str | None:
        """Generate a description of the image using the vision bridge."""
        try:
            return self.vision_bridge.describe(image_input)
        except Exception:
            return None

    def _extract_metadata_fallback(self, raw_input, input_ref: str) -> list[PerceptualTuple]:
        """
        Fallback: extract basic tuples from image metadata/filename cues.
        """
        tuples: list[PerceptualTuple] = []

        # At minimum, create a categorical tuple for the image itself
        tuples.append(self._make_categorical(input_ref, "image", 0.3))

        # Try to extract cues from the reference string
        if isinstance(raw_input, str):
            name = os.path.basename(raw_input).split('.')[0] if '.' in raw_input else raw_input
            if name and len(name) > 2:
                tuples.append(self._make_categorical(name, "visual_subject", 0.3))

        # Add functional tuples
        tuples.append(self._make_functional("image", "perceivable", 0.5))

        return tuples[:5]
