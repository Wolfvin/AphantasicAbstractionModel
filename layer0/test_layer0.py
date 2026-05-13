"""
AAM Layer 0 — Comprehensive Test Suite

Tests for:
  - PerceptualTuple creation and metadata (L0-06)
  - PerceptualObservation construction
  - TextAbstractor._dict_to_tuple()
  - TextAbstractor._extract_fallback() (L0-04 improved fallback)
  - TextAbstractor retry + caching (L0-04)
  - Base class method contracts
  - Adapter: observation_to_ingest_data (L0-01)
  - Adapter: ingest_observation (L0-01)
  - AudioAbstractor, ImageAbstractor, VideoAbstractor (L0-03)
"""

import sys
import os
import time
import unittest

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from layer0.base import (
    BasePerceptualAbstractor,
    PerceptualObservation,
    PerceptualTuple,
    PerceptualTupleMeta,
    ModalityType,
    RelationType,
)
from layer0.text import TextAbstractor
from layer0.image import ImageAbstractor
from layer0.video import VideoAbstractor
from layer0.audio import AudioAbstractor
from layer0.adapter import (
    observation_to_ingest_data,
    observation_to_ingest_dicts,
    ingest_observation,
    RsvsIngestProtocol,
)


# ---------------------------------------------------------------------------
# PerceptualTuple & PerceptualTupleMeta tests (L0-06)
# ---------------------------------------------------------------------------

class TestPerceptualTupleMeta(unittest.TestCase):
    """Tests for PerceptualTupleMeta (L0-06)."""

    def test_default_creation(self):
        meta = PerceptualTupleMeta()
        self.assertEqual(meta.source_url, "")
        self.assertEqual(meta.extraction_model, "")
        self.assertEqual(meta.extraction_timestamp, 0.0)
        self.assertEqual(meta.extra, {})

    def test_creation_with_values(self):
        meta = PerceptualTupleMeta(
            source_url="text:abc123",
            extraction_model="llm",
            extraction_timestamp=1234567890.0,
        )
        self.assertEqual(meta.source_url, "text:abc123")
        self.assertEqual(meta.extraction_model, "llm")
        self.assertEqual(meta.extraction_timestamp, 1234567890.0)

    def test_extra_fields(self):
        meta = PerceptualTupleMeta(
            source_url="test",
            extra={"custom_field": "value", "count": 42},
        )
        self.assertEqual(meta.extra["custom_field"], "value")
        self.assertEqual(meta.extra["count"], 42)

    def test_to_dict(self):
        meta = PerceptualTupleMeta(
            source_url="text:abc",
            extraction_model="fallback",
            extraction_timestamp=100.0,
            extra={"key": "val"},
        )
        d = meta.to_dict()
        self.assertEqual(d["source_url"], "text:abc")
        self.assertEqual(d["extraction_model"], "fallback")
        self.assertEqual(d["extraction_timestamp"], 100.0)
        self.assertEqual(d["key"], "val")

    def test_from_dict(self):
        d = {
            "source_url": "img:xyz",
            "extraction_model": "vision",
            "extraction_timestamp": 200.0,
            "custom": "data",
        }
        meta = PerceptualTupleMeta.from_dict(d)
        self.assertEqual(meta.source_url, "img:xyz")
        self.assertEqual(meta.extraction_model, "vision")
        self.assertEqual(meta.extraction_timestamp, 200.0)
        self.assertEqual(meta.extra["custom"], "data")

    def test_from_dict_minimal(self):
        """Backward compat: dict with missing keys."""
        d = {"source_url": "test"}
        meta = PerceptualTupleMeta.from_dict(d)
        self.assertEqual(meta.source_url, "test")
        self.assertEqual(meta.extraction_model, "")
        self.assertEqual(meta.extraction_timestamp, 0.0)

    def test_from_dict_idempotent(self):
        meta = PerceptualTupleMeta(source_url="test", extraction_model="llm")
        result = PerceptualTupleMeta.from_dict(meta)
        self.assertIs(result, meta)  # Should return same object


class TestPerceptualTuple(unittest.TestCase):
    """Tests for PerceptualTuple creation and metadata (L0-06)."""

    def test_basic_creation(self):
        t = PerceptualTuple(
            subject="apple",
            relation_type=RelationType.CATEGORICAL,
            predicate="fruit",
        )
        self.assertEqual(t.subject, "apple")
        self.assertEqual(t.relation_type, RelationType.CATEGORICAL)
        self.assertEqual(t.predicate, "fruit")
        self.assertIsNone(t.dimension)
        self.assertIsNone(t.direction)
        self.assertEqual(t.confidence, 1.0)
        self.assertEqual(t.source_modality, ModalityType.TEXT)

    def test_full_creation(self):
        t = PerceptualTuple(
            subject="apple",
            relation_type=RelationType.DIFFERENTIAL,
            predicate="pear",
            dimension="shape",
            direction="rounder",
            confidence=0.85,
            source_modality=ModalityType.IMAGE,
        )
        self.assertEqual(t.dimension, "shape")
        self.assertEqual(t.direction, "rounder")
        self.assertEqual(t.confidence, 0.85)
        self.assertEqual(t.source_modality, ModalityType.IMAGE)

    def test_metadata_default_is_PerceptualTupleMeta(self):
        """L0-06: Default metadata should be PerceptualTupleMeta, not dict."""
        t = PerceptualTuple(
            subject="test",
            relation_type=RelationType.CATEGORICAL,
            predicate="value",
        )
        self.assertIsInstance(t.metadata, PerceptualTupleMeta)

    def test_metadata_dict_backward_compat(self):
        """L0-06: Dict metadata should auto-convert to PerceptualTupleMeta."""
        t = PerceptualTuple(
            subject="test",
            relation_type=RelationType.CATEGORICAL,
            predicate="value",
            metadata={"source_url": "test", "extraction_model": "llm", "extraction_timestamp": 1.0},
        )
        self.assertIsInstance(t.metadata, PerceptualTupleMeta)
        self.assertEqual(t.metadata.source_url, "test")

    def test_get_metadata_dict(self):
        t = PerceptualTuple(
            subject="test",
            relation_type=RelationType.CATEGORICAL,
            predicate="value",
            metadata=PerceptualTupleMeta(source_url="url", extraction_model="model"),
        )
        d = t.get_metadata_dict()
        self.assertIsInstance(d, dict)
        self.assertEqual(d["source_url"], "url")


class TestPerceptualObservation(unittest.TestCase):
    """Tests for PerceptualObservation construction."""

    def test_basic_creation(self):
        tuples = [
            PerceptualTuple(subject="a", relation_type=RelationType.CATEGORICAL, predicate="b"),
        ]
        obs = PerceptualObservation(
            modality=ModalityType.TEXT,
            raw_input_ref="text:abc123",
            tuples=tuples,
        )
        self.assertEqual(obs.modality, ModalityType.TEXT)
        self.assertEqual(obs.raw_input_ref, "text:abc123")
        self.assertEqual(len(obs.tuples), 1)
        self.assertEqual(obs.context, {})
        self.assertEqual(obs.timestamp, "")

    def test_creation_with_context(self):
        obs = PerceptualObservation(
            modality=ModalityType.AUDIO,
            raw_input_ref="audio:xyz",
            tuples=[],
            context={"language": "en"},
            timestamp="2026-01-01T00:00:00",
        )
        self.assertEqual(obs.context["language"], "en")
        self.assertEqual(obs.timestamp, "2026-01-01T00:00:00")


# ---------------------------------------------------------------------------
# Base class contract tests
# ---------------------------------------------------------------------------

class TestBasePerceptualAbstractor(unittest.TestCase):
    """Tests for BasePerceptualAbstractor contracts."""

    def test_abstract_raises_not_implemented(self):
        base = BasePerceptualAbstractor()
        with self.assertRaises(NotImplementedError):
            base.abstract("test")

    def test_modality_not_implemented(self):
        base = BasePerceptualAbstractor()
        self.assertEqual(base.modality, NotImplemented)

    def test_make_categorical(self):
        base = BasePerceptualAbstractor()
        base.modality = ModalityType.TEXT
        t = base._make_categorical("apple", "fruit", 0.9)
        self.assertEqual(t.subject, "apple")
        self.assertEqual(t.relation_type, RelationType.CATEGORICAL)
        self.assertEqual(t.predicate, "fruit")
        self.assertEqual(t.confidence, 0.9)

    def test_make_differential(self):
        base = BasePerceptualAbstractor()
        base.modality = ModalityType.IMAGE
        t = base._make_differential("cat", "mouse", "size", "larger", 0.8)
        self.assertEqual(t.relation_type, RelationType.DIFFERENTIAL)
        self.assertEqual(t.dimension, "size")
        self.assertEqual(t.direction, "larger")

    def test_make_functional(self):
        base = BasePerceptualAbstractor()
        base.modality = ModalityType.TEXT
        t = base._make_functional("knife", "cut", 0.95)
        self.assertEqual(t.relation_type, RelationType.FUNCTIONAL)
        self.assertEqual(t.predicate, "cut")

    def test_make_spatial(self):
        base = BasePerceptualAbstractor()
        base.modality = ModalityType.IMAGE
        t = base._make_spatial("cat", "on the table", 0.7)
        self.assertEqual(t.relation_type, RelationType.SPATIAL)

    def test_make_temporal(self):
        base = BasePerceptualAbstractor()
        base.modality = ModalityType.VIDEO
        t = base._make_temporal("rain", "before flood", 0.6)
        self.assertEqual(t.relation_type, RelationType.TEMPORAL)

    def test_make_causal(self):
        base = BasePerceptualAbstractor()
        base.modality = ModalityType.TEXT
        t = base._make_causal("rain", "flood", 0.8)
        self.assertEqual(t.relation_type, RelationType.CAUSAL)


# ---------------------------------------------------------------------------
# TextAbstractor tests (L0-04)
# ---------------------------------------------------------------------------

class TestTextAbstractor(unittest.TestCase):
    """Tests for TextAbstractor including L0-04 improvements."""

    def test_abstract_fallback_no_llm(self):
        """Without LLM, should use fallback extraction."""
        ta = TextAbstractor()
        obs = ta.abstract("The quick brown fox jumps over the lazy dog")
        self.assertIsInstance(obs, PerceptualObservation)
        self.assertEqual(obs.modality, ModalityType.TEXT)
        self.assertTrue(len(obs.tuples) > 0)

    def test_abstract_returns_observation(self):
        ta = TextAbstractor()
        obs = ta.abstract("An apple is a fruit.")
        self.assertIsInstance(obs, PerceptualObservation)
        self.assertTrue(obs.raw_input_ref.startswith("text:"))
        self.assertNotEqual(obs.timestamp, "")

    def test_dict_to_tuple(self):
        ta = TextAbstractor()
        d = {
            "subject": "apple",
            "relation_type": "categorical",
            "predicate": "fruit",
            "confidence": 0.99,
            "dimension": None,
            "direction": None,
        }
        t = ta._dict_to_tuple(d)
        self.assertEqual(t.subject, "apple")
        self.assertEqual(t.relation_type, RelationType.CATEGORICAL)
        self.assertEqual(t.predicate, "fruit")
        self.assertEqual(t.confidence, 0.99)

    def test_dict_to_tuple_defaults(self):
        ta = TextAbstractor()
        d = {"subject": "x", "predicate": "y"}
        t = ta._dict_to_tuple(d)
        self.assertEqual(t.relation_type, RelationType.CATEGORICAL)
        self.assertEqual(t.confidence, 1.0)
        self.assertIsNone(t.dimension)

    def test_extract_fallback_produces_tuples(self):
        """L0-04: Improved fallback should produce meaningful tuples."""
        ta = TextAbstractor()
        tuples = ta._extract_fallback("An apple is a fruit. The cat can jump.")
        self.assertTrue(len(tuples) > 0)
        # Should have at least categorical tuples
        cat_tuples = [t for t in tuples if t.relation_type == RelationType.CATEGORICAL]
        self.assertTrue(len(cat_tuples) > 0)

    def test_extract_fallback_is_pattern(self):
        """L0-04: Fallback should detect 'X is Y' patterns."""
        ta = TextAbstractor()
        tuples = ta._extract_fallback("The apple is a fruit")
        # Should find categorical from 'is' pattern or noun-phrase extraction
        # The fallback extracts nouns via regex and 'is' patterns
        # At minimum, should produce some tuples
        self.assertTrue(len(tuples) > 0)
        # Check that "apple" appears as a subject in some tuple
        self.assertTrue(any("apple" in t.subject.lower() for t in tuples))

    def test_extract_fallback_can_pattern(self):
        """L0-04: Fallback should detect 'X can Y' patterns."""
        ta = TextAbstractor()
        tuples = ta._extract_fallback("The cat can jump")
        func_tuples = [t for t in tuples if t.relation_type == RelationType.FUNCTIONAL]
        self.assertTrue(len(func_tuples) > 0)

    def test_caching(self):
        """L0-04: Same text should be cached after first extraction."""
        ta = TextAbstractor()
        obs1 = ta.abstract("An apple is a fruit")
        obs2 = ta.abstract("An apple is a fruit")
        # Both should succeed — the second should come from cache
        self.assertTrue(len(obs1.tuples) > 0)
        self.assertTrue(len(obs2.tuples) > 0)
        # Cache should have the key
        self.assertTrue(len(ta._cache) > 0)

    def test_retry_with_failing_llm(self):
        """L0-04: LLM failures should trigger fallback, not return []."""
        class FailingBridge:
            def generate(self, prompt):
                raise RuntimeError("LLM unavailable")

        ta = TextAbstractor(llm_bridge=FailingBridge())
        obs = ta.abstract("An apple is a fruit and can be eaten")
        # Should NOT return empty — should use fallback
        self.assertTrue(len(obs.tuples) > 0)

    def test_retry_with_bad_json_llm(self):
        """L0-04: Bad JSON from LLM should trigger fallback."""
        class BadJsonBridge:
            def generate(self, prompt):
                return "not valid json at all"

        ta = TextAbstractor(llm_bridge=BadJsonBridge())
        obs = ta.abstract("An apple is a fruit")
        self.assertTrue(len(obs.tuples) > 0)


# ---------------------------------------------------------------------------
# Modality Abstractor tests (L0-03)
# ---------------------------------------------------------------------------

class TestAudioAbstractor(unittest.TestCase):
    """Tests for AudioAbstractor (L0-03)."""

    def test_fallback_no_bridges(self):
        aa = AudioAbstractor()
        obs = aa.abstract("test_audio.mp3")
        self.assertIsInstance(obs, PerceptualObservation)
        self.assertEqual(obs.modality, ModalityType.AUDIO)
        self.assertTrue(len(obs.tuples) > 0)

    def test_stt_bridge_pipeline(self):
        """AudioAbstractor with STT bridge should pipe through TextAbstractor."""
        class MockSTT:
            def transcribe(self, audio):
                return "The weather is sunny today"

        ta = TextAbstractor()  # Will use fallback
        aa = AudioAbstractor(stt_bridge=MockSTT())
        obs = aa.abstract("audio_data_placeholder")
        self.assertTrue(len(obs.tuples) > 0)
        # All tuples should have AUDIO modality
        for t in obs.tuples:
            self.assertEqual(t.source_modality, ModalityType.AUDIO)

    def test_stt_bridge_failure_falls_back(self):
        """Failing STT bridge should not crash — should use metadata fallback."""
        class FailingSTT:
            def transcribe(self, audio):
                raise RuntimeError("STT unavailable")

        aa = AudioAbstractor(stt_bridge=FailingSTT())
        obs = aa.abstract("test_audio.wav")
        self.assertTrue(len(obs.tuples) > 0)


class TestImageAbstractor(unittest.TestCase):
    """Tests for ImageAbstractor (L0-03)."""

    def test_fallback_no_bridges(self):
        ia = ImageAbstractor()
        obs = ia.abstract("photo_of_cat.jpg")
        self.assertIsInstance(obs, PerceptualObservation)
        self.assertEqual(obs.modality, ModalityType.IMAGE)
        self.assertTrue(len(obs.tuples) > 0)

    def test_vision_bridge_pipeline(self):
        class MockVision:
            def describe(self, image):
                return "A cat is on the table. The cat is a mammal."

        ia = ImageAbstractor(vision_bridge=MockVision())
        obs = ia.abstract("cat_photo.jpg")
        self.assertTrue(len(obs.tuples) > 0)
        for t in obs.tuples:
            self.assertEqual(t.source_modality, ModalityType.IMAGE)

    def test_vision_bridge_failure_falls_back(self):
        class FailingVision:
            def describe(self, image):
                raise RuntimeError("Vision unavailable")

        ia = ImageAbstractor(vision_bridge=FailingVision())
        obs = ia.abstract("test.jpg")
        self.assertTrue(len(obs.tuples) > 0)

    def test_llm_bridge_with_description(self):
        """LLM bridge without vision should still work with string descriptions."""
        ia = ImageAbstractor()  # No bridges
        obs = ia.abstract("A dog sits on a rug in the living room")
        self.assertTrue(len(obs.tuples) > 0)


class TestVideoAbstractor(unittest.TestCase):
    """Tests for VideoAbstractor (L0-03)."""

    def test_fallback_no_bridges(self):
        va = VideoAbstractor()
        obs = va.abstract("movie_clip.mp4")
        self.assertIsInstance(obs, PerceptualObservation)
        self.assertEqual(obs.modality, ModalityType.VIDEO)
        self.assertTrue(len(obs.tuples) > 0)

    def test_frame_bridge_pipeline(self):
        class MockFrameBridge:
            def sample_frames(self, video, n=3):
                return [
                    "A car drives down the road",
                    "The car stops at a red light",
                    "Pedestrians cross the street",
                ]

        va = VideoAbstractor(frame_bridge=MockFrameBridge())
        obs = va.abstract("traffic_cam.mp4")
        self.assertTrue(len(obs.tuples) > 0)
        # Should have temporal tuples linking frames
        temporal_tuples = [t for t in obs.tuples if t.relation_type == RelationType.TEMPORAL]
        self.assertTrue(len(temporal_tuples) > 0)

    def test_frame_bridge_failure_falls_back(self):
        class FailingFrameBridge:
            def sample_frames(self, video, n=3):
                raise RuntimeError("Frame extraction failed")

        va = VideoAbstractor(frame_bridge=FailingFrameBridge())
        obs = va.abstract("broken.mp4")
        self.assertTrue(len(obs.tuples) > 0)

    def test_with_audio_abstractor(self):
        """VideoAbstractor can use AudioAbstractor for audio tracks."""
        class MockSTT:
            def transcribe(self, audio):
                return "Someone is talking about weather"

        va = VideoAbstractor(audio_abstractor=AudioAbstractor(stt_bridge=MockSTT()))
        obs = va.abstract("interview.mp4")
        self.assertTrue(len(obs.tuples) > 0)


# ---------------------------------------------------------------------------
# Adapter tests (L0-01)
# ---------------------------------------------------------------------------

class TestAdapter(unittest.TestCase):
    """Tests for Layer 0 → Layer 1 adapter (L0-01)."""

    def _make_observation(self) -> PerceptualObservation:
        """Create a sample observation for testing."""
        tuples = [
            PerceptualTuple(subject="apple", relation_type=RelationType.CATEGORICAL,
                            predicate="fruit", confidence=0.99),
            PerceptualTuple(subject="apple", relation_type=RelationType.DIFFERENTIAL,
                            predicate="pear", dimension="shape", direction="rounder",
                            confidence=0.90),
            PerceptualTuple(subject="apple", relation_type=RelationType.FUNCTIONAL,
                            predicate="edible", confidence=0.95),
        ]
        return PerceptualObservation(
            modality=ModalityType.TEXT,
            raw_input_ref="text:abc123",
            tuples=tuples,
            timestamp="2026-01-01T00:00:00",
        )

    def test_observation_to_ingest_data(self):
        obs = self._make_observation()
        text = observation_to_ingest_data(obs)
        self.assertIsInstance(text, str)
        # Should contain natural language sentences
        self.assertIn("apple", text)
        self.assertIn("fruit", text)

    def test_observation_to_ingest_data_categorical(self):
        obs = PerceptualObservation(
            modality=ModalityType.TEXT,
            raw_input_ref="test",
            tuples=[PerceptualTuple(subject="cat", relation_type=RelationType.CATEGORICAL,
                                    predicate="mammal")],
        )
        text = observation_to_ingest_data(obs)
        self.assertIn("cat is a mammal", text)

    def test_observation_to_ingest_data_differential(self):
        obs = PerceptualObservation(
            modality=ModalityType.IMAGE,
            raw_input_ref="test",
            tuples=[PerceptualTuple(subject="cat", relation_type=RelationType.DIFFERENTIAL,
                                    predicate="mouse", dimension="size", direction="larger")],
        )
        text = observation_to_ingest_data(obs)
        self.assertIn("cat is larger than mouse in size", text)

    def test_observation_to_ingest_data_functional(self):
        obs = PerceptualObservation(
            modality=ModalityType.TEXT,
            raw_input_ref="test",
            tuples=[PerceptualTuple(subject="knife", relation_type=RelationType.FUNCTIONAL,
                                    predicate="cut")],
        )
        text = observation_to_ingest_data(obs)
        self.assertIn("knife can cut", text)

    def test_observation_to_ingest_data_spatial(self):
        obs = PerceptualObservation(
            modality=ModalityType.IMAGE,
            raw_input_ref="test",
            tuples=[PerceptualTuple(subject="book", relation_type=RelationType.SPATIAL,
                                    predicate="on the table")],
        )
        text = observation_to_ingest_data(obs)
        self.assertIn("book is located on the table", text)

    def test_observation_to_ingest_data_temporal(self):
        obs = PerceptualObservation(
            modality=ModalityType.VIDEO,
            raw_input_ref="test",
            tuples=[PerceptualTuple(subject="rain", relation_type=RelationType.TEMPORAL,
                                    predicate="before the flood")],
        )
        text = observation_to_ingest_data(obs)
        self.assertIn("rain occurs before the flood", text)

    def test_observation_to_ingest_data_causal(self):
        obs = PerceptualObservation(
            modality=ModalityType.TEXT,
            raw_input_ref="test",
            tuples=[PerceptualTuple(subject="rain", relation_type=RelationType.CAUSAL,
                                    predicate="flood")],
        )
        text = observation_to_ingest_data(obs)
        self.assertIn("rain causes flood", text)

    def test_observation_to_ingest_dicts(self):
        obs = self._make_observation()
        dicts = observation_to_ingest_dicts(obs)
        self.assertIsInstance(dicts, list)
        self.assertEqual(len(dicts), 3)
        self.assertEqual(dicts[0]["subject"], "apple")
        self.assertEqual(dicts[0]["relation_type"], "categorical")
        self.assertEqual(dicts[1]["dimension"], "shape")

    def test_ingest_observation_with_mock(self):
        """Test ingest_observation with a mock RSVS-like object."""
        class MockRsvs:
            def __init__(self):
                self.ingested = []
            def ingest(self, text):
                self.ingested.append(text)
                return {"sentences_processed": 1}

        mock = MockRsvs()
        obs = self._make_observation()
        result = ingest_observation(mock, obs)
        self.assertEqual(len(mock.ingested), 1)
        self.assertIn("apple", mock.ingested[0])

    def test_rsvs_protocol_check(self):
        """RsvsIngestProtocol should be satisfied by objects with ingest()."""
        class ValidRsvs:
            def ingest(self, text):
                pass

        class InvalidRsvs:
            def query(self, text):
                pass

        self.assertIsInstance(ValidRsvs(), RsvsIngestProtocol)
        # InvalidRsvs does NOT satisfy the protocol (no ingest method)


# ---------------------------------------------------------------------------
# Integration test: full pipeline
# ---------------------------------------------------------------------------

class TestIntegration(unittest.TestCase):
    """End-to-end integration tests for Layer 0."""

    def test_text_to_rsvs_pipeline(self):
        """Full pipeline: text → TextAbstractor → adapter → mock RSVS."""
        class MockRsvs:
            def __init__(self):
                self.ingested_texts = []
            def ingest(self, text):
                self.ingested_texts.append(text)
                return {"ok": True}

        ta = TextAbstractor()
        mock_rsvs = MockRsvs()

        obs = ta.abstract("An apple is a fruit. It can be eaten.")
        result = ingest_observation(mock_rsvs, obs)

        self.assertTrue(len(mock_rsvs.ingested_texts) > 0)
        ingested = mock_rsvs.ingested_texts[0]
        self.assertIn("apple", ingested)

    def test_audio_to_rsvs_pipeline(self):
        """Full pipeline: audio → AudioAbstractor → adapter → mock RSVS."""
        class MockSTT:
            def transcribe(self, audio):
                return "The weather is rainy and cold"
        class MockRsvs:
            def ingest(self, text):
                return {"ok": True}

        aa = AudioAbstractor(stt_bridge=MockSTT())
        obs = aa.abstract("weather_report.wav")
        text = observation_to_ingest_data(obs)
        self.assertIn("weather", text)

    def test_image_to_rsvs_pipeline(self):
        """Full pipeline: image → ImageAbstractor → adapter → mock RSVS."""
        class MockRsvs:
            def ingest(self, text):
                return {"ok": True}

        ia = ImageAbstractor()
        obs = ia.abstract("A cat sits on a mat in the living room")
        text = observation_to_ingest_data(obs)
        # Fallback should still produce output
        self.assertTrue(len(text) > 0)


if __name__ == "__main__":
    unittest.main()
