# @WHO:   self-ai/src/derivation/model_registry.py
# @WHAT:  Shared model singletons — one bge-m3, one Qwen3-0.6B for the entire process
# @PART:  self-ai/derivation
# @ENTRY: get_shared_embedding_model(), get_shared_qwen()

"""Model Registry — shared model singletons to prevent RAM explosion.

Problem:
    bge-m3 (2.2GB RAM) was loaded 3 separate times:
      - UnderstandingRetriever._ensure_model()
      - EmbeddingConceptDetector._ensure_model()
      - ExperienceStore._ensure_model()

    Qwen3-0.6B (1.5GB RAM) was loaded 2 separate times:
      - LLMReasoningEngine._try_local_qwen() — NO CACHING, reloaded every call!
      - UnderstandingComposer._try_local_qwen() — cached, but separate instance

    Total: 6.6–9.6GB RAM wasted on duplicate model instances.
    With 8GB RAM, this causes OOM when comprehend() triggers both models.

Solution:
    This module provides shared singletons for both models:
      - get_shared_embedding_model() → one SentenceTransformer('BAAI/bge-m3')
      - get_shared_qwen() → one (model, tokenizer) pair for Qwen3-0.6B

    All consumers call these functions instead of loading their own copies.
    RAM savings: ~4.4GB (bge-m3) + ~1.5GB (Qwen3) = ~5.9GB saved.

Disk space guard:
    Before loading any model, we check available disk space.
    If < 500MB free, we skip loading and log a warning — this prevents
    the auto-download of model.safetensors that previously filled the disk.

    We also set HF_HUB_OFFLINE=1 if the model is already cached,
    preventing HuggingFace from re-downloading files.
"""

import os
import logging
import shutil

logger = logging.getLogger(__name__)

# ═══════════════════════════════════════════════════════════
#  Disk space guard
# ═══════════════════════════════════════════════════════════

MIN_DISK_FREE_MB = 500  # Minimum free space before model loading


def _check_disk_space() -> bool:
    """Check if there's enough disk space for model operations.

    Returns True if at least MIN_DISK_FREE_MB is available.
    """
    try:
        usage = shutil.disk_usage('/')
        free_mb = usage.free / (1024 * 1024)
        if free_mb < MIN_DISK_FREE_MB:
            logger.warning(
                "Low disk space: %.0fMB free (minimum %dMB). "
                "Skipping model loading to prevent disk-full errors.",
                free_mb, MIN_DISK_FREE_MB
            )
            return False
        return True
    except Exception:
        return True  # If we can't check, allow loading


def _is_model_cached(model_name: str) -> bool:
    """Check if a HuggingFace model is already cached locally.

    This prevents unnecessary downloads and disk usage.
    """
    cache_dir = os.path.expanduser('~/.cache/huggingface/hub')
    # Normalize model name to directory format: "BAAI/bge-m3" → "models--BAAI--bge-m3"
    dir_name = f"models--{model_name.replace('/', '--')}"
    model_path = os.path.join(cache_dir, dir_name)
    return os.path.isdir(model_path)


# ═══════════════════════════════════════════════════════════
#  bge-m3 singleton
# ═══════════════════════════════════════════════════════════

_shared_embedding_model = None
_embedding_model_loading = False  # Prevent recursive loading


def get_shared_embedding_model():
    """Get the shared bge-m3 SentenceTransformer singleton.

    Loads the model only once, then returns the cached instance.
    All consumers (UnderstandingRetriever, EmbeddingConceptDetector,
    ExperienceStore) should use this instead of loading their own copy.

    Returns:
        SentenceTransformer instance, or None if loading fails.
    """
    global _shared_embedding_model, _embedding_model_loading

    if _shared_embedding_model is not None:
        return _shared_embedding_model

    if _embedding_model_loading:
        logger.warning("Recursive embedding model load detected — returning None")
        return None

    _embedding_model_loading = True

    try:
        # Disk space check before loading
        if not _check_disk_space():
            logger.warning("Insufficient disk space — bge-m3 loading skipped")
            return None

        # If model is already cached, set offline mode to prevent re-download
        if _is_model_cached('BAAI/bge-m3'):
            os.environ['HF_HUB_OFFLINE'] = '1'
            logger.info("bge-m3 is cached — setting HF_HUB_OFFLINE=1")

        from sentence_transformers import SentenceTransformer
        logger.info("Loading shared bge-m3 model (first time)...")
        _shared_embedding_model = SentenceTransformer('BAAI/bge-m3')
        logger.info("Shared bge-m3 loaded successfully (dim=1024)")

        # Clear offline flag after loading
        os.environ.pop('HF_HUB_OFFLINE', None)

        return _shared_embedding_model

    except ImportError:
        logger.warning("sentence_transformers not available — embedding model disabled")
        return None
    except Exception as e:
        logger.warning("Failed to load shared bge-m3 model: %s", e)
        os.environ.pop('HF_HUB_OFFLINE', None)
        return None
    finally:
        _embedding_model_loading = False


# ═══════════════════════════════════════════════════════════
#  Qwen3-0.6B singleton
# ═══════════════════════════════════════════════════════════

_shared_qwen_model = None
_shared_qwen_tokenizer = None
_qwen_loading = False  # Prevent recursive loading


def get_shared_qwen():
    """Get the shared Qwen3-0.6B (model, tokenizer) singleton.

    Loads the model only once, then returns the cached tuple.
    All consumers (LLMReasoningEngine, UnderstandingComposer)
    should use this instead of loading their own copy.

    Returns:
        Tuple of (model, tokenizer), or (None, None) if loading fails.
    """
    global _shared_qwen_model, _shared_qwen_tokenizer, _qwen_loading

    if _shared_qwen_model is not None and _shared_qwen_tokenizer is not None:
        return _shared_qwen_model, _shared_qwen_tokenizer

    if _qwen_loading:
        logger.warning("Recursive Qwen model load detected — returning None")
        return None, None

    _qwen_loading = True

    try:
        # Check if model is cached
        if not _is_model_cached('Qwen/Qwen3-0.6B'):
            logger.info("Qwen3-0.6B not cached locally — skipping shared loading")
            return None, None

        # Disk space check
        if not _check_disk_space():
            logger.warning("Insufficient disk space — Qwen3-0.6B loading skipped")
            return None, None

        # If model is cached, set offline mode
        os.environ['HF_HUB_OFFLINE'] = '1'

        from transformers import AutoModelForCausalLM, AutoTokenizer
        logger.info("Loading shared Qwen3-0.6B model (first time)...")
        _shared_qwen_tokenizer = AutoTokenizer.from_pretrained('Qwen/Qwen3-0.6B')
        _shared_qwen_model = AutoModelForCausalLM.from_pretrained(
            'Qwen/Qwen3-0.6B', torch_dtype='auto', device_map='auto'
        )
        logger.info("Shared Qwen3-0.6B loaded successfully")

        # Clear offline flag
        os.environ.pop('HF_HUB_OFFLINE', None)

        return _shared_qwen_model, _shared_qwen_tokenizer

    except ImportError:
        logger.warning("transformers not available — Qwen3 model disabled")
        return None, None
    except Exception as e:
        logger.warning("Failed to load shared Qwen3-0.6B model: %s", e)
        os.environ.pop('HF_HUB_OFFLINE', None)
        return None, None
    finally:
        _qwen_loading = False


def is_qwen_available() -> bool:
    """Quick check if Qwen3-0.6B is cached locally (without loading it)."""
    return _is_model_cached('Qwen/Qwen3-0.6B')


def is_embedding_model_available() -> bool:
    """Quick check if bge-m3 is cached locally (without loading it)."""
    return _is_model_cached('BAAI/bge-m3')
