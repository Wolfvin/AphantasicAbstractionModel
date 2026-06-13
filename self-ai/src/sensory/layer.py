# @WHO:   self-ai/src/sensory/layer.py
# @WHAT:  Layer 1 — Sensory: konversi input teks menjadi embedding (intuisi)
# @PART:  sensory
# @ENTRY: SensoryLayer.encode(), SensoryLayer.encode_batch()

import numpy as np
from typing import List

try:
    from sentence_transformers import SentenceTransformer
    _HAS_SBERT = True
except ImportError:
    _HAS_SBERT = False


class SensoryLayer:
    """
    Layer 1: Sensory — "Intuisi yang sudah matang"

    SELF tidak tahu cara ini bekerja.
    Seperti manusia tidak tahu cara mata memproses foton.
    Dia hanya tahu: "ini terasa berbeda dari itu."

    Ini bukan raw input — ini INTUISI yang sudah terkompres.
    Pengalaman manusia, dalam bentuk vektor.

    Default model: BAAI/bge-m3 (1024-dim, multilingual 100+ bahasa, encoder-only).
    Bge-m3 adalah encoder-only embedding model — purpose-built untuk embeddings,
    tanpa decoder yang harus di-strip. Semua 568M params dipakai untuk embeddings.
    Indonesian support jauh lebih baik dari Qwen3-Embedding-0.6B.
    Fallback: random projection untuk testing tanpa GPU.
    """

    # Model yang didukung dan dimensinya
    MODEL_DIMENSIONS = {
        "Qwen/Qwen3-Embedding-0.6B": 1024,
        "BAAI/bge-m3": 1024,
        "intfloat/multilingual-e5-large": 1024,
        "KaLM-Embedding/KaLM-embedding-multilingual-mini-instruct-v2.5": 1024,
        "text-embedding-3-large": 3072,  # OpenAI (legacy fallback)
    }

    def __init__(self, model_name: str = "BAAI/bge-m3"):
        # @FLOW:     SENSORY_INIT
        # @CALLS:    SentenceTransformer() → model encoder
        # @MUTATES:  none
        # @BEHAVIOR: Lazy-load model. Fallback ke random projection jika library tidak tersedia.
        #            BAAI/bge-m3: 1024-dim, multilingual (100+ bahasa termasuk Indonesian),
        #            encoder-only (568M params). Top MTEB multilingual. Purpose-built untuk embeddings.
        #            Indonesian support jauh lebih baik dari Qwen3-Embedding-0.6B.
        self.model_name = model_name
        self._model = None
        self._dimension = self.MODEL_DIMENSIONS.get(model_name, 1024)

    def _ensure_model(self):
        # @FLOW:     SENSORY_INIT
        # @CALLS:    SentenceTransformer() load
        # @MUTATES:  self._model, self._dimension
        if self._model is not None:
            return
        if _HAS_SBERT:
            self._model = SentenceTransformer(self.model_name)
            self._dimension = self._model.get_embedding_dimension()
        else:
            # Fallback: random projection untuk testing tanpa model berat
            # Dimensi tetap sesuai model yang dipilih
            self._model = None

    def encode(self, text: str) -> np.ndarray:
        """
        @FLOW:     SENSORY_ENCODE
        @CALLS:    SentenceTransformer.encode() → np.ndarray
        @MUTATES:  none
        @BEHAVIOR: Mengembalikan normalized embedding. Jika model tidak tersedia,
                   menggunakan random projection yang konsisten (seed dari hash teks).
                   Bge-m3 support dense, sparse, dan ColBERT — disini hanya dense.
                   Untuk instruction-based embedding, gunakan encode_with_instruction().
        """
        self._ensure_model()
        if self._model is not None:
            vec = self._model.encode(text, normalize_embeddings=True)
            return np.array(vec, dtype=np.float32)
        else:
            # Deterministic random projection berdasarkan hash teks
            rng = np.random.RandomState(hash(text) % (2**31))
            vec = rng.randn(self._dimension).astype(np.float32)
            norm = np.linalg.norm(vec)
            if norm > 0:
                vec = vec / norm
            return vec

    def encode_batch(self, texts: List[str]) -> np.ndarray:
        """
        @FLOW:     SENSORY_ENCODE_BATCH
        @CALLS:    SentenceTransformer.encode() → np.ndarray
        @MUTATES:  none
        @BEHAVIOR: Batch encoding untuk efisiensi. Normalized output.
        """
        self._ensure_model()
        if self._model is not None:
            vecs = self._model.encode(texts, normalize_embeddings=True)
            return np.array(vecs, dtype=np.float32)
        else:
            return np.stack([self.encode(t) for t in texts])

    def encode_with_instruction(self, text: str, instruction: str = "") -> np.ndarray:
        """
        @FLOW:     SENSORY_ENCODE_INSTRUCTED
        @CALLS:    SentenceTransformer.encode() dengan prefix
        @MUTATES:  none
        @BEHAVIOR: Instruction-based embedding untuk task-specific encoding.
                   Format prefix otomatis menyesuaikan model:
                   - BAAI/bge-m3: Tidak support instruction prefix — encode biasa.
                     (bge-m3 sudah multilingual dan context-aware tanpa instruction)
                   - Qwen3-Embedding: "Instruct: ...\\nQuery: ..."
                   - intfloat/e5: "Instruction: ...\\nSentence: ..."
                   Jika instruction kosong, fallback ke encode() biasa.
                   Ini penting untuk SELF karena embedding bisa di-optimalkan
                   berdasarkan konteks: teaching vs exploration vs query.
        """
        if not instruction:
            return self.encode(text)

        # Bge-m3: tidak support instruction prefix — encode biasa
        if "bge" in self.model_name.lower():
            return self.encode(text)

        # Qwen3-Embedding format
        if "qwen" in self.model_name.lower():
            prefixed = f"Instruct: {instruction}\nQuery: {text}"
            return self.encode(prefixed)

        # E5 format
        if "e5" in self.model_name.lower():
            prefixed = f"Instruction: {instruction}\nSentence: {text}"
            return self.encode(prefixed)

        # Generic fallback
        prefixed = f"{instruction}: {text}"
        return self.encode(prefixed)

    @property
    def dimension(self) -> int:
        self._ensure_model()
        return self._dimension
