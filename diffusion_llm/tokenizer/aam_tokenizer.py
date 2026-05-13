"""
AAM Diffusion LLM — Tokenizer

Sentence-level + subword BPE hybrid tokenizer designed specifically
for AAM's sentence arrangement task.

Unlike standard tokenizers (GPT-2 BPE, SentencePiece) that tokenize
at the subword level, AAM's tokenizer is designed with SENTENCE
ARRANGEMENT in mind:

1. Sentences are the primary unit of generation (not individual tokens)
2. Within sentences, subword BPE handles individual words
3. Special tokens for graph structure (evidence, anomaly, confidence)
4. Sentence boundary markers for the diffusion model

The tokenizer maintains two levels:
- Sentence level: Where sentences begin/end, for the diffusion model
  to arrange and revise non-sequentially
- Token level: Subword units within sentences, for detailed generation

Analogi: Jin Soun tidak berpikir dalam kata-per-kata — dia
berpikir dalam KALIMAT. "Pencuri = Diancang pair. Ju Jangmok = cover."
Setiap kalimat sudah utuh, yang dia susun adalah URUTAN kalimat.
"""

from __future__ import annotations

import json
import re
import unicodedata
from collections import Counter
from pathlib import Path
from typing import Optional

from diffusion_llm.config.model_config import TokenizerConfig


# Special token IDs (always at the start of vocabulary)
SPECIAL_TOKENS = [
    "<pad>",        # 0
    "<bos>",        # 1
    "<eos>",        # 2
    "<mask>",       # 3
    "<noise>",      # 4
    "<sent>",       # 5 - sentence boundary
    "<evidence>",   # 6
    "<anomaly>",    # 7
    "<confidence>", # 8
    "<reasoning>",  # 9
    "<composition>",# 10
    "<temporal>",   # 11
    "<unk>",        # 12
]


class AamTokenizer:
    """AAM Sentence-Level + Subword BPE Hybrid Tokenizer.

    This tokenizer is specifically designed for the AAM Diffusion LLM:
    - It understands sentence boundaries (<sent> tokens)
    - It has special tokens for graph structure
    - It uses BPE for subword tokenization within sentences
    - It can encode/decode both plain text and graph-conditioned text

    Usage:
        tokenizer = AamTokenizer()
        tokenizer.train(texts, vocab_size=28000)

        # Encode text
        ids = tokenizer.encode("Berdasarkan analisis, pencuri adalah Diancang.")

        # Decode back
        text = tokenizer.decode(ids)

        # With graph structure tokens
        ids = tokenizer.encode_with_structure(
            "Pencuri = Diancang pair",
            evidence_nodes=["hefei", "diancang"],
            anomalies=[{"desc": "no external pill consumption"}],
        )
    """

    def __init__(self, config: Optional[TokenizerConfig] = None):
        """Initialize the tokenizer.

        Args:
            config: Tokenizer configuration. Uses defaults if None.
        """
        self.config = config or TokenizerConfig()

        # Build initial vocabulary with special tokens
        self.vocab: dict[str, int] = {}
        self.id_to_token: dict[int, str] = {}
        self._init_special_tokens()

        # BPE merges (learned during training)
        self.merges: dict[tuple[str, str], int] = {}
        self._bpe_cache: dict[str, str] = {}

        # Compiled patterns
        self._sentence_pattern = re.compile(
            r'(?<=[.!?])\s+|(?<=\n)\s*'
        )
        self._word_pattern = re.compile(
            r'\w+|[^\w\s]'
        )

        # Flag
        self._is_trained = False

    def _init_special_tokens(self) -> None:
        """Initialize special tokens in vocabulary."""
        for i, token in enumerate(SPECIAL_TOKENS):
            self.vocab[token] = i
            self.id_to_token[i] = token

    @property
    def pad_id(self) -> int:
        return self.vocab[self.config.pad_token]

    @property
    def bos_id(self) -> int:
        return self.vocab[self.config.bos_token]

    @property
    def eos_id(self) -> int:
        return self.vocab[self.config.eos_token]

    @property
    def mask_id(self) -> int:
        return self.vocab[self.config.mask_token]

    @property
    def noise_id(self) -> int:
        return self.vocab[self.config.noise_token]

    @property
    def sent_id(self) -> int:
        return self.vocab[self.config.sentence_boundary_token]

    @property
    def unk_id(self) -> int:
        return self.vocab.get("<unk>", len(SPECIAL_TOKENS) - 1)

    @property
    def vocab_size(self) -> int:
        """Current vocabulary size."""
        return len(self.vocab)

    @property
    def is_trained(self) -> bool:
        """Whether the tokenizer has been trained."""
        return self._is_trained

    def train(
        self,
        texts: list[str],
        vocab_size: Optional[int] = None,
    ) -> None:
        """Train the BPE tokenizer on a corpus.

        Args:
            texts: List of training texts.
            vocab_size: Target vocabulary size. Uses config if None.
        """
        target_vocab = vocab_size or self.config.bpe_vocab_size

        # Step 1: Pre-tokenize into words
        word_freqs: Counter = Counter()
        for text in texts:
            words = self._pre_tokenize(text)
            for word in words:
                word_freqs[word] += 1

        # Step 2: Initialize character-level vocabulary
        char_vocab: set[str] = set()
        for word in word_freqs:
            for char in word:
                char_vocab.add(char)

        # Add character tokens to vocabulary
        for char in sorted(char_vocab):
            if char not in self.vocab:
                idx = len(self.vocab)
                self.vocab[char] = idx
                self.id_to_token[idx] = char

        # Step 3: Split words into character sequences
        word_splits: dict[str, list[str]] = {}
        for word in word_freqs:
            word_splits[word] = list(word)
            # Add end-of-word marker
            if len(word_splits[word]) > 1:
                word_splits[word][-1] = word_splits[word][-1] + "</w>"

        # Step 4: Learn BPE merges
        n_merges = target_vocab - len(self.vocab)
        for i in range(n_merges):
            # Count pairs
            pair_freqs: Counter = Counter()
            for word, freq in word_freqs.items():
                symbols = word_splits.get(word, [])
                for j in range(len(symbols) - 1):
                    pair = (symbols[j], symbols[j + 1])
                    pair_freqs[pair] += freq

            if not pair_freqs:
                break

            # Find most frequent pair
            best_pair = pair_freqs.most_common(1)[0][0]

            # Record merge
            self.merges[best_pair] = i

            # Apply merge
            new_symbol = best_pair[0] + best_pair[1]
            for word in word_splits:
                symbols = word_splits[word]
                new_symbols = []
                j = 0
                while j < len(symbols):
                    if (
                        j < len(symbols) - 1
                        and symbols[j] == best_pair[0]
                        and symbols[j + 1] == best_pair[1]
                    ):
                        new_symbols.append(new_symbol)
                        j += 2
                    else:
                        new_symbols.append(symbols[j])
                        j += 1
                word_splits[word] = new_symbols

            # Add merged token to vocabulary
            if new_symbol not in self.vocab:
                idx = len(self.vocab)
                self.vocab[new_symbol] = idx
                self.id_to_token[idx] = new_symbol

        self._is_trained = True
        self._bpe_cache.clear()

    def _pre_tokenize(self, text: str) -> list[str]:
        """Pre-tokenize text into words.

        Args:
            text: Input text.

        Returns:
            List of words.
        """
        # Normalize unicode
        text = unicodedata.normalize("NFC", text)
        # Split into words and punctuation
        words = self._word_pattern.findall(text.lower())
        return words

    def _bpe_encode(self, word: str) -> list[str]:
        """Apply BPE to a single word.

        Args:
            word: Input word (lowercase).

        Returns:
            List of BPE tokens.
        """
        if word in self._bpe_cache:
            return self._bpe_cache[word].split()

        # Start with character-level split
        symbols = list(word)
        if len(symbols) > 1:
            symbols[-1] = symbols[-1] + "</w>"

        # Apply merges in order
        while len(symbols) > 1:
            # Find the pair with the lowest merge rank
            best_pair = None
            best_rank = float("inf")

            for i in range(len(symbols) - 1):
                pair = (symbols[i], symbols[i + 1])
                rank = self.merges.get(pair, float("inf"))
                if rank < best_rank:
                    best_rank = rank
                    best_pair = pair

            if best_pair is None or best_rank == float("inf"):
                break

            # Apply merge
            new_symbol = best_pair[0] + best_pair[1]
            new_symbols = []
            i = 0
            while i < len(symbols):
                if (
                    i < len(symbols) - 1
                    and symbols[i] == best_pair[0]
                    and symbols[i + 1] == best_pair[1]
                ):
                    new_symbols.append(new_symbol)
                    i += 2
                else:
                    new_symbols.append(symbols[i])
                    i += 1
            symbols = new_symbols

        # Cache result
        self._bpe_cache[word] = " ".join(symbols)
        return symbols

    def encode(self, text: str, add_special: bool = True) -> list[int]:
        """Encode text to token IDs.

        The encoding process:
        1. Split text into sentences
        2. Insert sentence boundary tokens between sentences
        3. BPE-encode each word within sentences
        4. Add BOS/EOS tokens if requested

        Args:
            text: Input text.
            add_special: Whether to add BOS/EOS tokens.

        Returns:
            List of token IDs.
        """
        ids = []

        if add_special:
            ids.append(self.bos_id)

        # Split into sentences
        sentences = self._split_sentences(text)

        for i, sentence in enumerate(sentences):
            if i > 0:
                ids.append(self.sent_id)  # Sentence boundary

            # Tokenize words in the sentence
            words = self._pre_tokenize(sentence)
            for word in words:
                if self._is_trained:
                    bpe_tokens = self._bpe_encode(word)
                    for token in bpe_tokens:
                        if token in self.vocab:
                            ids.append(self.vocab[token])
                        else:
                            ids.append(self.unk_id)
                else:
                    # Fallback: character-level encoding
                    for char in word:
                        if char in self.vocab:
                            ids.append(self.vocab[char])
                        else:
                            ids.append(self.unk_id)

        if add_special:
            ids.append(self.eos_id)

        return ids

    def encode_with_structure(
        self,
        text: str,
        evidence_nodes: Optional[list[str]] = None,
        compositions: Optional[list[str]] = None,
        anomalies: Optional[list[str]] = None,
        reasoning_steps: Optional[list[str]] = None,
        confidence: Optional[float] = None,
    ) -> list[int]:
        """Encode text with graph structure tokens.

        Adds structural tokens that represent the graph conditioning,
        so the model knows what kind of evidence/anomalies it's
        generating from.

        Args:
            text: The narrative text.
            evidence_nodes: List of evidence node labels.
            compositions: List of composition descriptions.
            anomalies: List of anomaly descriptions.
            reasoning_steps: List of reasoning step descriptions.
            confidence: Overall confidence score.

        Returns:
            List of token IDs with structure tokens.
        """
        ids = [self.bos_id]

        # Evidence section
        if evidence_nodes:
            ids.append(self.vocab["<evidence>"])
            for node in evidence_nodes:
                node_ids = self.encode(node, add_special=False)
                ids.extend(node_ids)
            ids.append(self.vocab["<evidence>"])  # Close section

        # Anomaly section
        if anomalies:
            ids.append(self.vocab["<anomaly>"])
            for anomaly in anomalies:
                anom_ids = self.encode(anomaly, add_special=False)
                ids.extend(anom_ids)
            ids.append(self.vocab["<anomaly>"])

        # Reasoning section
        if reasoning_steps:
            ids.append(self.vocab["<reasoning>"])
            for step in reasoning_steps:
                step_ids = self.encode(step, add_special=False)
                ids.extend(step_ids)
                ids.append(self.sent_id)
            ids.append(self.vocab["<reasoning>"])

        # Confidence
        if confidence is not None:
            ids.append(self.vocab["<confidence>"])
            # Encode confidence as a token (discretized)
            conf_bucket = min(int(confidence * 10), 9)
            conf_token = f"<conf_{conf_bucket}>"
            if conf_token in self.vocab:
                ids.append(self.vocab[conf_token])

        # Composition section
        if compositions:
            ids.append(self.vocab["<composition>"])
            for comp in compositions:
                comp_ids = self.encode(comp, add_special=False)
                ids.extend(comp_ids)
                ids.append(self.sent_id)
            ids.append(self.vocab["<composition>"])

        # Main narrative
        narrative_ids = self.encode(text, add_special=False)
        ids.extend(narrative_ids)

        ids.append(self.eos_id)
        return ids

    def decode(self, ids: list[int], skip_special: bool = False) -> str:
        """Decode token IDs back to text.

        Args:
            ids: List of token IDs.
            skip_special: Whether to skip special tokens in output.

        Returns:
            Decoded text string.
        """
        special_ids = set()
        if skip_special:
            for token in SPECIAL_TOKENS:
                if token in self.vocab:
                    special_ids.add(self.vocab[token])

        tokens = []
        for id_ in ids:
            if skip_special and id_ in special_ids:
                continue
            if id_ in self.id_to_token:
                tokens.append(self.id_to_token[id_])
            else:
                tokens.append("<unk>")

        # Join and clean up BPE tokens
        text = "".join(tokens)
        text = text.replace("</w>", " ")
        # Clean up sentence boundaries
        text = text.replace("<sent>", ". ")
        # Clean up multiple spaces
        text = re.sub(r'\s+', ' ', text).strip()

        return text

    def _split_sentences(self, text: str) -> list[str]:
        """Split text into sentences.

        Args:
            text: Input text.

        Returns:
            List of sentence strings.
        """
        sentences = self._sentence_pattern.split(text)
        return [s.strip() for s in sentences if s.strip()]

    def pad_sequence(
        self,
        ids: list[int],
        max_len: int,
        pad_id: Optional[int] = None,
    ) -> list[int]:
        """Pad a sequence to max_len.

        Args:
            ids: Token IDs.
            max_len: Target length.
            pad_id: Padding token ID. Uses config if None.

        Returns:
            Padded sequence.
        """
        padding_id = pad_id if pad_id is not None else self.pad_id
        if len(ids) >= max_len:
            return ids[:max_len]
        return ids + [padding_id] * (max_len - len(ids))

    def get_sentence_boundaries(self, ids: list[int]) -> list[int]:
        """Find sentence boundary positions in a token sequence.

        This is used by the diffusion model to identify which tokens
        belong to which sentence, enabling non-sequential generation
        and revision at the sentence level.

        Args:
            ids: Token IDs.

        Returns:
            List of indices where sentence boundaries occur.
        """
        boundaries = []
        for i, id_ in enumerate(ids):
            if id_ == self.sent_id:
                boundaries.append(i)
        return boundaries

    def save(self, path: str | Path) -> None:
        """Save tokenizer to file.

        Args:
            path: Output file path (JSON).
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        data = {
            "config": {
                "bpe_vocab_size": self.config.bpe_vocab_size,
                "max_sentences": self.config.max_sentences,
                "sentence_boundary_token": self.config.sentence_boundary_token,
                "pad_token": self.config.pad_token,
                "bos_token": self.config.bos_token,
                "eos_token": self.config.eos_token,
                "mask_token": self.config.mask_token,
                "noise_token": self.config.noise_token,
                "min_frequency": self.config.min_frequency,
            },
            "vocab": self.vocab,
            "merges": {f"{k[0]}|||{k[1]}": v for k, v in self.merges.items()},
            "is_trained": self._is_trained,
        }

        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    @classmethod
    def load(cls, path: str | Path) -> AamTokenizer:
        """Load tokenizer from file.

        Args:
            path: Input file path (JSON).

        Returns:
            Loaded AamTokenizer.
        """
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        config = TokenizerConfig(**data.get("config", {}))
        tokenizer = cls(config=config)

        # Restore vocabulary
        tokenizer.vocab = data["vocab"]
        tokenizer.id_to_token = {int(v): k for k, v in data["vocab"].items()}

        # Restore merges
        tokenizer.merges = {}
        for k_str, v in data.get("merges", {}).items():
            parts = k_str.split("|||")
            tokenizer.merges[(parts[0], parts[1])] = v

        tokenizer._is_trained = data.get("is_trained", False)

        return tokenizer

    def __len__(self) -> int:
        return self.vocab_size

    def __repr__(self) -> str:
        status = "trained" if self._is_trained else "untrained"
        return (
            f"AamTokenizer(vocab_size={self.vocab_size}, "
            f"merges={len(self.merges)}, status={status})"
        )
