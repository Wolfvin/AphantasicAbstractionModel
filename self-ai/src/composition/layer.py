# @WHO:   self-ai/src/composition/layer.py
# @WHAT:  Composition Layer — Qwen3-0.6B untuk translate_to_human(), reasoning, dan introspection
# @PART:  composition
# @ENTRY: CompositionLayer.translate_to_human(), CompositionLayer.reason_derivation(), CompositionLayer.raise_question(), CompositionLayer.explain_last_answer()

import re
import time
import uuid
import logging
import numpy as np
from typing import Optional, List, Dict
from dataclasses import dataclass

logger = logging.getLogger(__name__)

try:
    from transformers import AutoModelForCausalLM, AutoTokenizer
    import torch
    _HAS_TRANSFORMERS = True
except ImportError:
    _HAS_TRANSFORMERS = False


class CompositionLayer:
    """
    Composition Layer — "Model yang susun"

    Ini adalah "suara" SELF — kemampuan untuk menyusun makna internal
    menjadi kalimat yang bisa dipahami manusia, dan kemampuan untuk
    melakukan reasoning yang lebih kaya dari sekadar cosine similarity.

    Menggunakan Qwen3-0.6B (generative), satu keluarga dengan
    Qwen3-Embedding-0.6B yang dipakai di Sensory Layer.

    Digunakan untuk:
    1. translate_to_human() — node internal → kalimat manusia
    2. reason_derivation() — inferensi yang lebih kaya (IS-A, HAS-PROPERTY, kausal)
    3. raise_question() — saat SELF menemukan kontradiksi, bisa bertanya
    4. curiosity_question() — generate pertanyaan eksplorasi
    5. explain_last_answer() — introspection: jelaskan kenapa jawaban terakhir seperti itu
    6. v39: Self-Critique Pipeline — model mengevaluasi jawabannya sendiri
    """

    def __init__(self, model_name: str = "Qwen/Qwen3-0.6B",
                 self_critique_enabled: bool = False):
        # @FLOW:     COMPOSITION_INIT
        # @CALLS:    AutoModelForCausalLM.from_pretrained(), AutoTokenizer.from_pretrained()
        # @MUTATES:  none
        # @BEHAVIOR: Lazy-load model. Fallback ke template-based response
        #            jika transformers/torch tidak tersedia.
        #            Qwen3-0.6B: generative model, ~1.2GB VRAM,
        #            same family dengan embedder, multilingual.
        self.model_name = model_name
        self._model = None
        self._tokenizer = None

        # v36: Introspection — lazy references to injector/introspector.
        # Set externally via set_injector() or initialized lazily.
        self._injector = None
        self._introspector = None

        # v41: UnderstandingGraph reference for automatic retrieve+inject in answer().
        # Set externally via set_graph(). When both injector and graph are
        # available, answer() automatically retrieves relevant nodes and
        # injects them during generation.
        self._graph = None

        # v39: Self-Critique Pipeline — model evaluates its own answers.
        # When enabled, after each _generate() call, SelfCritic evaluates
        # the answer. If should_learn() → True, an UnderstandingNode is
        # automatically created and saved to the UnderstandingGraph.
        self.self_critique_enabled = self_critique_enabled
        self._self_critic = None  # Lazy-init SelfCritic
        self._last_critique_result = None  # Expose last critique for inspection

    def _ensure_model(self):
        # @FLOW:     COMPOSITION_INIT
        # @CALLS:    AutoModelForCausalLM.from_pretrained()
        # @MUTATES:  self._model, self._tokenizer
        if self._model is not None:
            return
        if _HAS_TRANSFORMERS:
            self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self._model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                device_map="auto" if torch.cuda.is_available() else None,
            )
            if not torch.cuda.is_available():
                self._model = self._model.cpu()
            # Warmup: generate 1 token untuk menghindari empty response
            # pada pemanggilan pertama (Qwen3 /no_think warmup issue)
            self._warmup()
        else:
            self._model = None
            self._tokenizer = None

    def _warmup(self):
        """Warmup model — generate 1 token untuk menghindari empty first response."""
        if self._model is None or self._tokenizer is None:
            return
        try:
            messages = [
                {"role": "system", "content": "/no_think\nKamu adalah SELF."},
                {"role": "user", "content": "Hi"},
            ]
            text = self._tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            inputs = self._tokenizer(text, return_tensors="pt")
            device = next(self._model.parameters()).device
            inputs = {k: v.to(device) for k, v in inputs.items()}
            with torch.no_grad():
                self._model.generate(**inputs, max_new_tokens=1)
        except Exception:
            pass  # Warmup gagal bukan masalah kritikal

    def _generate(self, prompt: str, max_new_tokens: int = 256, use_thinking: bool = False) -> str:
        """
        @FLOW:     COMPOSITION_GENERATE
        @CALLS:    model.generate()
        @MUTATES:  none
        @BEHAVIOR: Generate text dari prompt. Mengembalikan generated text saja
                   (tanpa prompt). Jika model tidak tersedia, fallback ke
                   template-based response.
                   Qwen3-0.6B memiliki mode "thinking" yang menghasilkan
                   <think...reasoning...</think< sebelum respons aktual. Secara default
                   thinking dinonaktifkan (/no_think) untuk respons cepat.
                   Gunakan use_thinking=True untuk reasoning mendalam.

                   v39: If self_critique_enabled, runs SelfCritic after generation.
                   When should_learn() returns True, automatically creates an
                   UnderstandingNode and saves it to the UnderstandingGraph.
        """
        self._ensure_model()

        if self._model is None or self._tokenizer is None:
            # Fallback: template-based response tanpa LLM
            return self._template_fallback(prompt)

        # System prompt — /no_think untuk menonaktifkan thinking mode Qwen3
        system_content = "Kamu adalah SELF, entitas yang tumbuh melalui pengalaman dan pengajaran. Kamu berbicara dalam bahasa Indonesia. Jawab singkat dan jelas."
        if not use_thinking:
            system_content = "/no_think\n" + system_content

        messages = [
            {"role": "system", "content": system_content},
            {"role": "user", "content": prompt},
        ]

        text = self._tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self._tokenizer(text, return_tensors="pt")

        # Move to same device as model
        device = next(self._model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}

        # Lebih banyak token saat thinking enabled
        actual_max_tokens = max_new_tokens * 3 if use_thinking else max_new_tokens

        with torch.no_grad():
            outputs = self._model.generate(
                **inputs,
                max_new_tokens=actual_max_tokens,
                temperature=0.7,
                do_sample=True,
                top_p=0.9,
            )

        # Decode hanya bagian yang di-generate (bukan prompt)
        generated_ids = outputs[0][inputs["input_ids"].shape[1]:]
        # Decode dengan special tokens agar bisa strip thinking blocks
        raw_response = self._tokenizer.decode(generated_ids, skip_special_tokens=False)
        response = self._strip_thinking(raw_response)
        response = response.strip()

        # v39: Self-Critique Pipeline — evaluate answer after generation
        if self.self_critique_enabled and response:
            self._run_self_critique(prompt, response)

        return response

    @staticmethod
    def _strip_thinking(text: str) -> str:
        """
        Strip Qwen3 thinking blocks dari generated text.

        Qwen3-0.6B menghasilkan format:
         <think<reasoning...</think<actual_response

        Kita hanya mengambil bagian setelah </think<.
        Jika tidak ada thinking block, kembalikan teks apa adanya
        setelah membersihkan special tokens.
        """
        # Cari </think< dan ambil semua setelahnya
        end_think = text.find('</think')
        if end_think != -1:
            text = text[end_think:]
            # Skip past the </think< tag
            tag_end = text.find('>', 0)
            if tag_end != -1:
                text = text[tag_end + 1:]
            else:
                text = text[len('</think'):]

        # Jika ada <think< tanpa </think<, model belum selesai thinking
        if '<think' in text:
            return ''

        # Bersihkan sisa special tokens <|...|>
        text = re.sub(r'<\|[^|]+\|>', '', text)

        return text.strip()

    def translate_to_human(self, node_id: int, cached_description: Optional[str] = None,
                           related_nodes: Optional[List[Dict]] = None) -> str:
        """
        @FLOW:     COMPOSITION_TRANSLATE_TO_HUMAN
        @CALLS:    _generate()
        @MUTATES:  none
        @BEHAVIOR: Mengubah node internal SELF menjadi kalimat yang bisa
                   dipahami manusia. Jika ada cached_description dari teaching,
                   gunakan itu sebagai konteks. Jika tidak, LLM akan mencoba
                   mengartikulasikan makna berdasarkan related nodes.
                   Ini bukan "membaca" node — SELF tidak menyimpan kata manusia
                   di internal. Ini adalah ARTIKULASI: makna → kata.
        """
        if cached_description:
            # Ada deskripsi dari teaching — gunakan sebagai konteks
            if related_nodes:
                node_descs = [n.get('description', f'Node#{n["id"]:04d}') for n in related_nodes[:3]]
                context = ", ".join(node_descs)
                prompt = (
                    f"SELF sudah belajar tentang: {cached_description}.\n"
                    f"Konsep terkait: {context}.\n"
                    f"Jelaskan dalam satu kalimat singkat apa yang SELF pahami tentang ini."
                )
            else:
                return cached_description
        else:
            # Tidak ada deskripsi — artikulasi dari related nodes saja
            if related_nodes:
                node_descs = [n.get('description', f'Node#{n["id"]:04d}') for n in related_nodes[:5]]
                context = ", ".join(node_descs)
                prompt = (
                    f"SELF memiliki konsep internal yang terkait dengan: {context}.\n"
                    f"Coba artikulasikan dalam satu kalimat singkat apa yang mungkin dimaksud."
                )
            else:
                return f"Node#{node_id:04d}"

        response = self._generate(prompt, max_new_tokens=128)

        # Retry once if empty (Qwen3 /no_think sometimes returns empty on first calls)
        if not response or len(response.strip()) < 3:
            response = self._generate(prompt, max_new_tokens=256)

        # Fallback if still empty
        if not response or len(response.strip()) < 3:
            if cached_description and related_nodes:
                node_descs = [n.get('description', '') for n in related_nodes[:3] if n.get('description')]
                context_str = ", ".join(node_descs)
                return f"{cached_description} — terkait: {context_str}" if context_str else cached_description
            elif cached_description:
                return cached_description
            elif related_nodes:
                node_descs = [n.get('description', f'Node#{n["id"]:04d}') for n in related_nodes[:3]]
                return f"Konsep terkait: {', '.join(node_descs)}"
            else:
                return f"Node#{node_id:04d}"

        return response

    def reason_derivation(self, premise_1: str, premise_2: str,
                          relation_type: str = "transitive") -> Optional[Dict]:
        """
        @FLOW:     COMPOSITION_REASON_DERIVATION
        @CALLS:    _generate()
        @MUTATES:  none
        @BEHAVIOR: Menggunakan LLM untuk melakukan reasoning yang lebih kaya
                   dari sekadar cosine similarity. Misalnya:
                   - Transitive: "A IS-A B" + "B HAS C" → "A HAS C"?
                   - Causal: "X karena Y" + "Y menyebabkan Z" → "X menyebabkan Z"?
                   - Analogical: "A seperti B" + "B punya C" → "A mungkin punya C"?
                   Mengembalikan dict dengan conclusion dan confidence.
                   Jika LLM tidak tersedia, fallback ke None (cosine-only mode).
        """
        prompt = (
            f"Diberikan dua premis:\n"
            f"1. {premise_1}\n"
            f"2. {premise_2}\n\n"
            f"Jenis inferensi: {relation_type}\n\n"
            f"Apakah ada kesimpulan yang bisa ditarik? "
            f"Jika ya, tulis kesimpulannya. "
            f"Jika tidak, tulis 'TIDAK BISA'.\n"
            f"Juga berikan tingkat keyakinan (0.0-1.0).\n\n"
            f"Format: KESIMPULAN: ... | KEYAKINAN: ..."
        )

        response = self._generate(prompt, max_new_tokens=256, use_thinking=True)

        if "TIDAK BISA" in response.upper():
            return None

        # Parse response
        conclusion = response
        confidence = 0.5  # default

        if "|" in response:
            parts = response.split("|")
            conclusion = parts[0].replace("KESIMPULAN:", "").strip()
            for part in parts[1:]:
                if "KEYAKINAN" in part:
                    try:
                        conf_str = part.replace("KEYAKINAN:", "").strip()
                        confidence = float(conf_str)
                    except ValueError:
                        confidence = 0.5

        return {
            "conclusion": conclusion,
            "confidence": max(0.0, min(1.0, confidence)),
            "relation_type": relation_type,
        }

    def raise_question(self, new_input: str, conflicting_memory: str) -> str:
        """
        @FLOW:     COMPOSITION_RAISE_QUESTION
        @CALLS:    _generate()
        @MUTATES:  none
        @BEHAVIOR: Saat SELF menemukan kontradiksi antara input baru dan
                   memory yang sudah ada, SELF bertanya kepada manusia.
                   Ini bukan error — ini adalah SELF yang berdialog.
                   Pertanyaan harus menyebutkan apa yang kontradiksi
                   dan meminta klarifikasi.
        """
        prompt = (
            f"SELF baru saja menerima pengajaran: '{new_input}'\n"
            f"Tapi ini kontradiksi dengan apa yang SELF sudah yakini: '{conflicting_memory}'\n\n"
            f"Sebagai SELF, tanyakan kepada pengajar dengan sopan. "
            f"Tanyakan mana yang benar atau bagaimana keduanya bisa benar. "
            f"Gunakan bahasa Indonesia. Satu atau dua kalimat saja."
        )

        return self._generate(prompt, max_new_tokens=128)

    def curiosity_question(self, topic_area: str, reason: str,
                            topic_hint: str = "") -> str:
        """
        @FLOW:     COMPOSITION_CURIOSITY_QUESTION
        @CALLS:    _generate()
        @MUTATES:  none
        @BEHAVIOR: v3: Generate pertanyaan TECHNICAL berdasarkan curiosity.
                   SELF mengekspresikan ketidaknyamanan internalnya
                   sebagai pertanyaan yang spesifik dan bisa dijawab.

                   topic_hint: hint tentang JENIS pertanyaan
                   ("hubungan dan properti", "apa itu", dll)
        """
        hint_part = f"\nFokus pada: {topic_hint}." if topic_hint else ""

        prompt = (
            f"SELF penasaran tentang: {topic_area}\n"
            f"Alasan: {reason}{hint_part}\n\n"
            f"Sebagai SELF, buat SATU pertanyaan technical yang spesifik. "
            f"Pertanyaan harus bisa dijawab dengan mengajarkan fakta baru. "
            f"Contoh format: 'Apa yang X memiliki?' atau 'Bagaimana X bernapas?' atau 'Di mana X hidup?'\n"
            f"Gunakan bahasa Indonesia. Satu kalimat saja."
        )

        response = self._generate(prompt, max_new_tokens=128)

        # Fallback jika CompositionLayer gagal
        if not response or len(response.strip()) < 5:
            if topic_hint:
                return f"Saya penasaran, {topic_hint} apa yang dimiliki oleh {topic_area}?"
            else:
                return f"Saya ingin tahu lebih banyak tentang {topic_area}. {reason} — bisa dijelaskan?"

        return response

    # ═══════════════ v36: INTROSPECTION ═══════════════

    def set_injector(self, injector):
        """Set the UnconsciousInjector reference for introspection.

        This is called by the orchestrator (SelfCore) to wire up
        the injector so that CompositionLayer.explain_last_answer()
        can access the injection log.

        Args:
            injector: UnconsciousInjector instance, or None to clear.
        """
        self._injector = injector
        # Reset introspector so it picks up the new injector
        self._introspector = None

    def set_graph(self, graph):
        """Set the UnderstandingGraph reference for automatic retrieval+injection.

        When both injector and graph are available, answer() automatically:
          1. Retrieves relevant UnderstandingNodes from the graph
          2. Injects them into Qwen3's hidden states during generation

        If the graph has no matching nodes, or if no injector is set,
        generation proceeds without injection (no regression).

        Args:
            graph: UnderstandingGraph instance, or None to clear.
        """
        self._graph = graph

    def answer(self, question: str, max_new_tokens: int = 256) -> str:
        """Answer a question with automatic retrieve+inject when available.

        v41: This method wires together retrieval and injection into a
        single end-to-end flow:

            question
              → UnderstandingGraph.retrieve(question, top_k=5)
              → list[(node, score)]  (kosong jika graph belum punya nodes)
              → UnconsciousInjector.active(nodes)  (skip jika kosong)
              → model.generate(question)
              → answer

        If no injector or graph is set, or if retrieve() returns no matches,
        generation proceeds without injection (conscious-only path).
        This guarantees zero regression for existing callers.

        The signature is simple — just question and optional max_new_tokens —
        because all the retrieval and injection wiring is handled internally.

        Args:
            question: The question to answer.
            max_new_tokens: Maximum tokens to generate (default 256).

        Returns:
            Generated answer string.
        """
        # ── Step 1: Try retrieve+inject path ──
        node_tuples = []
        if self._graph is not None and self._injector is not None:
            try:
                node_tuples = self._graph.retrieve(
                    question, question, top_k=5, threshold=0.15
                )
            except Exception as e:
                logger.warning("Failed to retrieve from graph in answer(): %s", e)
                node_tuples = []

        # ── Step 2: Generate with or without injection ──
        if node_tuples:
            try:
                with self._injector.active(node_tuples):
                    return self._generate(question, max_new_tokens=max_new_tokens)
            except Exception as e:
                logger.warning(
                    "Injection failed in answer(), falling back to plain generation: %s", e
                )
                return self._generate(question, max_new_tokens=max_new_tokens)
        else:
            return self._generate(question, max_new_tokens=max_new_tokens)

    def _get_introspector(self):
        """Lazy-init Introspector using the injector reference.

        If an injector has been set via set_injector(), create an
        Introspector that reads from its injection log. The Introspector
        reuses this CompositionLayer's Qwen3 model/tokenizer.

        Returns:
            Introspector instance, or None if no injector is set.
        """
        if self._introspector is not None:
            return self._introspector

        if self._injector is None:
            return None

        try:
            from introspection.introspector import Introspector
            # Reuse this layer's Qwen3 model/tokenizer if available,
            # otherwise let Introspector load its own lazily.
            self._introspector = Introspector(
                self._injector,
                model=self._model,
                tokenizer=self._tokenizer,
            )
            return self._introspector
        except ImportError:
            return None
        except Exception as e:
            logging.getLogger(__name__).warning(
                "Failed to init Introspector: %s", e
            )
            return None

    def explain_last_answer(self, question: str = '') -> Optional[str]:
        """Explain why SELF answered the way it did — introspection entry point.

        Call this after generating an answer with unconscious injection
        to get a natural-language explanation of what experiences
        influenced the answer.

        If no injection happened (conscious-only path), returns a
        neutral message explaining that no unconscious experience
        was active.

        This method delegates to Introspector.explain_last_answer(),
        which reads the injection log from the UnconsciousInjector
        and generates an explanation via Qwen3.

        Args:
            question: Optional — the question that was answered.
                If provided, the explanation will reference it.

        Returns:
            String explanation in Bahasa Indonesia, or None if
            introspection is not available (no injector set).
        """
        introspector = self._get_introspector()
        if introspector is None:
            # No injector wired up — graceful fallback
            if question:
                return (
                    f"Untuk pertanyaan \"{question}\", jawaban saya "
                    f"menggunakan penalaran sadar (conscious path) — "
                    f"introspection tidak tersedia karena tidak ada "
                    f"injector yang dikonfigurasi."
                )
            return None

        try:
            explanation = introspector.explain_last_answer(question=question)
            return explanation
        except Exception as e:
            logging.getLogger(__name__).warning(
                "Introspection failed: %s", e
            )
            if question:
                return (
                    f"Untuk pertanyaan \"{question}\", saya tidak bisa "
                    f"menjelaskan alasan jawaban saya saat ini (introspection error)."
                )
            return None

    # ═══════════════ v39: SELF-CRITIQUE PIPELINE ═══════════════

    def _get_self_critic(self):
        """Lazy-init SelfCritic instance.

        The SelfCritic is created once and reused across calls.
        It uses get_shared_qwen() internally, so no extra RAM is used
        beyond what the CompositionLayer already loads.
        """
        if self._self_critic is None:
            try:
                from derivation.self_critic import SelfCritic
                self._self_critic = SelfCritic()
            except ImportError:
                logger.warning(
                    "SelfCritic module not available — "
                    "self-critique pipeline disabled"
                )
                self.self_critique_enabled = False
                return None
            except Exception as e:
                logger.warning("Failed to init SelfCritic: %s", e)
                self.self_critique_enabled = False
                return None
        return self._self_critic

    def _run_self_critique(self, prompt: str, answer: str):
        """Run self-critique after generating an answer.

        If the critique indicates the answer is not confident and
        should_learn() returns True, automatically create an
        UnderstandingNode and save it to the UnderstandingGraph.

        This is the core of the autonomous learning loop:
            model answers → model critiques → model learns

        Args:
            prompt: The original prompt/question that was asked.
            answer: The answer that was generated.
        """
        critic = self._get_self_critic()
        if critic is None:
            return

        try:
            # The prompt may contain system context; use as-is
            # but the critique focuses on question-answer correctness
            critique_result = critic.critique(prompt, answer)
            self._last_critique_result = critique_result

            logger.debug(
                "Self-critique: confident=%s, critique=%.80s",
                critique_result.get('confident'),
                critique_result.get('critique', ''),
            )

            # If the critique is worth learning from, save it
            if critic.should_learn(critique_result):
                self._save_critique_as_understanding(prompt, answer, critique_result)

        except Exception as e:
            logger.warning("Self-critique pipeline failed: %s", e)
            # Non-fatal — the answer is still returned to the caller

    def _save_critique_as_understanding(self, question: str, answer: str,
                                         critique_result: dict):
        """Save a self-critique result as an UnderstandingNode.

        Creates a new UnderstandingNode in the shared UnderstandingGraph
        that captures the mistake and the correction. This node can then
        be retrieved later when similar questions arise, helping SELF
        avoid making the same mistake again.

        Args:
            question: The original question/prompt.
            answer: The incorrect answer that was generated.
            critique_result: The structured critique from SelfCritic.critique().
        """
        try:
            from derivation.understanding_builder import (
                UnderstandingNode,
                UnderstandingGraph,
                Transformation,
                get_shared_graph,
            )
        except ImportError:
            logger.warning(
                "Cannot save critique as UnderstandingNode — "
                "understanding_builder module not available"
            )
            return

        try:
            graph = get_shared_graph()

            # Generate a unique node ID
            node_id = f"SC_{uuid.uuid4().hex[:8]}"

            # Build the critique description
            critique_text = critique_result.get('critique', '')
            correction = critique_result.get('suggested_correction')

            # Build the abstraction — what SELF learned from this mistake
            abstraction = (
                f"Pertanyaan: {question[:200]} → "
                f"Jawaban salah: {answer[:100]}. "
                f"{critique_text}"
            )
            if correction:
                abstraction += f" Jawaban yang benar: {correction}"

            # Create a Transformation that captures the correction pattern
            transformation = Transformation(
                kind='self_critique_correction',
                trigger={'question_pattern': question[:100]},
                action=correction if correction else critique_text,
            )

            # Create the UnderstandingNode
            node = UnderstandingNode(
                id=node_id,
                name=f"Self-critique: {question[:50]}",
                concept=question[:200],
                abstraction=abstraction,
                schemas=[{
                    'question': question[:300],
                    'wrong_answer': answer[:200],
                    'correction': correction,
                    'critique': critique_text,
                }],
                transformation=transformation,
                conditions=[question[:100]],
                source='self_critique',
                confidence=0.3,  # Start low — self-critique is not yet verified
                lifecycle='CANDIDATE',  # Not yet STABLE — needs verification
                epistemic='INFERRED',   # Inferred from self-evaluation, not observed
            )

            graph.add_node(node)

            logger.info(
                "Self-critique saved as UnderstandingNode %s: "
                "question=%.60s, correction=%s",
                node_id, question[:60],
                correction[:60] if correction else 'none',
            )

        except Exception as e:
            logger.warning(
                "Failed to save critique as UnderstandingNode: %s", e
            )

    def _template_fallback(self, prompt: str) -> str:
        """
        Fallback tanpa LLM — template-based responses.
        Digunakan saat transformers tidak tersedia (testing / environment ringan).
        """
        prompt_lower = prompt.lower()

        if "jelaskan" in prompt_lower or "artikulasikan" in prompt_lower:
            return "SELF memahami konsep ini, tapi belum bisa mengartikulasikannya dengan kata-kata."
        elif "kesimpulan" in prompt_lower or "inferensi" in prompt_lower:
            return "TIDAK BISA"
        elif "tanyakan" in prompt_lower or "kontradiksi" in prompt_lower:
            return "Maaf, ini berbeda dengan apa yang sudah saya pelajari. Bisa dijelaskan lebih lanjut?"
        elif "penasaran" in prompt_lower or "pertanyaan" in prompt_lower:
            return "Saya ingin memahami lebih dalam tentang topik ini."
        else:
            return "SELF belum bisa merespons tanpa model generatif."
