# @WHO:   self-ai/src/derivation/engine.py
# @WHAT:  3-Strategy derivation engine + confidence reconciler + text comprehension + correction detection
# @PART:  self-ai/derivation
# @ENTRY: DerivationEngine.derive(), DerivationEngine.derive_from_text()

"""Derivation Engine - 3-Strategy + Confidence Reconciler
Strategies: Learned Rules + TransE Geometric + Operational Schema
Bug #1 fix: defaultdict import
Bug #3 fix: None guard in explain_derivation
v10.1: PERIMETER, TIME_DURATION, improved text comprehension with multi-step.
v11:    Text comprehension routing: if question is about who/what/why/how/when/where/
       message/feeling → route to TextComprehension. If quantitative ("berapa") →
       use existing operational schema.
"""
import re
from collections import defaultdict
from typing import Optional, Dict, Any


class DerivationEngine:
    def __init__(self, self_core=None):
        self.self_core = self_core
        self.rule_learner = None
        self.transe = None
        self.operational = None
        self.text_comprehension = None

    def _init_modules(self):
        if self.rule_learner is None:
            from derivation.rule_learner import RuleLearner
            self.rule_learner = RuleLearner(self.self_core)
        if self.transe is None:
            from derivation.rule_learner import TransEEmbedding
            self.transe = TransEEmbedding(d=64)
        if self.operational is None:
            from derivation.operational import OperationalSchema
            self.operational = OperationalSchema(self.self_core)
        if self.text_comprehension is None:
            from derivation.text_comprehension import TextComprehension
            self.text_comprehension = TextComprehension(self.self_core)

    # @FLOW: DERIVE
    # @CALLS: _apply_learned_rules, _apply_transe, _apply_operational, _reconcile
    # @MUTATES: none
    # @BEHAVIOR: Derives answer using 3 strategies. Operational schema is primary for math.
    #            Supports SUBTRACT, ADD, MULTIPLY, DIVIDE, FRACTION_MULTIPLY, PERIMETER,
    #            TIME_DURATION operations.
    def derive(self, text: str, context: dict = None) -> dict:
        """Derive answer using understanding-first pipeline + legacy strategies.

        v33: Understanding graph is now the PRIMARY strategy.
        Priority: Understanding > Self-correction > Legacy > LLM

        The understanding graph is SELF's autonomous reasoning — it applies
        self-built transformations WITHOUT LLM. Legacy strategies (rules,
        TransE, operational) are fallback only.
        """
        self._init_modules()
        context = context or {}

        # Strategy 0 (PRIMARY): Understanding graph pipeline
        understanding_result = self._apply_understanding_pipeline(text, context)
        if understanding_result and understanding_result.get('confidence', 0) >= 0.35:
            return understanding_result

        results = {}

        # Keep understanding result for reconciliation
        if understanding_result:
            results['understanding'] = understanding_result

        # Strategy 1: Learned rules
        rule_result = self._apply_learned_rules(text, context)
        results['rule'] = rule_result

        # Strategy 2: TransE geometric
        transe_result = self._apply_transe(text, context)
        results['transe'] = transe_result

        # Strategy 3: Operational schema
        op_result = self._apply_operational(text, context)
        results['operational'] = op_result

        # Reconcile with confidence
        final = self._reconcile(results)

        # v33: If reconciliation failed, try understanding pipeline as last resort
        # (it may have a low-confidence answer better than nothing)
        if final.get('answer') is None and understanding_result and understanding_result.get('answer'):
            return understanding_result

        return final

    def _apply_understanding_pipeline(self, text: str, context: dict) -> Optional[dict]:
        """Apply understanding graph pipeline — SELF's PRIMARY reasoning.

        This is the v33 understanding-first approach:
        1. Retrieve matching understandings via embedding
        2. Try applying the best match (transformation, no LLM)
        3. If that fails, try composing multiple understandings via Qwen3
        4. Return result or None
        """
        try:
            from derivation.understanding_builder import get_shared_graph
            graph = get_shared_graph()

            if not graph._nodes:
                return None

            # Step 1: Retrieve matching understandings
            question = text  # For direct questions like "8 × 7 = ?"
            matches = graph.find_matching_multi(
                text, question, top_k=3, threshold=0.15
            )

            if not matches:
                return None

            # Step 2: Try applying each understanding individually
            for node, score in matches:
                if node.transformation is None:
                    continue
                result = graph.apply(node, text, question)
                if result and result.get('answer'):
                    # Boost confidence based on embedding similarity score
                    base_confidence = result.get('confidence', 0.55)
                    # Embedding similarity boosts confidence
                    boosted = min(0.95, base_confidence + (score - 0.3) * 0.3)
                    result['confidence'] = max(base_confidence, boosted)
                    result['method'] = f'understanding_{node.transformation.kind}'
                    result['embedding_score'] = float(score)
                    return result

            # Step 3: Compose multiple understandings
            if len(matches) >= 1:
                try:
                    from derivation.understanding_composer import get_shared_composer
                    composer = get_shared_composer()

                    composed_answer = composer.compose_answer_from_understandings(
                        text, question, matches
                    )

                    if composed_answer:
                        return {
                            'answer': composed_answer,
                            'confidence': 0.55,
                            'method': 'understanding_composition',
                            'explanation': f'Composed from {len(matches)} understandings via Qwen3',
                            'source': 'composed',
                        }
                except ImportError:
                    pass
                except Exception:
                    pass

        except ImportError:
            pass
        except Exception:
            pass

        return None

    # @FLOW: DERIVE_TEXT
    # @CALLS: GrammarParser, _is_text_comprehension_question, derive_from_comprehension,
    #         OperationalSchema.compute_from_facts, derive
    # @MUTATES: none
    # @BEHAVIOR: Derives answer from narrative text + question. Routes to
    #            TextComprehension if question is about who/what/why/how/when/where/
    #            message/feeling. Uses operational schema for quantitative ("berapa") questions.
    #            Correction handling is now done by TrainingAgent (explicit intent).
    def derive_from_text(self, narrative_text: str, question: str) -> dict:
        """Derive answer from narrative text + question - Grade 4 text comprehension
        v11: Routes to TextComprehension for non-quantitative questions.
        For quantitative questions, tries operational schema first, then falls back
        to TextComprehension if the operational schema fails (e.g., the number is
        directly stated, not computed).
        v43: Correction detection removed — now handled by TrainingAgent (explicit intent).
        """
        self._init_modules()

        try:
            from grammar.parser import GrammarParser
        except ImportError:
            from grammar.simple_parser import SimpleParser as GrammarParser

        parser = GrammarParser(self.self_core)

        # Check if this is a text comprehension question (not quantitative)
        is_text_q = self._is_text_comprehension_question(question)

        if is_text_q:
            return self.derive_from_comprehension(narrative_text, question)

        # Parse the narrative text to extract facts
        parsed_narrative = parser.parse(narrative_text)
        facts = parsed_narrative.get('extracted_facts', [])

        # Parse the question
        parsed_question = parser.parse(question)

        # v20: Handle per-entity number aggregation BEFORE operational schema
        # "Berapa jumlah kelereng Andi?" → sum all numbers in Andi's sentence
        entity_sum = self._try_entity_number_sum(narrative_text, question)
        if entity_sum is not None:
            return entity_sum

        # Try to answer using operational schema with facts
        result = self.operational.compute_from_facts(facts, question)

        # v18: Check for kembalian (change) pattern — the single most
        # common multi-step math pattern in elementary school.
        # If the question asks for "kembalian" but compute_from_facts
        # only returned a MULTIPLY result (not SUBTRACT), we need
        # the full multi-step derivation.
        if result is not None and 'kembalian' in question.lower():
            # Verify: is this a complete answer or just partial (multiply only)?
            # Check if the result equals just qty × price (partial)
            all_nums = []
            for fact in facts:
                for n in fact.get('numbers', []):
                    all_nums.append(n['value'])
            # Add question numbers too
            for n in parsed_question.get('numbers', []):
                all_nums.append(n['value'] if isinstance(n, dict) else n)

            if len(all_nums) >= 3:
                # We have at least 3 numbers — likely needs multi-step
                # qty, price, payment → answer should be payment - (qty × price)
                # If result == qty × price, we only computed partial (total cost)
                # We still need to subtract from payment for kembalian
                qty_candidates = [v for v in all_nums if v < 100]
                price_candidates = [v for v in all_nums if v >= 100 and v < max(all_nums)]
                payment = max(all_nums)

                if qty_candidates and price_candidates and payment:
                    total_cost = qty_candidates[0] * price_candidates[0]
                    if abs(result - total_cost) < 1 and payment > total_cost:
                        # Result is just total_cost, but question asks for kembalian
                        # Compute: payment - total_cost = change
                        result = payment - total_cost

        if result is not None:
            q_op = parsed_question.get('operation_hint', 'UNKNOWN')
            if q_op == 'UNKNOWN' and facts:
                q_op = facts[0].get('operation_hint', 'operational')
            return {
                'answer': result,
                'confidence': 0.65,
                'method': f'text_comprehension_{q_op}',
                'explanation': f"From narrative: {len(facts)} facts extracted, question: {question}",
                'all_results': {
                    'operational': {'answer': result, 'confidence': 0.65, 'method': 'text_comprehension'},
                    'facts': facts
                }
            }

        # Fallback: try direct derivation on combined text first (handles fractions, etc.)
        combined = f"{narrative_text} {question}"
        direct_result = self.derive(combined)
        if direct_result.get('answer') is not None:
            direct_result['method'] = f"text_fallback_{direct_result.get('method', 'unknown')}"
            return direct_result

        # Fallback: try text comprehension (for "berapa" questions where the
        # answer is directly stated, not computed)
        if 'berapa' in question.lower():
            tc_result = self.derive_from_comprehension(narrative_text, question)
            if tc_result.get('answer') is not None:
                return tc_result

        return {
            'answer': None,
            'confidence': 0.0,
            'method': 'text_comprehension_failed',
            'explanation': f"Could not derive answer from {len(facts)} facts",
            'all_results': {'facts': facts}
        }

    # @FLOW: DERIVE_COMPREHENSION
    # @CALLS: TextComprehension.comprehend
    # @MUTATES: none
    # @BEHAVIOR: Uses TextComprehension module to answer non-quantitative
    #            questions about narrative text.
    def derive_from_comprehension(self, narrative_text: str, question: str) -> dict:
        """Derive answer using TextComprehension module — for reading comprehension questions"""
        self._init_modules()
        result = self.text_comprehension.comprehend(narrative_text, question)
        return result

    # v43: _CORRECTION_PATTERNS, _detect_correction(), _handle_correction() DELETED.
    # Correction handling is now done by TrainingAgent (explicit intent).
    # See self-ai/src/training/training_agent.py

    def _try_entity_number_sum(self, narrative_text: str, question: str) -> Optional[dict]:
        """v20: Handle per-entity number aggregation.

        "Berapa jumlah kelereng Andi?" → sum all numbers in Andi's sentence
        "Andi punya 5 kelereng merah dan 3 kelereng biru" → 5 + 3 = 8

        This handles the case where multiple numbers belong to the same entity
        and need to be summed, but the operational schema only picks the first one.
        """
        import re as _re
        ql = question.lower()
        tl = narrative_text.lower()

        # Must be a "berapa" question
        if 'berapa' not in ql:
            return None

        # Must mention "jumlah" or "total" or "semua" or have a specific entity name
        has_jumlah = any(w in ql for w in ['jumlah', 'total', 'semua'])

        # Extract the target entity name from the question
        entity_name = None
        q_names = _re.findall(r'\b([A-Z][a-z]+)\b', question)
        # Filter out question words that might be capitalized
        question_words = {'berapa', 'siapa', 'mengapa', 'kenapa', 'kapan', 'dimana',
                          'bagaimana', 'apakah', 'berapa', 'mengapa', 'dari'}
        q_names = [n for n in q_names if n.lower() not in question_words]
        if q_names:
            entity_name = q_names[0].lower()

        if entity_name is None:
            m = _re.search(r'jumlah\s+\w+\s+(\w+)', ql)
            if m:
                entity_name = m.group(1)

        if entity_name is None:
            return None

        # Skip common words that aren't entity names
        skip_names = {'kelereng', 'buah', 'ekor', 'orang', 'lembar', 'batang', 'butir',
                      'bungkus', 'botol', 'gelas', 'porsi', 'mangkok', 'piring'}
        if entity_name in skip_names:
            m = _re.search(r'(?:jumlah\s+)?(?:kelereng|buah|ekor|orang)\s+(\w+)', ql)
            if m:
                entity_name = m.group(1)
            else:
                return None

        # Split text into per-entity sentences
        sentences = _re.split(r'[.!?]\s*', narrative_text)
        entity_sentences = [s for s in sentences if entity_name in s.lower()]

        if not entity_sentences:
            return None

        # Sum all numbers in the entity's sentences
        total = 0
        numbers_found = []
        for sent in entity_sentences:
            nums = _re.findall(r'\b(\d+\.?\d*)\b', sent)
            for num_str in nums:
                try:
                    val = float(num_str)
                    if val < 100000 and val == int(val):
                        total += int(val)
                        numbers_found.append(int(val))
                except ValueError:
                    pass

        if not numbers_found or len(numbers_found) < 2:
            return None  # Let existing handlers handle single numbers

        # v20: Safety check — don't sum if this looks like a MULTIPLY pattern
        # "8 kotak permen, setiap kotak berisi 24" → 8 × 24, NOT 8 + 24
        # Heuristic: if text has "setiap", "masing-masing", "per" + noun, it's multiply
        multiply_signals = ['setiap', 'masing-masing', 'tiap', 'per kotak', 'per buah',
                           'per orang', 'per ekor', 'per lembar', 'masing']
        entity_text = ' '.join(entity_sentences).lower()
        if any(sig in entity_text for sig in multiply_signals):
            return None  # Let operational schema handle multiply

        # Also check: if one number is much larger than the other,
        # it's likely a multiply scenario (e.g., 8 × 24)
        if len(numbers_found) >= 2:
            min_num = min(numbers_found)
            max_num = max(numbers_found)
            if max_num > min_num * 5 and min_num > 1:
                # Likely multiply: small number × large number
                return None

        return {
            'answer': total,
            'confidence': 0.80,
            'method': 'entity_number_sum',
            'explanation': f"Sum of {entity_name}'s numbers: {' + '.join(str(n) for n in numbers_found)} = {total}"
        }

    # @FLOW: IS_TEXT_QUESTION
    # @CALLS: GrammarParser.detect_question_roles
    # @MUTATES: none
    # @BEHAVIOR: Determines if a question is a text comprehension question
    #            (about who/what/why/how/when/where/message/feeling) vs a
    #            quantitative question ("berapa").
    def _is_text_comprehension_question(self, question: str) -> bool:
        """Check if question is a text comprehension question (not quantitative).
        Text comprehension: siapa, apa, mengapa, kenapa, bagaimana, kapan, di mana,
                            amanat, pesan, perasaan
        Quantitative: berapa (how much/many) — BUT some "berapa" questions are
                      actually asking about directly stated numbers (eksplisit),
                      not requiring computation.
        """
        q_lower = question.lower()

        try:
            from grammar.parser import GrammarParser
        except ImportError:
            from grammar.simple_parser import SimpleParser as GrammarParser
        parser = GrammarParser(self.self_core)
        q_roles = parser.detect_question_roles(question)

        # If any question role is detected, it's a text comprehension question
        text_roles = ['question_who', 'question_what', 'question_why', 'question_how',
                      'question_when', 'question_where', 'question_message', 'question_feeling']
        for role in text_roles:
            if role in q_roles:
                return True

        # Also check for common Indonesian text comprehension question words
        # v13: Added Kelas 5 Hard Mode question types
        # v14: Added Extreme Mode question types
        text_q_words = ['siapa', 'mengapa', 'kenapa', 'bagaimana', 'kapan',
                        'di mana', 'dimana', 'amanat', 'pesan', 'makna',
                        'perasaan', 'merasa', 'pelajaran', 'tujuan',
                        'disampaikan', 'dapat dipetik', 'dapat kita',
                        'ide pokok', 'gagasan utama', 'peribahasa', 'pepatah',
                        'majas', 'kiasan', 'personifikasi', 'perumpamaan',
                        'opini', 'fakta', 'pendapat', 'kesimpulan',
                        'persamaan', 'perbedaan', 'kelebihan', 'kekurangan',
                        'mendorong', 'motivasi', 'dorongan',
                        'kalimat mana', 'merupakan',
                        # Kelas 5 Hard Mode types
                        'sinonim', 'lawan kata', 'makna kata', 'bermakna',
                        'sikap', 'sifat', 'perilaku',
                        'langkah', 'cara membuat', 'cara mendaftar', 'ajakan',
                        'persuasif', 'persuasi',
                        'tokoh utama', 'latar', 'tema', 'unsur cerita',
                        'pernyataan', 'benar', 'salah',
                        'analogi', 'kesan',
                        'utama', 'dominan', 'akar masalah',
                        'termasuk majas', 'berdasarkan teks',
                        'pertama kali', 'berapa lama',
                        # Extreme Mode types
                        'tidak disebutkan', 'tidak dilakukan', 'tidak tepat',
                        'suasana', 'tergambar', 'tercipta', 'terasa',
                        'menyebabkan', 'proses apa',
                        'lebih cocok', 'sepi pembeli',
                        'bertanggung jawab', 'menunjukkan',
                        # v14: Tricky phrasing patterns
                        'sebenarnya ingin mengatakan', 'inti dari bacaan', 'keseluruhan membahas',
                        'ungkapan bijak', 'kata-kata bijak', 'kata bijak',
                        'gaya bahasa', 'ragam bahasa', 'benda mati seolah-olah',
                        'arti sama dengan', 'berlawanan arti',
                        'karakter apa', 'watak',
                        'urutan yang benar', 'harus dilakukan pertama',
                        'mengajak pembaca', 'undangan untuk bertindak',
                        'pelaku yang paling', 'peristiwa berlangsung',
                        'hikmah apa', 'mengajarkan nilai',
                        'perasaan apa yang muncul', 'diciptakan penulis',
                        'sama seperti hubungan', 'kalau x pakai',
                        'kalau tukang kayu', 'kalau tukang jahit',
                        'sesuai dengan isi teks', 'alasan x tetap',
                        'melebih-lebihkan', 'terlalu berlebihan',
                        'mengada-ada',
                        'hal apa yang sama', 'kelebihan dibandingkan',
                        'terangkai dari sebab',
                        'akhirnya menyebabkan', 'yang menyebabkan']
        for qw in text_q_words:
            if qw in q_lower:
                return True

        # Check for analogy patterns: A : B = C : D
        if ':' in q_lower and '=' in q_lower:
            return True

        # "berapa" questions: some are quantitative (math), some are eksplisit (text)
        # "pukul berapa" → text comprehension (asking about time directly stated)
        # "berapa jumlah X" where X is directly stated → could be either
        # "berapa total", "berapa harga", "berapa hasil" → quantitative (math)
        if 'berapa' in q_lower:
            # "pukul berapa" → text comprehension (time lookup)
            if 'pukul' in q_lower:
                return True
            # "berapa jumlah" alone without computation context → could be eksplisit
            # But we default to quantitative and let operational schema try first
            return False

        # Check for "apa" that is NOT "berapa"
        if 'apa' in q_lower:
            return True

        return False

    def _apply_learned_rules(self, text: str, context: dict) -> dict:
        """Apply learned rules"""
        if self.self_core is None:
            return {'answer': None, 'confidence': 0.0, 'method': 'no_rules'}

        for axiom_id, axiom in self.self_core.axioms.items():
            if axiom.get('predicate') in ['SUBTRACT', 'ADD', 'MULTIPLY', 'DIVIDE',
                                           'FRACTION_MULTIPLY', 'PERIMETER', 'TIME_DURATION']:
                if self._axiom_matches(axiom, text, context):
                    return {
                        'answer': self._apply_axiom(axiom, context),
                        'confidence': axiom.get('confidence', 0.5),
                        'method': 'learned_rule',
                        'axiom_id': axiom_id
                    }

        return {'answer': None, 'confidence': 0.0, 'method': 'no_matching_rule'}

    def _axiom_matches(self, axiom: dict, text: str, context: dict) -> bool:
        """Check if an axiom matches the current context — NOW ACTUALLY WORKS"""
        predicate = axiom.get('predicate', '')
        text_lower = text.lower()

        # Check if axiom's operation matches text context
        if predicate in ['SUBTRACT', 'ADD', 'MULTIPLY', 'DIVIDE',
                         'FRACTION_MULTIPLY', 'PERIMETER', 'TIME_DURATION']:
            # Check if text contains numbers and relevant role keywords
            numbers = re.findall(r'\d+', text)
            if not numbers:
                return False

            # Check subject/object overlap with text
            subject = axiom.get('subject', '').lower()
            obj = axiom.get('object', '').lower()
            if subject and any(w in text_lower for w in subject.split('_')):
                return True
            if obj and any(w in text_lower for w in obj.split('_')):
                return True

        return False

    def _apply_axiom(self, axiom: dict, context: dict) -> Optional[float]:
        """Apply an axiom to compute answer — NOW ACTUALLY WORKS"""
        predicate = axiom.get('predicate', '')
        values = axiom.get('values', context.get('values', []))

        if predicate == 'SUBTRACT' and len(values) >= 2:
            return max(values[:2]) - min(values[:2])
        elif predicate == 'ADD' and len(values) >= 2:
            return sum(values[:2])
        elif predicate == 'MULTIPLY' and len(values) >= 2:
            return values[0] * values[1]
        elif predicate == 'DIVIDE' and len(values) >= 2:
            divisor = min(values[:2])
            if divisor != 0:
                return max(values[:2]) / divisor
        elif predicate == 'FRACTION_MULTIPLY':
            fraction = axiom.get('fraction') or context.get('fraction')
            if fraction is not None and values:
                return values[0] * fraction

        return None

    def _compute_from_triplet(self, predicate: str, values: list, triplet: dict) -> Optional[float]:
        """Compute a numeric answer from a parsed triplet's predicate and values"""
        try:
            numeric_values = [float(v) for v in values if isinstance(v, (int, float))]
        except (ValueError, TypeError):
            return None

        if not numeric_values:
            return None

        if predicate == 'SUBTRACT' and len(numeric_values) >= 2:
            return max(numeric_values[:2]) - min(numeric_values[:2])
        elif predicate == 'ADD' and len(numeric_values) >= 2:
            return sum(numeric_values[:2])
        elif predicate == 'MULTIPLY' and len(numeric_values) >= 2:
            return numeric_values[0] * numeric_values[1]
        elif predicate == 'DIVIDE' and len(numeric_values) >= 2:
            divisor = min(numeric_values[:2])
            if divisor != 0:
                return max(numeric_values[:2]) / divisor
        elif predicate == 'FRACTION_MULTIPLY':
            fraction = triplet.get('fraction')
            if fraction is not None and numeric_values:
                return numeric_values[0] * fraction

        return None

    def _apply_transe(self, text: str, context: dict) -> dict:
        """Apply TransE geometric reasoning — now ACTUALLY wired in"""
        if self.self_core is None or len(self.self_core.axioms) < 2:
            return {'answer': None, 'confidence': 0.0, 'method': 'insufficient_data'}

        # Feed axioms to TransE
        for axiom_id, axiom in self.self_core.axioms.items():
            h = axiom.get('subject', '')
            r = axiom.get('predicate', '')
            t = axiom.get('object', '')
            if h and r and t:
                self.transe.add_triplet(h, r, t)

        # Try to match text against known triplets and score
        # If we can extract a partial triplet from text, use TransE to predict
        try:
            from grammar.parser import GrammarParser
        except ImportError:
            from grammar.simple_parser import SimpleParser as GrammarParser
        parser = GrammarParser(self.self_core)
        parsed = parser.parse(text)
        triplets = parsed.get('triplets', [])

        if triplets:
            best_score = -float('inf')
            best_answer = None
            for triplet in triplets:
                h = triplet.get('subject', '')
                r = triplet.get('predicate', '')
                t = triplet.get('object', '')

                if h and r and t:
                    score = self.transe.score_triplet(h, r, t)
                    if score > best_score:
                        best_score = score
                        # If the triplet has computed values, derive answer from them
                        values = triplet.get('values', [])
                        if values:
                            best_answer = self._compute_from_triplet(r, values, triplet)
                        else:
                            best_answer = t

                # Try predicting tail if we have h and r
                if h and r and not t:
                    predictions = self.transe.predict_tail(h, r, top_k=3)
                    if predictions:
                        best_answer = predictions[0][0]
                        best_score = predictions[0][1]

            if best_answer is not None:
                # Normalize confidence from TransE score
                confidence = min(0.8, max(0.1, (best_score + 5) / 10))  # Rough normalization
                return {'answer': best_answer, 'confidence': confidence, 'method': 'transe_geometric'}

        return {'answer': None, 'confidence': 0.05, 'method': 'transe_no_match'}

    def _apply_operational(self, text: str, context: dict) -> dict:
        """Apply operational schema - primary strategy for math problems
        v10.1: Added multi-step detection for patterns like "masing-masing X, bayar Y, kembalian?"
        """
        try:
            from grammar.parser import GrammarParser
        except ImportError:
            from grammar.simple_parser import SimpleParser as GrammarParser
        parser = GrammarParser(self.self_core)
        parsed = parser.parse(text)

        roles = parsed.get('roles', {})
        numbers = parsed.get('numbers', [])
        operation_hint = parsed.get('operation_hint', 'UNKNOWN')
        fractions = parsed.get('fractions', [])
        facts = parsed.get('extracted_facts', [])

        # Detect multi-step patterns: MULTIPLY then SUBTRACT
        # e.g., "4 buku masing-masing Rp8.000. bayar Rp50.000. kembalian?"
        if facts and len(facts) >= 2:
            multi_result = self._try_multi_step_from_text(facts, roles, numbers)
            if multi_result is not None:
                return multi_result

        if len(numbers) < 2 and not fractions:
            # Single number with fraction
            if fractions and len(numbers) >= 1:
                operation = 'FRACTION_MULTIPLY'
                fraction_val = fractions[0]['value']
                result = self.operational.compute(operation, numbers, roles, fraction_val)
                if result is not None:
                    return {
                        'answer': result,
                        'confidence': 0.7,
                        'method': f'operational_{operation}'
                    }
            # Single number with SQUARE_AREA (sisi × sisi)
            if operation_hint == 'SQUARE_AREA' and len(numbers) >= 1:
                result = self.operational.compute('SQUARE_AREA', numbers, roles)
                if result is not None:
                    return {
                        'answer': result,
                        'confidence': 0.7,
                        'method': 'operational_SQUARE_AREA'
                    }
            return {'answer': None, 'confidence': 0.0, 'method': 'insufficient_numbers'}

        # Use operational schema to infer and compute
        operation = self.operational.infer_operation(roles, numbers, text, fractions)

        if operation == 'UNKNOWN':
            operation = operation_hint

        if operation != 'UNKNOWN':
            fraction_val = fractions[0]['value'] if fractions else None
            result = self.operational.compute(operation, numbers, roles, fraction_val)
            if result is not None:
                if result < 0:
                    return {
                        'answer': result,
                        'confidence': 0.3,
                        'method': f'operational_{operation}',
                        'warning': 'negative_result'
                    }
                return {
                    'answer': result,
                    'confidence': 0.7,
                    'method': f'operational_{operation}'
                }

        return {'answer': None, 'confidence': 0.0, 'method': 'operational_unknown'}

    def _try_multi_step_from_text(self, facts: list, all_roles: dict, all_numbers: list) -> Optional[dict]:
        """Try multi-step derivation from text facts
        Detects patterns like:
        - MULTIPLY then SUBTRACT (kembalian: buy items then subtract from payment)
        - MULTIPLY then ADD (groups + additional)
        - SUBTRACT then ADD (sold some, added more)
        """
        # Build steps from facts
        steps = []
        for fact in facts:
            nums = fact.get('numbers', [])
            roles = fact.get('roles', {})
            op = fact.get('operation_hint', 'UNKNOWN')
            fracs = fact.get('fractions', [])

            # Special case: consumed + added in same fact → SUBTRACT then ADD
            if 'consumed' in roles and 'added' in roles and len(nums) >= 2:
                # The first number in the sentence with consumed role is consumed
                # The second number (or last) is added
                consumed_val = nums[0]['value']  # First number = consumed (terjual 35)
                added_val = nums[-1]['value']     # Last number = added (ditambah 28)
                # Actually, find initial from previous fact or all_numbers
                initial_val = None
                for prev_fact in facts[:facts.index(fact)]:
                    prev_nums = prev_fact.get('numbers', [])
                    if prev_nums:
                        initial_val = prev_nums[0]['value']
                        break
                if initial_val is None and all_numbers:
                    initial_val = all_numbers[0]['value'] if isinstance(all_numbers[0], dict) else all_numbers[0]

                if initial_val:
                    # Build: initial - consumed, then prev + added
                    steps.append({
                        'operation': 'SUBTRACT',
                        'values': [initial_val, consumed_val],
                        'roles': {},
                    })
                    steps.append({
                        'operation': 'ADD',
                        'values': ['prev', added_val],
                        'roles': {},
                    })
                continue

            if op != 'UNKNOWN' and len(nums) >= 2:
                fraction_val = fracs[0]['value'] if fracs else None
                step = {
                    'operation': op,
                    'values': [n['value'] for n in nums[:2]],
                    'roles': roles,
                    'fraction': fraction_val,
                }
                steps.append(step)
            elif len(nums) >= 2:
                # Infer operation from roles
                op = self.operational.infer_operation(roles, nums, '', fracs)
                if op != 'UNKNOWN':
                    fraction_val = fracs[0]['value'] if fracs else None
                    step = {
                        'operation': op,
                        'values': [n['value'] for n in nums[:2]],
                        'roles': roles,
                        'fraction': fraction_val,
                    }
                    steps.append(step)
            elif 'consumed' in roles and 'added' in roles and len(nums) >= 2:
                # Split: first SUBTRACT consumed, then ADD added
                # Find the initial number from previous fact
                consumed_val = None
                added_val = None
                for n in nums:
                    # Heuristic: consumed values tend to be smaller
                    if consumed_val is None:
                        consumed_val = n['value']
                    else:
                        added_val = n['value']
                if consumed_val and added_val:
                    # Need to chain: prev - consumed + added
                    steps.append({
                        'operation': 'SUBTRACT',
                        'values': ['prev', consumed_val],
                        'roles': {},
                    })
                    steps.append({
                        'operation': 'ADD',
                        'values': ['prev', added_val],
                        'roles': {},
                    })

        # If we have a single fact with initial + one number, and the second fact
        # has consumed role, create SUBTRACT step from initial
        if not steps and len(facts) >= 2:
            first_nums = facts[0].get('numbers', [])
            second_roles = facts[1].get('roles', {})
            second_nums = facts[1].get('numbers', [])

            if first_nums and second_nums and 'consumed' in second_roles:
                # Step 1: initial - consumed
                initial = first_nums[0]['value']
                consumed = second_nums[0]['value']
                steps.append({
                    'operation': 'SUBTRACT',
                    'values': [initial, consumed],
                    'roles': {},
                })
                # If there's also an added role
                if 'added' in second_roles and len(second_nums) >= 2:
                    added = second_nums[1]['value']
                    steps.append({
                        'operation': 'ADD',
                        'values': ['prev', added],
                        'roles': {},
                    })

        # v18: Check if we need a final step (e.g., SUBTRACT for kembalian)
        # Fixed C-04: kembalian = payment - total_cost (was using wrong order)
        # Detect "kembalian" in question context more robustly
        text_lower_for_kembalian = str(all_roles).lower() if all_roles else ''
        has_kembalian = (
            any('kembalian' in kw for kw in all_roles.get('price', [])) or
            'kembalian' in text_lower_for_kembalian or
            any('kembalian' in str(n.get('raw', '')).lower() for n in all_numbers if isinstance(n, dict))
        )

        if has_kembalian:
            # Find the payment amount — heuristic: the LARGEST number is likely the payment
            # because payment is typically larger than per-item price
            payment = None
            all_num_values = []
            for num in all_numbers:
                if isinstance(num, dict):
                    all_num_values.append(num['value'])
                elif isinstance(num, (int, float)):
                    all_num_values.append(num)

            if all_num_values:
                payment = max(all_num_values)

            if payment and steps:
                # Add SUBTRACT step: payment - total_cost
                # Direction is critical: payment (larger) - prev (total cost)
                steps.append({
                    'operation': 'SUBTRACT',
                    'values': [payment, 'prev'],
                    'roles': {},
                })
            elif payment and not steps:
                # No steps yet but we have kembalian — try to build
                # MULTIPLY step first (quantity × price), then SUBTRACT
                # This handles "3 buku Rp5000, bayar Rp20000, kembalian?"
                num_values_sorted = sorted(all_num_values, reverse=True)
                if len(num_values_sorted) >= 3:
                    # payment = largest, price = second largest, qty = smallest
                    # Actually: need to figure out which is qty vs price
                    # qty is typically small (< 100), price is medium (100-100000)
                    qty_candidates = [v for v in num_values_sorted if v < 100]
                    price_candidates = [v for v in num_values_sorted if v >= 100 and v < payment]

                    if qty_candidates and price_candidates:
                        qty = qty_candidates[0]
                        price = price_candidates[0]
                        steps.append({
                            'operation': 'MULTIPLY',
                            'values': [qty, price],
                            'roles': {},
                        })
                        steps.append({
                            'operation': 'SUBTRACT',
                            'values': [payment, 'prev'],
                            'roles': {},
                        })

            # v18: Also handle case where steps has 1 step (just MULTIPLY)
            # and we need to add SUBTRACT for kembalian
            if len(steps) == 1 and steps[0]['operation'] == 'MULTIPLY' and payment:
                # The MULTIPLY result is the total cost, subtract from payment
                already_has_subtract = any(s['operation'] == 'SUBTRACT' for s in steps)
                if not already_has_subtract:
                    steps.append({
                        'operation': 'SUBTRACT',
                        'values': [payment, 'prev'],
                        'roles': {},
                    })

        # Check for "datang lagi" or "ditambah" pattern → ADD prev result
        if 'added' in all_roles and steps:
            # Find numbers not already used in steps
            used_values = set()
            for step in steps:
                for v in step.get('values', []):
                    if v != 'prev' and not isinstance(v, dict):
                        used_values.add(v)
            added_num = None
            for num in all_numbers:
                if isinstance(num, dict) and num['value'] not in used_values:
                    added_num = num['value']
                    break
            if added_num:
                steps.append({
                    'operation': 'ADD',
                    'values': ['prev', added_num],
                    'roles': {},
                })

        if len(steps) >= 2:
            result = self.operational.compute_multi_step(steps)
            if result is not None:
                return {
                    'answer': result,
                    'confidence': 0.6,
                    'method': 'operational_multi_step',
                }

        return None

    def _reconcile(self, results: dict) -> dict:
        """Reconcile results from 3 strategies using confidence"""
        alpha = 0.5
        if self.self_core:
            alpha = self.self_core.confidence_alpha

        best_answer = None
        best_confidence = 0.0
        best_method = 'none'
        all_answers = []

        for strategy, result in results.items():
            answer = result.get('answer')
            confidence = result.get('confidence', 0.0)
            method = result.get('method', strategy)

            if answer is not None:
                all_answers.append({'answer': answer, 'confidence': confidence, 'method': method, 'strategy': strategy})

                if strategy == 'rule':
                    weighted_conf = alpha * confidence
                else:
                    weighted_conf = (1 - alpha) * confidence

                if weighted_conf > best_confidence:
                    best_confidence = weighted_conf
                    best_answer = answer
                    best_method = method

        if len(all_answers) >= 2:
            answer_counts = defaultdict(int)
            for a in all_answers:
                answer_counts[str(a['answer'])] += 1
            max_agreement = max(answer_counts.values())
            if max_agreement >= 2:
                best_confidence = min(1.0, best_confidence * 1.3)

        method_parts = []
        for strategy, result in results.items():
            if result.get('answer') is not None:
                method_parts.append(f"{strategy}={result['answer']}({result.get('confidence', 0.0):.2f})")
        explanation = " | ".join(method_parts) if method_parts else "no_derivation"

        return {
            'answer': best_answer,
            'confidence': best_confidence,
            'method': best_method,
            'explanation': explanation,
            'all_results': results
        }

    def explain_derivation(self, result: dict) -> str:
        """Explain how the derivation was made - Bug #3 fix: None guard"""
        if result is None:
            return "No derivation result available."

        answer = result.get('answer')
        method = result.get('method', 'unknown')
        confidence = result.get('confidence', 0.0)

        if answer is None:
            return f"Could not derive an answer using {method}."

        return f"Derived {answer} using {method} with confidence {confidence:.2f}"
