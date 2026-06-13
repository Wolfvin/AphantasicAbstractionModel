# @WHO:   self-ai/src/derivation/operational.py
# @WHAT:  Operational schema with fraction, multi-step, text comprehension, perimeter, time
# @PART:  self-ai/derivation
# @ENTRY: OperationalSchema.infer_operation(), OperationalSchema.compute(),
#         OperationalSchema.compute_multi_step(), OperationalSchema.compute_from_facts()

"""Operational Schema Discovery
SUBTRACT/ADD/MULTIPLY/DIVIDE/FRACTION schemas emerge from quantity observation patterns.
CRITICAL FIX: Role-based direction detection, NOT verb-based.
- consumed + remaining → SUBTRACT direction
- initial + added → ADD direction (MEMBERI = giving = ADD, not SUBTRACT)
v10.1: PERIMETER (2×(p+l)), TIME_DURATION (end-start in minutes),
       FRACTION_MULTIPLY, multi-step computation, text comprehension,
       price computation, "dibagi"/"dikali" keyword support.
"""
from typing import Optional, Dict, Any, List


class OperationalSchema:
    """Discovers and applies operational schemas from quantity patterns"""

    def __init__(self, self_core=None):
        self.self_core = self_core
        self.schemas = {
            'SUBTRACT': {'direction': 'consumed_from_initial', 'roles': ['initial', 'consumed', 'remaining']},
            'ADD': {'direction': 'added_to_initial', 'roles': ['initial', 'added', 'total']},
            'MULTIPLY': {'direction': 'each_times_count', 'roles': ['each', 'count', 'total']},
            'DIVIDE': {'direction': 'total_by_groups', 'roles': ['total', 'group_count', 'per_group']},
            'FRACTION_MULTIPLY': {'direction': 'fraction_of_whole', 'roles': ['whole', 'fraction', 'result']},
            'PERIMETER': {'direction': '2_times_length_plus_width', 'roles': ['length', 'width']},
            'TIME_DURATION': {'direction': 'end_minus_start', 'roles': ['start', 'end']},
            'SQUARE_AREA': {'direction': 'side_times_side', 'roles': ['side']},
        }

    # @FLOW: OPS_INFER
    # @CALLS: _infer_from_pattern
    # @MUTATES: none
    # @BEHAVIOR: Infers operation from roles, NOT verbs. Priority: divide_keyword >
    #            multiply_keyword > fraction > consumed/remaining > etc.
    def infer_operation(self, roles: dict, numbers: list, text: str = '', fractions: list = None) -> str:
        """Infer operation from roles - THE CRITICAL FIX"""
        fractions = fractions or []

        # DIVIDE keyword: "dibagi", "dibagikan"
        if 'divide_keyword' in roles:
            return 'DIVIDE'

        # MULTIPLY keyword: "dikali", "kali"
        if 'multiply_keyword' in roles:
            return 'MULTIPLY'

        # FRACTION: "setengah dari 24" → fraction * number
        if fractions and 'fraction' in roles:
            return 'FRACTION_MULTIPLY'

        # SUBTRACT: consumed from initial gives remaining
        if 'consumed' in roles and ('remaining' in roles or 'initial' in roles):
            if 'added' in roles:
                return 'ADD'
            return 'SUBTRACT'

        # ADD: initial + added = total (MEMBERI fix!)
        if 'initial' in roles and 'added' in roles:
            return 'ADD'
        if 'added' in roles:
            return 'ADD'

        # DIFFERENCE: selisih → SUBTRACT
        if 'difference' in roles:
            return 'SUBTRACT'

        # MULTIPLY: each * count = total
        if 'shared_each' in roles:
            return 'MULTIPLY'

        # Price * quantity
        if 'price' in roles:
            return 'MULTIPLY'

        # Perimeter/Area
        if 'perimeter_area' in roles:
            roles_str = str(roles.get('perimeter_area', []))
            if 'keliling' in roles_str or 'kelilingnya' in roles_str:
                return 'PERIMETER'
            elif 'luas' in roles_str or 'luasnya' in roles_str:
                return 'SQUARE_AREA'
            elif 'sisi' in roles_str:
                return 'SQUARE_AREA'
            return 'MULTIPLY'

        # Time duration
        if 'time_duration' in roles:
            return 'TIME_DURATION'

        # DIVIDE: total / groups = per_group
        if 'total' in roles and 'group_count' in roles:
            return 'DIVIDE'
        if 'total' in roles and 'shared_each' in roles:
            return 'DIVIDE'

        # Fallback
        return self._infer_from_pattern(numbers, text)

    def _infer_from_pattern(self, numbers: list, text: str = '') -> str:
        """Fallback: infer from number relationships"""
        if len(numbers) < 2:
            return 'UNKNOWN'
        values = [n['value'] if isinstance(n, dict) else n for n in numbers]
        if len(values) >= 2 and min(values) > 1 and max(values) / min(values) > 5:
            return 'MULTIPLY'
        return 'UNKNOWN'

    def compute(self, operation: str, numbers: list, roles: dict = None, fraction: float = None) -> Optional[float]:
        """Compute the result of an operation"""
        values = [n['value'] if isinstance(n, dict) else n for n in numbers]

        if operation == 'SUBTRACT':
            return self._compute_subtract(values, roles)
        elif operation == 'ADD':
            return self._compute_add(values, roles)
        elif operation == 'MULTIPLY':
            return self._compute_multiply(values, roles)
        elif operation == 'DIVIDE':
            return self._compute_divide(values, roles)
        elif operation == 'FRACTION_MULTIPLY':
            return self._compute_fraction(values, roles, fraction)
        elif operation == 'PERIMETER':
            return self._compute_perimeter(values, roles)
        elif operation == 'TIME_DURATION':
            return self._compute_time_duration(numbers, roles)
        elif operation == 'SQUARE_AREA':
            return self._compute_square_area(values, roles)

        return None

    def _compute_subtract(self, values: list, roles: dict = None) -> Optional[float]:
        """SUBTRACT: initial - consumed = remaining, or larger - smaller = difference"""
        if len(values) < 2:
            return None
        initial = max(values[:2])
        consumed = min(values[:2])
        return initial - consumed

    def _compute_add(self, values: list, roles: dict = None) -> Optional[float]:
        """ADD: initial + added = total"""
        if len(values) < 2:
            return None
        return sum(values)

    def _compute_multiply(self, values: list, roles: dict = None) -> Optional[float]:
        """MULTIPLY: each * count = total, price * quantity, sisi * sisi (area)"""
        if len(values) < 2:
            return None
        return values[0] * values[1]

    def _compute_divide(self, values: list, roles: dict = None) -> Optional[float]:
        """DIVIDE: total / groups = per_group"""
        if len(values) < 2:
            return None
        total = max(values[:2])
        groups = min(values[:2])
        if groups == 0:
            return None
        return total / groups

    def _compute_fraction(self, values: list, roles: dict = None, fraction: float = None) -> Optional[float]:
        """FRACTION_MULTIPLY: whole * fraction = part"""
        if fraction is not None and len(values) >= 1:
            return values[0] * fraction
        if len(values) >= 2:
            return values[0] * values[1]
        return None

    def _compute_perimeter(self, values: list, roles: dict = None) -> Optional[float]:
        """PERIMETER: 2 × (panjang + lebar) for persegi panjang"""
        if len(values) < 2:
            return None
        return 2 * (values[0] + values[1])

    def _compute_time_duration(self, numbers: list, roles: dict = None) -> Optional[float]:
        """TIME_DURATION: end_time - start_time in minutes"""
        time_nums = [n for n in numbers if isinstance(n, dict) and n.get('is_time')]
        if len(time_nums) >= 2:
            end = time_nums[-1]['value']
            start = time_nums[0]['value']
            return end - start
        values = [n['value'] if isinstance(n, dict) else n for n in numbers]
        if len(values) >= 2:
            return max(values) - min(values)
        return None

    def _compute_square_area(self, values: list, roles: dict = None) -> Optional[float]:
        """SQUARE_AREA: sisi × sisi (square area)
        If only one value is given, square it.
        If two values are given, multiply them (rectangle area).
        """
        if len(values) >= 2:
            return values[0] * values[1]
        elif len(values) == 1:
            return values[0] * values[0]
        return None

    # @FLOW: OPS_MULTI_STEP
    # @CALLS: compute for each step
    # @MUTATES: none
    # @BEHAVIOR: Executes a chain of operations sequentially. Each step's output
    #            becomes available as 'prev' for subsequent steps.
    def compute_multi_step(self, steps: list) -> Optional[float]:
        """Compute multi-step operations - Grade 4 feature"""
        prev_result = None
        for step in steps:
            operation = step.get('operation', 'UNKNOWN')
            values = list(step.get('values', []))
            roles = step.get('roles', {})
            fraction = step.get('fraction', None)

            resolved_values = []
            for v in values:
                if v == 'prev':
                    if prev_result is None:
                        return None
                    resolved_values.append({'value': prev_result, 'raw': str(prev_result), 'position': 0})
                else:
                    resolved_values.append({'value': v, 'raw': str(v), 'position': 0})

            result = self.compute(operation, resolved_values, roles, fraction)
            if result is None:
                return None
            prev_result = result

        return prev_result

    # @FLOW: OPS_COMPUTE_FROM_FACTS
    # @CALLS: GrammarParser.parse(), infer_operation(), compute(), compute_multi_step(),
    #         _try_multi_step_from_facts(), _try_fraction_from_facts()
    # @MUTATES: none
    # @BEHAVIOR: Computes answer from extracted facts + question. Merges all numbers/roles
    #            from narrative facts with question context. Special handling for:
    #            - Fraction across facts: "48 kelereng. 1/4 bagian" → 48 × 0.25
    #            - Multi-step: "4 buku Rp8.000. Bayar Rp50.000" → 50000 - (4×8000)
    #            - Single-number operations: sisi * sisi for area
    def compute_from_facts(self, facts: list, question: str) -> Optional[float]:
        """Compute answer from extracted facts + question"""
        try:
            from grammar.parser import GrammarParser
            parser = GrammarParser(self.self_core)
        except ImportError:
            from grammar.simple_parser import SimpleParser
            parser = SimpleParser(self.self_core)

        # Parse the question
        q_parsed = parser.parse(question)
        q_numbers = q_parsed.get('numbers', [])
        q_roles = q_parsed.get('roles', {})
        q_fractions = q_parsed.get('fractions', [])
        q_op_hint = q_parsed.get('operation_hint', 'UNKNOWN')

        # If question has enough numbers on its own, answer directly
        if len(q_numbers) >= 2:
            operation = self.infer_operation(q_roles, q_numbers, question, q_fractions)
            if operation != 'UNKNOWN':
                fraction_val = q_fractions[0]['value'] if q_fractions else None
                result = self.compute(operation, q_numbers, q_roles, fraction_val)
                if result is not None:
                    return result

        # Special case: fraction from one fact + number from another
        fraction_result = self._try_fraction_from_facts(facts, question, q_parsed)
        if fraction_result is not None:
            return fraction_result

        # Gather all facts and merge
        all_numbers = list(q_numbers)
        all_roles = dict(q_roles)
        all_fractions = list(q_fractions)

        for fact in facts:
            for num in fact.get('numbers', []):
                # Avoid duplicates by value
                if num['value'] not in [n['value'] for n in all_numbers]:
                    all_numbers.append(num)
            for role_name, keywords in fact.get('roles', {}).items():
                if role_name not in all_roles:
                    all_roles[role_name] = keywords
                else:
                    for kw in keywords:
                        if kw not in all_roles[role_name]:
                            all_roles[role_name].append(kw)
            all_fractions.extend(fact.get('fractions', []))

        # Try direct computation with merged context
        operation = self.infer_operation(all_roles, all_numbers, question, all_fractions)
        if operation != 'UNKNOWN' and len(all_numbers) >= 2:
            fraction_val = all_fractions[0]['value'] if all_fractions else None
            result = self.compute(operation, all_numbers, all_roles, fraction_val)
            if result is not None:
                return result

        # Try multi-step from facts
        return self._try_multi_step_from_facts(facts, question, q_parsed)

    def _try_fraction_from_facts(self, facts: list, question: str, q_parsed: dict) -> Optional[float]:
        """Try to compute fraction across facts
        e.g., Fact 1: "48 kelereng" (number=48, no fraction)
              Fact 2: "1/4 bagian" (fraction=0.25, no number)
              Question: "Berapa kelereng yang diberikan?"
              → 48 × 0.25 = 12
        """
        # Find the fact with the main number (no fraction)
        main_number = None
        fraction_val = None
        all_roles = dict(q_parsed.get('roles', {}))

        for fact in facts:
            nums = fact.get('numbers', [])
            fracs = fact.get('fractions', [])
            roles = fact.get('roles', {})

            # Collect roles from all facts
            for role_name, keywords in roles.items():
                if role_name not in all_roles:
                    all_roles[role_name] = keywords

            if fracs and not fraction_val:
                fraction_val = fracs[0]['value']

            if nums and main_number is None:
                # Prefer the fact that has an 'initial' role or larger number
                if 'initial' in roles or 'total' in roles:
                    main_number = max(n['value'] for n in nums)

        # If we found both a main number and a fraction, compute
        if main_number is not None and fraction_val is not None:
            return main_number * fraction_val

        # Also check if question has a number
        q_nums = q_parsed.get('numbers', [])
        if fraction_val is not None and q_nums:
            return q_nums[0]['value'] * fraction_val

        return None

    def _try_multi_step_from_facts(self, facts: list, question: str, q_parsed: dict) -> Optional[float]:
        """Try to compute multi-step answer from facts
        Strategy: Build steps from facts that have operations, then add
        question step that references previous result.
        """
        steps = []

        for fact in facts:
            nums = fact.get('numbers', [])
            roles = fact.get('roles', {})
            op = fact.get('operation_hint', 'UNKNOWN')
            fracs = fact.get('fractions', [])

            if op != 'UNKNOWN' and len(nums) >= 2:
                fraction_val = fracs[0]['value'] if fracs else None
                step = {
                    'operation': op,
                    'values': [n['value'] for n in nums[:2]],
                    'roles': roles,
                    'fraction': fraction_val,
                }
                steps.append(step)

        # Add question step
        q_nums = q_parsed.get('numbers', [])
        q_op = q_parsed.get('operation_hint', 'UNKNOWN')
        q_roles = q_parsed.get('roles', {})
        q_fracs = q_parsed.get('fractions', [])

        if q_op != 'UNKNOWN' and len(q_nums) >= 1 and steps:
            step = {
                'operation': q_op,
                'values': ['prev', q_nums[0]['value']],
                'roles': q_roles,
                'fraction': q_fracs[0]['value'] if q_fracs else None,
            }
            steps.append(step)
        elif q_op != 'UNKNOWN' and len(q_nums) >= 2:
            step = {
                'operation': q_op,
                'values': [n['value'] for n in q_nums[:2]],
                'roles': q_roles,
                'fraction': q_fracs[0]['value'] if q_fracs else None,
            }
            steps.append(step)

        if len(steps) >= 1:
            return self.compute_multi_step(steps)

        return None
