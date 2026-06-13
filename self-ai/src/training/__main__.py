# @WHO:   self-ai/src/training/__main__.py
# @WHAT:  CLI entry point for TrainingAgent — python -m self_ai.training
# @PART:  self-ai/training
# @ENTRY: main()

"""SELF Training Agent CLI — dedicated training session for teaching SELF.

Entry point:
    python -m self_ai.training

Commands:
    (q)uestion  — Ask SELF a question, see the answer
    (c)orrect   — Correct SELF's last answer (generates reasoning first)
    (b)enchmark — Run benchmark, measure accuracy
    (e)xport    — Export session to docs/training_sessions/
    (x)exit     — End session (auto-export if there's data)

Design principle: Intent harus eksplisit dari user, bukan ditebak sistem.
User must explicitly trigger each action — no auto-detection.
"""

import os
import sys
import logging

# Setup path agar training package bisa di-import
_this_dir = os.path.dirname(os.path.abspath(__file__))
_src_dir = os.path.join(_this_dir, '..')
if _src_dir not in sys.path:
    sys.path.insert(0, _src_dir)
if _this_dir not in sys.path:
    sys.path.insert(0, _this_dir)

os.environ['TOKENIZERS_PARALLELISM'] = '0'

from training.training_agent import TrainingAgent

logging.basicConfig(
    level=logging.WARNING,
    format='%(levelname)s: %(message)s'
)


# @FLOW: TRAINING_CLI
# @CALLS: TrainingAgent.run(), TrainingAgent.correct(), TrainingAgent.confirm_correction(), TrainingAgent.benchmark(), TrainingAgent.export_session()
# @MUTATES: learned_patterns.json (via teach_from_correction), docs/training_sessions/ (via export)
# @BEHAVIOR: Interactive loop — user must explicitly type 'c' to correct, never auto-corrects.
def main():
    """Run the TrainingAgent CLI session."""
    print()
    print("=" * 60)
    print("  SELF Training Agent v1")
    print("  Explicit-intent correction & learning")
    print("=" * 60)
    print()

    # Initialize agent
    print("Initializing engine...")
    agent = TrainingAgent()

    if agent.engine is None:
        print("FATAL: Engine failed to initialize. Cannot continue.")
        sys.exit(1)

    print(f"  Engine: OK")
    print(f"  TextComprehension: {'OK' if agent.tc else 'MISSING'}")
    print(f"  Patterns loaded: {len(agent.tc.learned_patterns) if agent.tc else 0}")

    # Check model availability
    try:
        from derivation.model_registry import get_shared_embedding_model, get_shared_qwen
        emb = get_shared_embedding_model()
        qwen, _ = get_shared_qwen()
        print(f"  Embedding model: {'LOADED' if emb else 'NOT LOADED'}")
        print(f"  Qwen3 model: {'LOADED' if qwen else 'NOT LOADED'}")
    except Exception as e:
        print(f"  Model check: {e}")

    print()
    print("Commands:")
    print("  (q)uestion  — Ask SELF a question")
    print("  (c)orrect   — Correct SELF's last answer")
    print("  (b)enchmark — Run benchmark")
    print("  (e)xport    — Export session to markdown")
    print("  (x)exit     — End session")
    print()

    # Session loop
    has_data = False

    while True:
        try:
            cmd = input("training> ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if not cmd:
            continue

        # ── QUESTION ──
        if cmd in ('q', 'question'):
            context = input("  Context text (optional, press Enter to skip): ").strip()
            if not context:
                context = input("  Question: ").strip()
                if not context:
                    continue
                question = context
                context = question  # Use question as context if no separate context
            else:
                question = input("  Question: ").strip()
                if not question:
                    continue

            result = agent.run(question, context)

            if 'error' in result:
                print(f"  ERROR: {result['error']}")
                continue

            answer = result.get('answer', '(no answer)')
            confidence = result.get('confidence', 0)
            method = result.get('method', 'unknown')

            print(f"\n  SELF: {answer} (confidence: {confidence:.2f}, method: {method})\n")
            has_data = True

        # ── CORRECT ──
        elif cmd in ('c', 'correct'):
            if agent._last_result is None:
                print("  No previous question. Ask one first with (q).")
                continue

            correct_answer = input("  Correct answer: ").strip()
            if not correct_answer:
                print("  (skipped — empty answer)")
                continue

            # Generate reasoning — does NOT teach yet
            result = agent.correct(correct_answer)

            if 'error' in result:
                print(f"  ERROR: {result['error']}")
                continue

            reasoning = result.get('reasoning', '')
            print(f'  Reasoning: "{reasoning}"')
            print()

            # Quality gate — user must confirm
            confirm = input("  Confirm this correction? (y/edit/n): ").strip().lower()

            if confirm == 'y':
                # Confirm → teach_from_correction() called
                confirm_result = agent.confirm_correction()
                if confirm_result.get('confirmed'):
                    print(f"  Pattern tersimpan (key: {confirm_result.get('pattern_key', '?')[:60]})")
                else:
                    print(f"  ERROR: {confirm_result.get('error', 'unknown')}")

            elif confirm == 'edit':
                edited = input("  Edited reasoning: ").strip()
                if edited:
                    confirm_result = agent.confirm_correction(edited_reasoning=edited)
                    if confirm_result.get('confirmed'):
                        print(f"  Pattern tersimpan (key: {confirm_result.get('pattern_key', '?')[:60]})")
                    else:
                        print(f"  ERROR: {confirm_result.get('error', 'unknown')}")
                else:
                    agent.reject_correction()
                    print("  (correction rejected — empty edit)")

            else:
                # Reject
                agent.reject_correction()
                print("  (correction rejected)")

        # ── BENCHMARK ──
        elif cmd in ('b', 'benchmark'):
            # Run before benchmark if not done yet
            if agent.session.benchmark_before is None:
                print("  Running BEFORE benchmark...")
                before = agent.benchmark()
                if 'error' in before:
                    print(f"  ERROR: {before['error']}")
                    continue
                agent.session.set_benchmark('before', before)
                print(f"  Before: {before['correct']}/{before['total']} ({before['accuracy']:.1%})")
            else:
                print("  BEFORE benchmark already recorded.")

            # Run after benchmark
            print("  Running AFTER benchmark...")
            after = agent.benchmark()
            if 'error' in after:
                print(f"  ERROR: {after['error']}")
                continue
            agent.session.set_benchmark('after', after)
            print(f"  After:  {after['correct']}/{after['total']} ({after['accuracy']:.1%})")

            # Show delta
            if agent.session.benchmark_before:
                before_acc = agent.session.benchmark_before['accuracy']
                after_acc = after['accuracy']
                delta = after_acc - before_acc
                sign = '+' if delta >= 0 else ''
                print(f"  Delta:  {sign}{delta:.1%}")

                # Per-type breakdown
                before_per = agent.session.benchmark_before.get('per_type', {})
                after_per = after.get('per_type', {})
                all_domains = sorted(set(list(before_per.keys()) + list(after_per.keys())))
                if all_domains:
                    print("  Domain breakdown:")
                    for domain in all_domains:
                        b = before_per.get(domain, {}).get('accuracy', 0)
                        a = after_per.get(domain, {}).get('accuracy', 0)
                        d = a - b
                        s = '+' if d >= 0 else ''
                        print(f"    {domain}: {b:.0%} → {a:.0%} ({s}{d:.0%})")

            has_data = True

        # ── EXPORT ──
        elif cmd in ('e', 'export'):
            filepath = agent.export_session()
            print(f"  Session exported to: {filepath}")
            has_data = False  # Reset — data sudah di-export

        # ── EXIT ──
        elif cmd in ('x', 'exit', 'quit'):
            # Auto-export if there's unexported data
            if has_data and (agent.session.questions or agent.session.corrections):
                print("  Auto-exporting session...")
                filepath = agent.export_session()
                print(f"  Session exported to: {filepath}")
            break

        else:
            print(f"  Unknown command: {cmd}")
            print("  Commands: (q)uestion (c)orrect (b)enchmark (e)xport (x)exit")

    print()
    print("Session ended.")


if __name__ == '__main__':
    main()
