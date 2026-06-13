#!/usr/bin/env python3
# @WHO:   self-ai/src/curiosity/answer_cli.py
# @WHAT:  CLI for answering curiosity questions
# @PART:  self-ai/curiosity
# @ENTRY: main()

"""CLI for answering curiosity questions
Usage: python -m curiosity.answer_cli
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from core.self import SelfCore
from curiosity.loop import CuriosityLoop


def main():
    # Load state
    core = SelfCore()
    core.load()

    loop = CuriosityLoop(core)
    loop.export_queue()

    unanswered = loop.get_unanswered()
    if not unanswered:
        print("✅ Tidak ada pertanyaan curiosity yang belum dijawab!")
        return

    print(f"📋 {len(unanswered)} pertanyaan curiosity menunggu jawaban:")
    print()

    for i, item in enumerate(unanswered):
        print(f"  [{i+1}/{len(unanswered)}] {item['question']}")

        # Try auto-answer first
        auto = loop.try_auto_answer(item['question'])
        if auto:
            print(f"  🤖 Auto-answer: {auto}")
            confirm = input("  Terima jawaban ini? (y/n/s=skip): ").strip().lower()
            if confirm == 'y':
                loop.answer_question(item['question'], auto, source='llm_confirmed')
                print("  ✅ Disimpan!")
            elif confirm == 's':
                continue
            else:
                answer = input("  Jawaban Anda: ").strip()
                if answer:
                    loop.answer_question(item['question'], answer, source='user')
                    print("  ✅ Disimpan!")
        else:
            answer = input("  Jawaban Anda (atau 'skip'): ").strip()
            if answer.lower() == 'skip':
                continue
            if answer:
                loop.answer_question(item['question'], answer, source='user')
                print("  ✅ Disimpan!")

        print()

    # Save state
    core.save()
    print("Selesai! Jawaban telah diajarkan ke sistem.")


if __name__ == '__main__':
    main()
