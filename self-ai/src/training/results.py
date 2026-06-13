# @WHO:   self-ai/src/training/results.py
# @WHAT:  Export training session ke Markdown documentation
# @PART:  self-ai/training
# @ENTRY: export_session()

"""Training Session Results — auto-documentation ke Markdown.

Setiap training session di-export ke docs/training_sessions/ sebagai
bukti bahwa sistem bekerja. Export berisi: semua soal, jawaban, koreksi,
reasoning, accuracy delta.

Dokumentasi otomatis dari export adalah bukti bahwa sistem bekerja.
"""

import os
from datetime import datetime
from typing import Optional


def export_session(session, output_dir: str) -> str:
    """Export training session ke Markdown file.

    # @FLOW: TRAINING_EXPORT
    # @CALLS: none
    # @MUTATES: filesystem (writes markdown file)
    # @BEHAVIOR: Creates a markdown file with full session documentation.
    #            File name includes timestamp for uniqueness.
    #            Returns the path of the created file.

    Args:
        session: TrainingSession instance
        output_dir: Directory untuk simpan file (typically docs/training_sessions/)

    Returns:
        str: Path file yang dibuat
    """
    os.makedirs(output_dir, exist_ok=True)

    # Generate filename from session start time
    timestamp = session.started_at.strftime('%Y-%m-%d_%H-%M')
    filename = f"{timestamp}.md"
    filepath = os.path.join(output_dir, filename)

    # Build markdown content
    lines = []

    # Header
    lines.append(f"# Training Session {session.started_at.strftime('%Y-%m-%d %H:%M')}")
    lines.append("")

    # Summary
    summary = session.summary()
    lines.append("## Summary")
    lines.append("")
    lines.append(f"- Total questions: {summary['total_questions']}")
    lines.append(f"- Total corrections: {summary['total_corrections']}")

    if summary['accuracy_before'] is not None:
        lines.append(f"- Accuracy before: {summary['accuracy_before']:.1%}")
    else:
        lines.append("- Accuracy before: (not measured)")

    if summary['accuracy_after'] is not None:
        lines.append(f"- Accuracy after: {summary['accuracy_after']:.1%}")
    else:
        lines.append("- Accuracy after: (not measured)")

    if summary['accuracy_delta'] is not None:
        delta_pct = summary['accuracy_delta'] * 100
        sign = '+' if delta_pct >= 0 else ''
        lines.append(f"- Delta: {sign}{delta_pct:.1f}%")
    else:
        lines.append("- Delta: (not measured)")

    if summary['ended_at']:
        lines.append(f"- Session ended: {summary['ended_at']}")
    lines.append("")

    # Corrections Made
    if session.corrections:
        lines.append("## Corrections Made")
        lines.append("")
        for i, corr in enumerate(session.corrections, 1):
            lines.append(f"### Correction {i}")
            lines.append("")
            lines.append(f"- **Question:** {corr.question}")
            lines.append(f"- **Context:** {_truncate(corr.question, 120)}")
            lines.append(f"- **SELF answered:** {corr.wrong_answer}")
            lines.append(f"- **Correct answer:** {corr.correct_answer}")
            lines.append(f"- **Reasoning saved:** {corr.reasoning}")
            lines.append(f"- **Pattern key:** {corr.pattern_key}")
            lines.append("")
    else:
        lines.append("## Corrections Made")
        lines.append("")
        lines.append("No corrections made in this session.")
        lines.append("")

    # Benchmark Results
    has_benchmark = session.benchmark_before or session.benchmark_after
    if has_benchmark:
        lines.append("## Benchmark Results")
        lines.append("")

        if session.benchmark_before:
            lines.append("### Before")
            lines.append("")
            _write_benchmark_results(lines, session.benchmark_before)

        if session.benchmark_after:
            lines.append("### After")
            lines.append("")
            _write_benchmark_results(lines, session.benchmark_after)

        if session.benchmark_before and session.benchmark_after:
            lines.append("### Delta")
            lines.append("")
            _write_benchmark_delta(lines, session.benchmark_before,
                                   session.benchmark_after)

    # All Questions
    if session.questions:
        lines.append("## All Questions")
        lines.append("")
        lines.append("| # | Question | Answer | Confidence | Method |")
        lines.append("|---|----------|--------|------------|--------|")
        for i, q in enumerate(session.questions, 1):
            q_text = _truncate(q.question, 50)
            a_text = _truncate(str(q.answer) if q.answer else '-', 30)
            lines.append(
                f"| {i} | {q_text} | {a_text} | {q.confidence:.2f} | "
                f"{_truncate(q.method, 25)} |"
            )
        lines.append("")

    # Write file
    content = "\n".join(lines)
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(content)

    return filepath


def _write_benchmark_results(lines: list, results: dict):
    """Write benchmark results section."""
    total = results.get('total', 0)
    correct = results.get('correct', 0)
    accuracy = results.get('accuracy', 0)

    lines.append(f"- **Total:** {correct}/{total} correct ({accuracy:.1%})")
    lines.append("")

    per_type = results.get('per_type', {})
    if per_type:
        lines.append("| Domain | Correct | Total | Accuracy |")
        lines.append("|--------|---------|-------|----------|")
        for domain, stats in sorted(per_type.items()):
            d_correct = stats.get('correct', 0)
            d_total = stats.get('total', 0)
            d_acc = stats.get('accuracy', 0)
            lines.append(f"| {domain} | {d_correct} | {d_total} | {d_acc:.1%} |")
        lines.append("")


def _write_benchmark_delta(lines: list, before: dict, after: dict):
    """Write benchmark delta section."""
    before_per = before.get('per_type', {})
    after_per = after.get('per_type', {})

    all_domains = sorted(set(list(before_per.keys()) + list(after_per.keys())))

    if all_domains:
        lines.append("| Domain | Before | After | Delta |")
        lines.append("|--------|--------|-------|-------|")
        for domain in all_domains:
            b_acc = before_per.get(domain, {}).get('accuracy', 0)
            a_acc = after_per.get(domain, {}).get('accuracy', 0)
            delta = a_acc - b_acc
            sign = '+' if delta >= 0 else ''
            lines.append(f"| {domain} | {b_acc:.1%} | {a_acc:.1%} | {sign}{delta:.1%} |")
        lines.append("")

    # Overall delta
    b_overall = before.get('accuracy', 0)
    a_overall = after.get('accuracy', 0)
    delta_overall = a_overall - b_overall
    sign = '+' if delta_overall >= 0 else ''
    lines.append(f"**Overall: {b_overall:.1%} → {a_overall:.1%} ({sign}{delta_overall:.1%})**")
    lines.append("")


def _truncate(text: str, max_len: int) -> str:
    """Truncate text with ellipsis."""
    if len(text) <= max_len:
        return text
    return text[:max_len - 3] + '...'
