"""
AAM Layer 3 — Reasoning Engine

Status: STUB — belum diimplementasi.

Rencana:
  Mengambil activated nodes dari Layer 2 PatternOutput dan
  membangun deductive chain yang fully traceable:
  - Setiap klaim punya evidence node di graph
  - Setiap confidence punya grounding score
  - Output bisa di-audit node-by-node

Analogi Jin Soun:
  "Gu Ilmu + Jang Hangi mencuri Snow Plum Pill."
  Evidence: [tanggal Hefei] → [misi Diancang] → [tidak ada pil di pasar]
  Confidence: 87%
"""


class ReasoningEngine:
    def build_chain(self, pattern_result, graph_state):
        raise NotImplementedError("ReasoningEngine belum diimplementasi.")
