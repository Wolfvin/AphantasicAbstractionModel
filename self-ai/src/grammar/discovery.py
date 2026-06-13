# @WHO:   self-ai/src/grammar/discovery.py
# @WHAT:  Pattern Discovery — SELF discover pola dari grammar roles
# @PART:  grammar
# @ENTRY: PatternDiscovery.discover(), PatternDiscovery.find_role_groups()
#
# Insight kunci dari user:
# "kambing, domba, sapi = hewan. kambing = subjek di konteks ini.
#  maka domba dan sapi kalau di konteks ini juga subjek.
#  dan akan ada banyak perhitungan yang sangat elegan."
#
# Cara kerja:
# 1. Cari semua SPO axioms dengan relasi yang sama (misal: IS_A)
# 2. Group by (relation, object) → subjects yang share relasi
# 3. Subjects dalam group yang sama punya "role" yang sama
# 4. Jika salah satu subject punya property lain → hypothesize yang lain juga
#
# Contoh:
# - kucing IS_A mamalia, anjing IS_A mamalia, gajah IS_A mamalia
#   → {kucing, anjing, gajah} = "subjek IS_A mamalia" group
# - kucing HAS bulu → hypothesize: anjing HAS bulu? gajah HAS bulu?
# - Ini adalah ANALOGICAL REASONING berbasis grammar roles

from typing import List, Dict, Tuple, Optional, Set
from dataclasses import dataclass, field

from src.axiom.store import AxiomStore
from src.translation.translator import NodeID
from src.grammar.relations import RELATIONS, RelationType


@dataclass
class RoleGroup:
    """
    Sekelompok node yang berbagi role yang sama.

    Misalnya:
    - relation = IS_A, object = mamalia
    - subjects = [kucing, anjing, gajah]
    - Artinya: kucing, anjing, gajah semua IS_A mamalia
    - Mereka berbagi role "subjek yang IS_A mamalia"
    """
    relation_name: str          # IS_A, HAS, dll
    object_node_id: int         # node ID dari objek
    object_desc: str            # deskripsi objek
    subject_node_ids: List[int] # node IDs dari semua subjek
    subject_descs: List[str]    # deskripsi semua subjek
    size: int                   # jumlah subjek dalam group

    # Property yang diketahui dari salah satu subjek
    # tapi belum diketahui dari subjek lainnya → hypothesis
    shared_properties: List[Dict] = field(default_factory=list)


@dataclass
class AnalogicalHypothesis:
    """
    Hipotesis analogical berbasis grammar roles.

    Jika A dan B berada di role group yang sama,
    dan A punya property P,
    maka B mungkin juga punya property P.

    Ini BUKAN deductive (tidak pasti benar).
    Ini ANALOGICAL (mungkin benar, perlu verifikasi).
    Confidence lebih rendah dari deductive derivation.
    """
    source_subject: str     # subjek yang sudah punya property
    target_subject: str     # subjek yang dihipotesiskan punya property
    relation_name: str      # jenis relasi
    object_desc: str        # objek dari property
    group_context: str      # konteks role group
    confidence: float       # confidence (lebih rendah dari deductive)


class PatternDiscovery:
    """
    Pattern Discovery — SELF discover pola dari grammar roles.

    "Jika kambing, domba, sapi semua IS_A hewan,
     dan kambing membutuhkan air,
     maka domba dan sapi mungkin juga membutuhkan air."

    Ini adalah ANALOGICAL REASONING:
    - Berbasis grammar roles (subjek yang berbagi IS_A target)
    - Bukan deductive (tidak pasti)
    - Tapi sangat powerful untuk SELF yang sedang belajar
    - SELF bisa generate HIPOTESIS lalu TANYAKAN ke manusia

    Confidence: lebih rendah dari deductive (0.3-0.5)
    Karena analogical reasoning bisa salah:
    - Semua mamalia IS_A hewan, tapi tidak semua hewan menyusui
    - Derivation: mamalia IS_A hewan → hewan menyusui? TIDAK!
    - Tapi: kucing IS_A mamalia → kucing menyusui? YA (deductive)

    Makanya analogical hypothesis → SELF BERTANYA dulu sebelum menyimpulkan.
    """

    def __init__(self, axiom_store: AxiomStore, derivation_engine=None,
                 translator=None, node_store=None):
        self.axiom_store = axiom_store
        self.derivation_engine = derivation_engine
        self.translator = translator
        self.node_store = node_store

    def find_role_groups(self, relation_name: str = "IS_A") -> List[RoleGroup]:
        """
        Menemukan semua role group berdasarkan relasi.

        Misalnya untuk IS_A:
        - Semua subjek yang IS_A mamalia → satu group
        - Semua subjek yang IS_A reptil → satu group
        - Semua subjek yang IS_A hewan → satu group
        """
        # Group: (relation_name, object_id) → list of subject_ids
        groups: Dict[Tuple[str, int], List[int]] = {}
        group_descs: Dict[Tuple[str, int], str] = {}

        for axiom in self.axiom_store.get_all():
            # Hanya pakai axiom dari teaching dan derived — skip autonomous
            # Autonomous axioms bisa noisy (berdasarkan cosine similarity, bukan grammar)
            if axiom.source not in ("teaching", "derived"):
                continue

            # Cek relasi
            rel_name = None
            if self.derivation_engine:
                rel_name = self.derivation_engine._get_relation_name(axiom.relation)

            if rel_name != relation_name:
                continue

            obj_id = axiom.node_b.id
            key = (relation_name, obj_id)

            if key not in groups:
                groups[key] = []
                # Dapatkan deskripsi objek
                if self.translator:
                    group_descs[key] = self.translator.translate_to_human(axiom.node_b)
                else:
                    group_descs[key] = f"Node#{obj_id:04d}"

            groups[key].append(axiom.node_a.id)

        # Convert ke RoleGroup objects
        result = []
        for (rel_name, obj_id), subject_ids in groups.items():
            if len(subject_ids) < 2:
                # Skip group dengan hanya 1 subjek — tidak bisa analogize
                continue

            # Dapatkan deskripsi semua subjek
            subject_descs = []
            for sid in subject_ids:
                if self.translator:
                    desc = self.translator.translate_to_human(NodeID(id=sid))
                else:
                    desc = f"Node#{sid:04d}"
                subject_descs.append(desc)

            obj_desc = group_descs.get((rel_name, obj_id), f"Node#{obj_id:04d}")

            group = RoleGroup(
                relation_name=rel_name,
                object_node_id=obj_id,
                object_desc=obj_desc,
                subject_node_ids=subject_ids,
                subject_descs=subject_descs,
                size=len(subject_ids),
            )

            # Temukan shared properties
            group.shared_properties = self._find_shared_properties(group)

            result.append(group)

        # Sort by size — group terbesar dulu
        result.sort(key=lambda g: g.size, reverse=True)
        return result

    def _find_shared_properties(self, group: RoleGroup) -> List[Dict]:
        """
        Menemukan property yang dimiliki oleh SEBAGIAN subjek
        tapi belum diketahui dari subjek lainnya.

        Misalnya:
        - Group: {kucing, anjing, gajah} IS_A mamalia
        - kucing HAS bulu → anjing punya bulu? gajah punya bulu?
        """
        properties_by_subject: Dict[int, List[Dict]] = {}

        for axiom in self.axiom_store.get_all():
            # Skip autonomous — noisy
            if axiom.source not in ("teaching", "derived"):
                continue

            if axiom.node_a.id not in group.subject_node_ids:
                continue

            sid = axiom.node_a.id
            if sid not in properties_by_subject:
                properties_by_subject[sid] = []

            rel_name = "?"
            if self.derivation_engine:
                rel_name = self.derivation_engine._get_relation_name(axiom.relation) or "?"

            obj_desc = f"Node#{axiom.node_b.id:04d}"
            if self.translator:
                obj_desc = self.translator.translate_to_human(axiom.node_b)

            # Skip IS_A/INSTANCE_OF — itu sudah jadi group
            if rel_name in ("IS_A", "INSTANCE_OF"):
                continue

            properties_by_subject[sid].append({
                "relation_name": rel_name,
                "object_node_id": axiom.node_b.id,
                "object_desc": obj_desc,
                "source": axiom.source,
                "confidence": axiom.confidence,
            })

        # Cari property yang dimiliki oleh >= 1 subjek
        # tapi belum diketahui dari subjek lainnya
        shared = []
        all_properties = {}  # (rel_name, obj_id) → list of subject_ids yang punya

        for sid, props in properties_by_subject.items():
            for prop in props:
                key = (prop["relation_name"], prop["object_node_id"])
                if key not in all_properties:
                    all_properties[key] = []
                all_properties[key].append(sid)

        for (rel_name, obj_id), having_subjects in all_properties.items():
            if len(having_subjects) < 1:
                continue

            # Subjek yang BELUM punya property ini
            missing = [s for s in group.subject_node_ids if s not in having_subjects]
            if not missing:
                continue  # Semua sudah punya → bukan hypothesis

            # Dapatkan deskripsi
            obj_desc = f"Node#{obj_id:04d}"
            if self.translator:
                obj_desc = self.translator.translate_to_human(NodeID(id=obj_id))

            having_descs = []
            for sid in having_subjects:
                if self.translator:
                    having_descs.append(self.translator.translate_to_human(NodeID(id=sid)))
                else:
                    having_descs.append(f"Node#{sid:04d}")

            missing_descs = []
            for sid in missing:
                if self.translator:
                    missing_descs.append(self.translator.translate_to_human(NodeID(id=sid)))
                else:
                    missing_descs.append(f"Node#{sid:04d}")

            shared.append({
                "relation_name": rel_name,
                "object_desc": obj_desc,
                "having_subjects": having_descs,
                "missing_subjects": missing_descs,
                "coverage": len(having_subjects) / len(group.subject_node_ids),
            })

        return shared

    def discover(self) -> List[AnalogicalHypothesis]:
        """
        Menjalankan pattern discovery → menghasilkan analogical hypotheses.

        Hypotheses ini BELUM disimpan sebagai axiom.
        SELF bisa bertanya tentangnya via wonder().
        Jika manusia mengkonfirmasi → teach() → jadi axiom.
        """
        hypotheses = []

        # Cari role groups untuk IS_A (paling penting)
        for relation_name in ["IS_A", "INSTANCE_OF"]:
            groups = self.find_role_groups(relation_name)

            for group in groups:
                for prop in group.shared_properties:
                    # Confidence berdasarkan coverage dan group size
                    base_conf = 0.3  # analogical = rendah
                    coverage_bonus = prop["coverage"] * 0.2  # semakin banyak yang punya, semakin yakin
                    size_bonus = min(group.size / 10, 0.1)  # group besar = lebih yakin

                    conf = min(base_conf + coverage_bonus + size_bonus, 0.6)

                    for missing_desc in prop["missing_subjects"]:
                        having_descs_str = ", ".join(prop["having_subjects"][:3])

                        hypotheses.append(AnalogicalHypothesis(
                            source_subject=having_descs_str,
                            target_subject=missing_desc,
                            relation_name=prop["relation_name"],
                            object_desc=prop["object_desc"],
                            group_context=f"{group.subject_descs[0]} dll. IS_A {group.object_desc}",
                            confidence=conf,
                        ))

        # Sort by confidence
        hypotheses.sort(key=lambda h: h.confidence, reverse=True)
        return hypotheses

    def generate_discovery_questions(self) -> List[str]:
        """
        Menghasilkan pertanyaan berdasarkan pattern discovery.

        Contoh output:
        - "Saya perhatikan kucing dan anjing adalah mamalia, dan kucing memiliki bulu.
           Apakah anjing juga memiliki bulu?"
        - "Saya tahu ular dan buaya adalah reptil, dan ular bertelur.
           Apakah buaya juga bertelur?"
        """
        hypotheses = self.discover()
        questions = []

        for hyp in hypotheses[:5]:  # Top 5 saja
            # Dapatkan label relasi
            rel_type = RELATIONS.get(hyp.relation_name)
            rel_label = rel_type.label_id if rel_type else hyp.relation_name.lower()

            group_is_a_part = hyp.group_context.split('IS_A')[-1].strip() if 'IS_A' in hyp.group_context else 'kelompok yang sama'
            question = (
                f"Saya perhatikan {hyp.source_subject} adalah {group_is_a_part}, "
                f"dan {hyp.source_subject} {rel_label} {hyp.object_desc}. "
                f"Apakah {hyp.target_subject} juga {rel_label} {hyp.object_desc}?"
            )
            questions.append(question)

        return questions
