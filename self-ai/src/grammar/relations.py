# @WHO:   self-ai/src/grammar/relations.py
# @WHAT:  EMERGENT relation types — SELF discovers relations from observation
# @PART:  grammar
# @ENTRY: get_relation(), register_relation(), all_relation_names()
#
# v6: EMERGENT RELATIONS — tidak ada hardcoded relation types!
#
# PRINSIP: Semua relation type harus di-generate oleh:
#   1. Bertanya (asking) → konfirmasi → register new relation
#   2. Pengamatan (observation) → detect recurring verb patterns → emerge
#   3. Teaching → SPO parse menghasilkan relasi baru → register
#
# Cara kerja:
# - SELF lahir HANYA dengan IS_A (satu-satunya relasi bawaan).
#   IS_A adalah relasi fundamental — tanpa itu, tidak ada hierarki,
#   tidak ada inheritance, tidak ada grouping.
# - Setiap kali SELF diajarkan kalimat dengan pola SPO baru,
#   parser mendeteksi verb phrase yang belum dikenali → RELASI BARU.
# - Relasi baru di-register secara dinamis.
# - Transitive/symmetric flags TIDAK di-hardcode — SELF belajar dari
#   observasi apakah suatu relasi bersifat transitif atau simetris.
#
# "Saya tidak dilahirkan tahu bahwa BREATHES_WITH itu transitif.
#  Saya mengamati bahwa setiap kali A bernapas dengan X dan A IS_A B,
#  B juga bernapas dengan X. Setelah itu, saya simpulkan sendiri
#  bahwa BREATHES_WITH itu transitif."

from dataclasses import dataclass, field
from typing import Optional, Dict, List, Set
from collections import defaultdict
import json
import os


@dataclass
class RelationType:
    """
    Tipe relasi dalam bahasa internal SELF.

    SELF tidak menyimpan kata manusia — tapi dia menyimpan JENIS relasi.
    IS_A, HAS, CAN — ini adalah "kata kerja" internalnya.

    v6: RelationType sekarang EMERGENT.
    - transitive dan symmetric TIDAK di-hardcode
    - SELF belajar dari observasi apakah relasi bersifat transitif/simetris
    - observation_count: berapa kali SELF mengamati relasi ini
    - transitive_observations: berapa kali SELF mengamati transitive pattern
    - symmetric_observations: berapa kali SELF mengamati symmetric pattern
    """
    name: str
    label_id: str           # Indonesian label: "adalah", "memiliki", dll
    description: str        # Semantic description for embedding
    transitive: bool = False  # Default: TIDAK transitif sampai terbukti
    symmetric: bool = False   # Default: TIDAK simetris sampai terbukti
    observation_count: int = 0  # Berapa kali relasi ini muncul di aksioma
    transitive_observations: int = 0  # Observasi yang mendukung transitive
    symmetric_observations: int = 0   # Observasi yang mendukung symmetric
    source: str = "emergent"  # "seed" | "emergent" | "teaching"

    # Thresholds untuk memutuskan transitive/symmetric
    TRANSITIVE_THRESHOLD: int = 2  # Min observations untuk declare transitive
    SYMMETRIC_THRESHOLD: int = 2   # Min observations untuk declare symmetric

    def observe_transitive(self):
        """
        Observasi bahwa relasi ini bersifat transitif.
        Dipanggil ketika RuleLearner menemukan rule: A R B + B R C → A R C
        untuk relasi yang sama (R = R1 = R2 = R3).
        """
        self.transitive_observations += 1
        if self.transitive_observations >= self.TRANSITIVE_THRESHOLD:
            self.transitive = True

    def observe_symmetric(self):
        """
        Observasi bahwa relasi ini bersifat simetris.
        Dipanggil ketika SELF menemukan A R B dan B R A
        untuk relasi yang sama.
        """
        self.symmetric_observations += 1
        if self.symmetric_observations >= self.SYMMETRIC_THRESHOLD:
            self.symmetric = True

    def increment_observation(self):
        """Setiap kali relasi ini muncul di aksioma baru."""
        self.observation_count += 1


# ============================================================
# SEED RELATIONS — Hanya relasi paling fundamental
# ============================================================
# SELF lahir dengan HANYA 2 relasi:
# 1. IS_A — relasi hierarki fundamental. Tanpa ini, tidak ada
#    grouping, tidak ada inheritance, tidak ada analogical reasoning.
# 2. INSTANCE_OF — distinguish antara class dan instance.
#    "kucing IS_A mamalia" vs "garfield INSTANCE_OF kucing"
#
# Semua relasi lain (HAS, CAN, BREATHES_WITH, dll) akan
# EMERGE dari teaching dan observation.
#
# Kenapa IS_A dan INSTANCE_OF saja?
# - IS_A diperlukan untuk cold-start — SELF perlu bisa mengelompokkan
#   konsep sejak awal. Tanpa IS_A, tidak ada cara untuk mengorganisir
#   pengetahuan.
# - INSTANCE_OF diperlukan untuk membedakan "kucing adalah mamalia"
#   (class membership) dari "garfield adalah kucing" (instance).
# - Semua relasi lain bisa ditemukan SELF sendiri melalui pengamatan.
# ============================================================

SEED_RELATIONS: Dict[str, RelationType] = {
    "IS_A": RelationType(
        name="IS_A",
        label_id="adalah",
        description="adalah jenis dari",
        transitive=True,  # IS_A adalah satu-satunya yang kita tahu pasti transitif
        symmetric=False,
        source="seed",
    ),
    "INSTANCE_OF": RelationType(
        name="INSTANCE_OF",
        label_id="contoh dari",
        description="merupakan contoh spesifik dari",
        transitive=True,  # INSTANCE_OF juga transitif (garfield INSTANCE_OF kucing, kucing IS_A mamalia)
        symmetric=False,
        source="seed",
    ),
}


class RelationRegistry:
    """
    Registry dinamis untuk relation types.

    SELF lahir dengan seed relations (IS_A, INSTANCE_OF).
    Semua relasi lain muncul dari:
    1. Teaching — SPO parse mendeteksi verb phrase baru
    2. Observation — pola berulang menunjukkan relasi baru
    3. Asking — SELF bertanya dan belajar relasi baru

    Registry ini persisten — relasi yang dipelajari tidak hilang.
    """

    def __init__(self):
        # Mulai dari seed relations
        self._relations: Dict[str, RelationType] = dict(SEED_RELATIONS)

        # Reverse lookup: Indonesian keyword → relation name
        self._keyword_to_relation: Dict[str, str] = {
            "adalah": "IS_A",
            "contoh dari": "INSTANCE_OF",
        }

        # Track which verb phrases have been seen but not yet registered
        self._pending_verbs: Dict[str, int] = defaultdict(int)

    def register_relation(self, name: str, label_id: str, description: str,
                          transitive: bool = False, symmetric: bool = False,
                          source: str = "emergent") -> RelationType:
        """
        Mendaftarkan relasi baru ke registry.

        Dipanggil ketika:
        1. SPO parser mendeteksi verb phrase yang belum dikenali
        2. Teaching menghasilkan relasi yang belum terdaftar
        3. SELF bertanya dan belajar relasi baru

        name: Nama kanonik (uppercase, snake_case) — misal "BREATHES_WITH"
        label_id: Label bahasa Indonesia — misal "bernapas dengan"
        description: Deskripsi semantik untuk embedding
        """
        # Jika sudah ada, update saja
        if name in self._relations:
            existing = self._relations[name]
            # Update label jika ada label baru
            if label_id and label_id != existing.label_id:
                # Register new keyword mapping
                self._keyword_to_relation[label_id.lower()] = name
            return existing

        # Buat relasi baru
        rel = RelationType(
            name=name,
            label_id=label_id,
            description=description,
            transitive=transitive,
            symmetric=symmetric,
            source=source,
        )

        self._relations[name] = rel

        # Register keyword mapping
        if label_id:
            self._keyword_to_relation[label_id.lower()] = name

        return rel

    def register_from_verb(self, verb_phrase: str) -> RelationType:
        """
        Mendaftarkan relasi dari verb phrase bahasa Indonesia.

        Dipanggil oleh SPO parser ketika mendeteksi verb phrase
        yang belum dikenali.

        Misalnya: "bernapas dengan" → BREATHES_WITH
                   "hidup di" → LIVES_IN
                   "suka makan" → EATS

        Nama kanonik di-generate dari verb phrase:
        1. Bersihkan dan uppercase
        2. Ganti spasi dengan underscore
        3. Jika sudah ada, return existing
        """
        verb_clean = verb_phrase.strip().lower()

        # Cek apakah sudah ada keyword mapping
        if verb_clean in self._keyword_to_relation:
            existing_name = self._keyword_to_relation[verb_clean]
            return self._relations[existing_name]

        # Generate canonical name
        canonical_name = verb_clean.upper().replace(" ", "_")

        # Jika canonical name sudah ada, return existing
        if canonical_name in self._relations:
            return self._relations[canonical_name]

        # Buat relasi baru — emergent!
        description = verb_phrase  # Description = verb phrase itu sendiri
        rel = self.register_relation(
            name=canonical_name,
            label_id=verb_clean,
            description=description,
            source="emergent",
        )

        return rel

    def observe_verb(self, verb_phrase: str):
        """
        Mencatat bahwa verb phrase ini muncul lagi.
        Jika muncul cukup sering, otomatis register sebagai relasi.

        Threshold: verb phrase harus muncul minimal 2 kali
        sebelum di-register sebagai relasi resmi.
        """
        verb_clean = verb_phrase.strip().lower()

        # Jika sudah terdaftar, skip
        if verb_clean in self._keyword_to_relation:
            return

        self._pending_verbs[verb_clean] += 1

        # Auto-register setelah 2 observasi
        if self._pending_verbs[verb_clean] >= 2:
            self.register_from_verb(verb_phrase)
            del self._pending_verbs[verb_clean]

    def get_relation(self, name: str) -> Optional[RelationType]:
        """Mengambil RelationType berdasarkan nama kanonik."""
        return self._relations.get(name)

    def get_relation_by_keyword(self, keyword: str) -> Optional[RelationType]:
        """Mengambil RelationType berdasarkan keyword bahasa Indonesia."""
        name = self._keyword_to_relation.get(keyword.lower())
        if name:
            return self._relations.get(name)
        return None

    def all_relation_names(self) -> List[str]:
        """Mengembalikan semua nama relasi yang tersedia."""
        return list(self._relations.keys())

    def all_relations(self) -> Dict[str, RelationType]:
        """Mengembalikan semua relasi."""
        return dict(self._relations)

    def get_seed_names(self) -> List[str]:
        """Mengembalikan nama relasi yang merupakan seed."""
        return [name for name, rel in self._relations.items() if rel.source == "seed"]

    def get_emergent_names(self) -> List[str]:
        """Mengembalikan nama relasi yang muncul dari pengamatan."""
        return [name for name, rel in self._relations.items() if rel.source == "emergent"]

    def get_status(self) -> Dict:
        """Status dari relation registry."""
        seed = []
        emergent = []
        for name, rel in sorted(self._relations.items()):
            info = {
                "name": name,
                "label": rel.label_id,
                "transitive": rel.transitive,
                "symmetric": rel.symmetric,
                "observations": rel.observation_count,
                "transitive_obs": rel.transitive_observations,
                "symmetric_obs": rel.symmetric_observations,
                "source": rel.source,
            }
            if rel.source == "seed":
                seed.append(info)
            else:
                emergent.append(info)

        return {
            "seed_relations": seed,
            "emergent_relations": emergent,
            "total": len(self._relations),
            "seed_count": len(seed),
            "emergent_count": len(emergent),
            "pending_verbs": len(self._pending_verbs),
        }


# ============================================================
# BACKWARD COMPATIBILITY — Legacy functions
# ============================================================
# File lain masih menggunakan RELATIONS dict dan get_relation().
# Kita maintain backward compatibility dengan global registry.

# Global registry instance
_registry = RelationRegistry()

# RELATIONS dict — backward compatibility
# Returns current state of the registry
class _RELATIONSProxy(Dict):
    """Proxy dict yang selalu reflect state terkini dari registry."""
    def __iter__(self):
        return iter(_registry.all_relations())
    def __len__(self):
        return len(_registry.all_relations())
    def __contains__(self, key):
        return key in _registry.all_relations()
    def __getitem__(self, key):
        return _registry.get_relation(key)
    def keys(self):
        return _registry.all_relations().keys()
    def values(self):
        return _registry.all_relations().values()
    def items(self):
        return _registry.all_relations().items()
    def get(self, key, default=None):
        rel = _registry.get_relation(key)
        return rel if rel is not None else default

RELATIONS = _RELATIONSProxy()

def get_relation(name: str) -> Optional[RelationType]:
    """Mengambil RelationType berdasarkan nama kanonik."""
    return _registry.get_relation(name)

def get_relation_by_keyword(keyword: str) -> Optional[RelationType]:
    """Mengambil RelationType berdasarkan keyword bahasa Indonesia."""
    return _registry.get_relation_by_keyword(keyword)

def all_relation_names() -> List[str]:
    """Mengembalikan semua nama relasi yang tersedia."""
    return _registry.all_relation_names()

def register_relation(name: str, label_id: str, description: str,
                      transitive: bool = False, symmetric: bool = False,
                      source: str = "emergent") -> RelationType:
    """Mendaftarkan relasi baru ke registry global."""
    return _registry.register_relation(name, label_id, description,
                                        transitive, symmetric, source)

def register_from_verb(verb_phrase: str) -> RelationType:
    """Mendaftarkan relasi dari verb phrase."""
    return _registry.register_from_verb(verb_phrase)

def observe_verb(verb_phrase: str):
    """Mencatat verb phrase yang muncul — auto-register setelah 2x."""
    _registry.observe_verb(verb_phrase)

def get_registry() -> RelationRegistry:
    """Mengakses global registry instance."""
    return _registry

def set_registry(registry: RelationRegistry):
    """Set global registry instance (untuk load dari persistence)."""
    global _registry
    _registry = registry
