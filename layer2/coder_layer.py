"""
AAM Layer 2 — Coder Layer (Base)

Code understanding as structured knowledge. Parses source code into
structural elements (classes, functions, variables, imports) and
represents them as data that can be ingested into the RSVS graph.

This is the Layer 2 base — providing core parsing and analysis.
Layer 3's DeductiveCoderLayer extends this with RSVS compositional
semantics for deeper cross-layer analysis.

Analogi: Layer 2 CoderLayer = Jin Soun bisa membaca manual teknik
bela diri dan mengerti kelemahannya dari struktur kode.
Layer 3 = juga menghubungkan teknik dari manual BERBEDA.

Design decisions:
  - No raw code stored in graph — only structural relations
  - Each CodeElement becomes a node; parent-child → composition
  - Language detection is heuristic-based (no external deps)
  - Source trust for code is moderate (0.6) by default
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Any, Optional

from .bridge import RsvsBridge, get_bridge

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

CODE_SOURCE_TRUST: float = 0.6
"""Default trust score for code as a knowledge source."""

DEFAULT_EXTENSIONS: dict[str, str] = {
    ".py": "python",
    ".js": "javascript",
    ".ts": "typescript",
    ".rs": "rust",
    ".go": "go",
    ".java": "java",
    ".cpp": "cpp",
    ".c": "c",
    ".rb": "ruby",
    ".php": "php",
    ".swift": "swift",
    ".kt": "kotlin",
    ".scala": "scala",
    ".sh": "shell",
    ".sql": "sql",
    ".html": "html",
    ".css": "css",
}
"""Map of file extensions to language names."""

ALL_SUPPORTED_EXTENSIONS: set[str] = set(DEFAULT_EXTENSIONS.keys())
"""All supported file extensions."""


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class CodeElement:
    """A structural element extracted from source code.

    Represents a function, class, variable, import, or other code
    construct with its name, kind, location, and relationships.

    Attributes:
        name: The element name (e.g. 'MyClass', 'calculate_total').
        kind: The element kind ('class', 'function', 'method', 'variable',
              'import', 'decorator', 'constant', 'parameter', 'return').
        source: Source identifier (filename or 'snippet').
        line_start: Starting line number (1-based).
        line_end: Ending line number.
        parent: Parent element name (e.g. class name for methods).
        children: Child element names (e.g. methods in a class).
        docstring: Docstring or comment, if any.
        code_text: The actual code text (used for RSVS ingestion only,
                   not stored permanently).
        confidence: Confidence score for this extraction.
    """

    name: str
    kind: str = "unknown"
    source: str = "snippet"
    line_start: int = 0
    line_end: int = 0
    parent: str = ""
    children: list[str] = field(default_factory=list)
    docstring: str = ""
    code_text: str = ""
    confidence: float = 0.7

    def to_dict(self) -> dict:
        """Serialize to plain dict."""
        return {
            "name": self.name,
            "kind": self.kind,
            "source": self.source,
            "line_start": self.line_start,
            "line_end": self.line_end,
            "parent": self.parent,
            "children": self.children,
            "docstring": self.docstring[:200] if self.docstring else "",
            "confidence": self.confidence,
        }

    def to_ingest_text(self) -> str:
        """Convert to text suitable for RSVS ingestion.

        Produces a natural language description of this element
        that preserves structural information without storing raw code.
        """
        parts = []
        if self.parent:
            parts.append(f"{self.parent}.{self.name} is a {self.kind}")
        else:
            parts.append(f"{self.name} is a {self.kind}")

        if self.children:
            child_str = ", ".join(self.children[:10])
            parts.append(f"contains: {child_str}")

        if self.docstring:
            # Use docstring as-is but truncate
            parts.append(f"documentation: {self.docstring[:200]}")

        return ". ".join(parts)


@dataclass
class CodeAnalysisResult:
    """Result of analyzing source code.

    Contains extracted elements, patterns, anomalies, similar code,
    and suggestions from structural analysis.

    Attributes:
        query: Description of what was analyzed.
        elements_found: List of extracted CodeElement dicts.
        similar_code: List of similar code pairs found.
        anomalies: List of detected anomalies.
        patterns: List of identified patterns.
        suggestions: List of improvement suggestions.
        evidence_nodes: RSVS node labels used as evidence.
        confidence: Overall confidence of the analysis.
    """

    query: str = ""
    elements_found: list[dict] = field(default_factory=list)
    similar_code: list[dict] = field(default_factory=list)
    anomalies: list[dict] = field(default_factory=list)
    patterns: list[dict] = field(default_factory=list)
    suggestions: list[str] = field(default_factory=list)
    evidence_nodes: list[str] = field(default_factory=list)
    confidence: float = 0.3

    def to_dict(self) -> dict:
        """Serialize to plain dict."""
        return {
            "query": self.query,
            "elements_found": self.elements_found,
            "similar_code": self.similar_code,
            "anomalies": self.anomalies,
            "patterns": self.patterns,
            "suggestions": self.suggestions,
            "evidence_nodes": self.evidence_nodes,
            "confidence": round(self.confidence, 4),
        }


# ---------------------------------------------------------------------------
# Parsing functions
# ---------------------------------------------------------------------------

def detect_language(code: str, filename: str = "") -> str:
    """Detect the programming language of a code snippet.

    Uses file extension first (if filename provided), then falls back
    to heuristic detection based on syntax patterns.

    Args:
        code: The source code string.
        filename: Optional filename for extension-based detection.

    Returns:
        Language name string (e.g. 'python', 'javascript').
    """
    # Try extension first
    if filename:
        for ext in sorted(DEFAULT_EXTENSIONS.keys(), key=len, reverse=True):
            if filename.endswith(ext):
                return DEFAULT_EXTENSIONS[ext]

    # Heuristic detection
    code_stripped = code.strip()

    # Python indicators
    py_indicators = [
        r'\bdef\s+\w+\s*\(',
        r'\bclass\s+\w+',
        r'\bimport\s+\w+',
        r'\bfrom\s+\w+\s+import',
        r':\s*$',
        r'\bself\b',
        r'@\w+\s*\n\s*def',
    ]
    py_score = sum(1 for p in py_indicators if re.search(p, code_stripped, re.MULTILINE))

    # JavaScript/TypeScript indicators
    js_indicators = [
        r'\bfunction\s+\w+',
        r'\bconst\s+\w+',
        r'\blet\s+\w+',
        r'\bvar\s+\w+',
        r'=>\s*{',
        r'\bconsole\.',
        r'\brequire\(',
    ]
    js_score = sum(1 for p in js_indicators if re.search(p, code_stripped, re.MULTILINE))

    # Rust indicators
    rs_indicators = [
        r'\bfn\s+\w+',
        r'\blet\s+mut\b',
        r'\bimpl\s+\w+',
        r'\bpub\s+fn\b',
        r'\buse\s+\w+::',
        r'->\s*\w+',
    ]
    rs_score = sum(1 for p in rs_indicators if re.search(p, code_stripped, re.MULTILINE))

    scores = {
        "python": py_score,
        "javascript": js_score,
        "rust": rs_score,
    }

    if max(scores.values()) > 0:
        return max(scores, key=scores.get)  # type: ignore[arg-type]

    return "unknown"


def parse_python_code(code: str, source: str = "snippet") -> list[CodeElement]:
    """Parse Python source code into structural elements.

    Uses regex-based parsing to extract classes, functions, methods,
    imports, and top-level variables. Does not use ast module for
    maximum compatibility and zero external dependencies.

    Args:
        code: Python source code string.
        source: Source identifier (filename or 'snippet').

    Returns:
        List of CodeElement instances representing the code structure.
    """
    elements: list[CodeElement] = []
    lines = code.split('\n')

    # Track class stack for method→class association
    current_class = ""
    current_class_children: list[str] = []

    for i, line in enumerate(lines, 1):
        stripped = line.strip()

        # Skip empty lines and comments
        if not stripped or stripped.startswith('#'):
            continue

        # Import statements
        import_match = re.match(
            r'^(?:from\s+[\w.]+\s+)?import\s+([\w.,\s*]+)', stripped
        )
        if import_match:
            import_names = [
                n.strip().split(' as ')[0].strip()
                for n in import_match.group(1).split(',')
                if n.strip()
            ]
            for name in import_names[:5]:
                elements.append(CodeElement(
                    name=name,
                    kind="import",
                    source=source,
                    line_start=i,
                    line_end=i,
                    confidence=0.9,
                ))
            continue

        # Class definitions
        class_match = re.match(r'^class\s+(\w+)', stripped)
        if class_match:
            # Save previous class
            if current_class:
                _update_class_children(elements, current_class, current_class_children)

            class_name = class_match.group(1)
            current_class = class_name
            current_class_children = []

            elements.append(CodeElement(
                name=class_name,
                kind="class",
                source=source,
                line_start=i,
                parent="",
                children=[],
                confidence=0.9,
            ))
            continue

        # Function/method definitions
        func_match = re.match(r'^(?:async\s+)?def\s+(\w+)\s*\(', stripped)
        if func_match:
            func_name = func_match.group(1)

            # Determine if method (indented under class) or function
            indent = len(line) - len(line.lstrip())
            if indent > 0 and current_class:
                kind = "method"
                parent = current_class
                current_class_children.append(func_name)
            else:
                kind = "function"
                parent = ""
                # Reset class tracking if unindented
                if indent == 0 and current_class:
                    _update_class_children(elements, current_class, current_class_children)
                    current_class = ""
                    current_class_children = []

            elements.append(CodeElement(
                name=func_name,
                kind=kind,
                source=source,
                line_start=i,
                parent=parent,
                confidence=0.85,
            ))
            continue

        # Decorator lines — associate with next def
        if stripped.startswith('@'):
            continue

        # Top-level variable assignment (simple heuristic)
        var_match = re.match(r'^(\w+)\s*=\s*', stripped)
        if var_match and not stripped.startswith('_') and len(line) - len(line.lstrip()) == 0:
            var_name = var_match.group(1)
            if var_name[0].isupper() or var_name.isupper():
                kind = "constant"
            else:
                kind = "variable"

            elements.append(CodeElement(
                name=var_name,
                kind=kind,
                source=source,
                line_start=i,
                line_end=i,
                confidence=0.6,
            ))

    # Save final class
    if current_class:
        _update_class_children(elements, current_class, current_class_children)

    return elements


def _parse_code_regex(code: str, source: str = "snippet", language: str = "unknown") -> list[CodeElement]:
    """Parse non-Python code using generic regex patterns.

    Uses language-aware heuristics to extract structural elements
    from JavaScript, Rust, Go, and other languages.

    Args:
        code: Source code string.
        source: Source identifier.
        language: Language name for pattern selection.

    Returns:
        List of CodeElement instances.
    """
    elements: list[CodeElement] = []
    lines = code.split('\n')

    if language in ("javascript", "typescript"):
        for i, line in enumerate(lines, 1):
            stripped = line.strip()
            # function declarations
            func_match = re.match(r'(?:async\s+)?function\s+(\w+)', stripped)
            if func_match:
                elements.append(CodeElement(
                    name=func_match.group(1), kind="function",
                    source=source, line_start=i, confidence=0.8,
                ))
                continue
            # class declarations
            class_match = re.match(r'class\s+(\w+)', stripped)
            if class_match:
                elements.append(CodeElement(
                    name=class_match.group(1), kind="class",
                    source=source, line_start=i, confidence=0.8,
                ))
                continue
            # const/let/var declarations
            var_match = re.match(r'(?:const|let|var)\s+(\w+)', stripped)
            if var_match:
                elements.append(CodeElement(
                    name=var_match.group(1), kind="variable",
                    source=source, line_start=i, confidence=0.6,
                ))

    elif language == "rust":
        for i, line in enumerate(lines, 1):
            stripped = line.strip()
            # fn declarations
            fn_match = re.match(r'(?:pub\s+)?(?:async\s+)?fn\s+(\w+)', stripped)
            if fn_match:
                kind = "method" if '&self' in stripped or 'self.' in stripped else "function"
                elements.append(CodeElement(
                    name=fn_match.group(1), kind=kind,
                    source=source, line_start=i, confidence=0.8,
                ))
                continue
            # struct/enum/impl
            struct_match = re.match(r'(?:pub\s+)?(?:struct|enum|impl)\s+(\w+)', stripped)
            if struct_match:
                kind = "class"
                elements.append(CodeElement(
                    name=struct_match.group(1), kind=kind,
                    source=source, line_start=i, confidence=0.8,
                ))

    else:
        # Generic: try to find word patterns that look like definitions
        for i, line in enumerate(lines, 1):
            stripped = line.strip()
            # Look for "name(...)" patterns that could be function calls/definitions
            name_pattern = re.match(r'(\w+)\s*\(', stripped)
            if name_pattern and not stripped.startswith('#') and not stripped.startswith('//'):
                name = name_pattern.group(1)
                # Skip common keywords
                if name not in ('if', 'for', 'while', 'switch', 'return', 'else', 'try', 'catch'):
                    elements.append(CodeElement(
                        name=name, kind="function",
                        source=source, line_start=i, confidence=0.4,
                    ))

    return elements


def _update_class_children(
    elements: list[CodeElement],
    class_name: str,
    children: list[str],
) -> None:
    """Update a class CodeElement with its children (methods)."""
    for elem in elements:
        if elem.name == class_name and elem.kind == "class":
            elem.children = list(children)
            break


# ---------------------------------------------------------------------------
# CoderLayer — Main class
# ---------------------------------------------------------------------------

class CoderLayer:
    """Code understanding as structured knowledge graph.

    Parses source code into structural elements and ingests them
    into the RSVS graph. Each function, class, and variable becomes
    a node; parent-child relationships become composition references.

    This is the Layer 2 base — providing core parsing, ingestion,
    and basic analysis. Layer 3's DeductiveCoderLayer extends this
    with RSVS compositional semantics for deeper analysis.

    Usage:
        coder = CoderLayer()
        result = coder.analyze_code(python_code, language="python")
        for elem in result.elements_found:
            print(f"Found {elem['kind']}: {elem['name']}")
    """

    def __init__(self, bridge: Optional[RsvsBridge] = None) -> None:
        """Initialize the CoderLayer.

        Args:
            bridge: Optional pre-built RsvsBridge. If None, creates one.
        """
        self._bridge = bridge or get_bridge()
        self._source_trust = CODE_SOURCE_TRUST

    # ==================================================================
    # Public API
    # ==================================================================

    def ingest_code(
        self,
        code: str,
        language: str = "auto",
        source: str = "code_snippet",
    ) -> list[CodeElement]:
        """Parse code and ingest structural elements into RSVS.

        Each element is converted to a natural language description
        and ingested into the knowledge graph. Parent-child
        relationships are stored as composition references.

        Args:
            code: Source code string.
            language: Programming language (default: "auto").
            source: Source identifier for provenance.

        Returns:
            List of CodeElement instances that were ingested.
        """
        if language == "auto":
            language = detect_language(code, filename=source)

        # Parse into structural elements
        if language == "python":
            elements = parse_python_code(code, source=source)
        else:
            elements = _parse_code_regex(code, source=source, language=language)

        # Ingest each element into RSVS
        for element in elements:
            ingest_text = element.to_ingest_text()
            if self._bridge.is_available:
                try:
                    self._bridge.ingest(ingest_text)
                except Exception as exc:
                    logger.debug("Ingest failed for '%s': %s", element.name, exc)

        logger.info(
            "CoderLayer.ingest_code(): language=%s, elements=%d, source='%s'",
            language, len(elements), source,
        )
        return elements

    def ingest_file(self, filepath: str) -> list[CodeElement]:
        """Parse a file and ingest its structural elements.

        Args:
            filepath: Path to the source code file.

        Returns:
            List of CodeElement instances.
        """
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                code = f.read()
        except Exception as exc:
            logger.error("Failed to read file '%s': %s", filepath, exc)
            return []

        language = detect_language(code, filename=filepath)
        return self.ingest_code(code, language=language, source=filepath)

    def ingest_directory(
        self,
        dirpath: str,
        extensions: Optional[set[str]] = None,
    ) -> dict[str, list[CodeElement]]:
        """Parse all code files in a directory and ingest them.

        Args:
            dirpath: Path to the directory.
            extensions: Set of file extensions to include (default: all supported).

        Returns:
            Dict mapping filename → list of CodeElement instances.
        """
        import os

        exts = extensions or ALL_SUPPORTED_EXTENSIONS
        results: dict[str, list[CodeElement]] = {}

        for root, dirs, files in os.walk(dirpath):
            # Skip hidden directories and common non-code dirs
            dirs[:] = [d for d in dirs if not d.startswith('.') and d not in (
                '__pycache__', 'node_modules', '.git', 'target', 'build',
            )]
            for fname in files:
                _, ext = os.path.splitext(fname)
                if ext in exts:
                    fpath = os.path.join(root, fname)
                    elements = self.ingest_file(fpath)
                    if elements:
                        results[fpath] = elements

        return results

    def analyze_code(
        self,
        code: str,
        language: str = "auto",
        source: str = "code_snippet",
    ) -> CodeAnalysisResult:
        """Analyze code and produce a structured result.

        Parses the code, ingests elements into RSVS, and produces
        a CodeAnalysisResult with elements found, patterns, and
        basic analysis.

        Args:
            code: Source code string.
            language: Programming language (default: "auto").
            source: Source identifier.

        Returns:
            CodeAnalysisResult with analysis output.
        """
        if language == "auto":
            language = detect_language(code, filename=source)

        result = CodeAnalysisResult(query=f"Analysis of {source}")

        # Parse and ingest
        elements = self.ingest_code(code, language=language, source=source)

        # Record elements found
        for element in elements:
            result.elements_found.append(element.to_dict())

        # Basic pattern detection via RSVS
        if self._bridge.is_available:
            # Check for similar elements via spreading activation
            element_names = [e.name for e in elements if e.name]
            for name in element_names[:10]:
                try:
                    senses = self._bridge.senses(name)
                    if senses and isinstance(senses, list):
                        for sense in senses[:3]:
                            if isinstance(sense, dict):
                                gs = sense.get("grounding_score", 0.0)
                                if gs > 0.3:
                                    result.evidence_nodes.append(name)
                                    result.confidence = max(result.confidence, 0.4)
                except Exception:
                    pass

        # Pattern: class with many methods
        classes = [e for e in elements if e.kind == "class"]
        for cls in classes:
            if len(cls.children) > 3:
                result.patterns.append({
                    "type": "large_class",
                    "description": f"Class '{cls.name}' has {len(cls.children)} methods",
                    "element": cls.name,
                })
                result.confidence = max(result.confidence, 0.5)

        # Pattern: function with no docstring
        funcs = [e for e in elements if e.kind in ("function", "method")]
        undocumented = [f for f in funcs if not f.docstring]
        if undocumented:
            result.suggestions.append(
                f"{len(undocumented)} function(s) lack docstrings"
            )

        # Overall confidence
        if elements:
            result.confidence = max(result.confidence, 0.4)

        logger.info(
            "CoderLayer.analyze_code(): language=%s, elements=%d, "
            "patterns=%d, confidence=%.3f",
            language, len(elements), len(result.patterns), result.confidence,
        )

        return result

    def get_code_summary(self, source: str = "") -> dict:
        """Get a summary of all code ingested from a source.

        Args:
            source: Source identifier to filter by (empty = all).

        Returns:
            Dict with summary statistics.
        """
        return {
            "source": source or "all",
            "note": "Code summary requires RSVS core for full statistics",
            "bridge_available": self._bridge.is_available,
        }
