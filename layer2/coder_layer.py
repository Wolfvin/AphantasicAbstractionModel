"""
Coder Layer — Code understanding as structured knowledge.

Analogi: Jin Soun bisa membaca manual teknik bela diri dan
mengerti kelemahannya dari struktur — bukan dari teks saja.
Coder Layer melakukan hal yang sama untuk kode: memahami
struktur, menemukan kelemahan (bug), dan menyarankan perbaikan.

Like ContextLayer but specialized for code:
- Code files are ingested as structured knowledge
- Functions/classes become nodes in the graph
- Dependencies become edges
- Bug detection = anomaly detection on code structure
- Code review = pattern completion on expected vs actual

Flow:
1. Code input → parse into structural elements (AST or regex)
2. Structural elements → ingest into RSVS graph
3. Analysis query → trigger recall + cross-reference + anomaly + pattern
4. Output → structured code analysis report
"""

from __future__ import annotations

import ast
import logging
import os
import re
import time
from dataclasses import dataclass, field
from typing import Any, Optional

from .bridge import RsvsBridge, get_bridge

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Code trust mapping — how much we trust each code source type
# Analogi: Jin Soun lebih percaya manual resmi sekolah daripada catatan pinggir
# ---------------------------------------------------------------------------

CODE_SOURCE_TRUST: dict[str, float] = {
    "code_file": 0.85,       # Direct code file — high trust (it IS the source)
    "code_snippet": 0.7,     # Code snippet — decent but may lack context
    "documentation": 0.75,   # Code documentation — generally trustworthy
    "test_file": 0.9,        # Test files — highest trust (defines expected behavior)
    "config_file": 0.6,      # Configuration — moderate, may be outdated
    "generated": 0.5,        # Generated code — lower trust
    "unknown": 0.5,          # Unknown provenance — neutral
}

# Default file extensions to ingest per language
DEFAULT_EXTENSIONS: dict[str, list[str]] = {
    "python": [".py"],
    "javascript": [".js", ".jsx", ".mjs"],
    "typescript": [".ts", ".tsx"],
    "rust": [".rs"],
    "go": [".go"],
    "java": [".java"],
    "c": [".c", ".h"],
    "cpp": [".cpp", ".hpp", ".cc", ".cxx"],
    "ruby": [".rb"],
    "php": [".php"],
    "swift": [".swift"],
    "kotlin": [".kt"],
}

# All supported extensions (flat list)
ALL_SUPPORTED_EXTENSIONS: list[str] = []
for _exts in DEFAULT_EXTENSIONS.values():
    ALL_SUPPORTED_EXTENSIONS.extend(_exts)


# ---------------------------------------------------------------------------
# Data classes for code structure
# ---------------------------------------------------------------------------

@dataclass
class CodeElement:
    """A structural element extracted from code.

    Analogi: Dalam manual bela diri, setiap teknik punya nama,
    parameter (sikap tangan, kaki), dan hubungan dengan teknik lain.
    CodeElement = representasi formal dari elemen kode.

    Attributes:
        kind: The type of element (function, class, method, import, variable).
        name: The name of the element.
        signature: The signature (parameters, return type) if applicable.
        docstring: The docstring or comment, if any.
        parent: The parent element (e.g., class name for a method).
        children: Child elements (e.g., methods within a class).
        line_start: Starting line number in the source.
        line_end: Ending line number in the source.
        source: The source identifier (file path or snippet label).
    """

    kind: str  # "function", "class", "method", "import", "variable", "decorator"
    name: str = ""
    signature: str = ""
    docstring: str = ""
    parent: str = ""
    children: list[str] = field(default_factory=list)
    line_start: int = 0
    line_end: int = 0
    source: str = ""

    def to_ingest_text(self) -> str:
        """Format this element as text suitable for RSVS ingestion.

        The format is designed so that RSVS can extract meaningful
        relationships from the structural description.

        Analogi: Jin Soun tidak menghafal halaman buku mentah —
        dia membuat catatan terstruktur: "Tekik X, sikap Y,
        kelemahan Z". Format ini = catatan terstruktur untuk kode.
        """
        parts: list[str] = []

        # Core identity
        parts.append(f"{self.kind} {self.name}")

        # Signature / parameters
        if self.signature:
            parts.append(f"signature: {self.signature}")

        # Parent-child relationship
        if self.parent:
            parts.append(f"belongs_to: {self.parent}")
        if self.children:
            parts.append(f"contains: {', '.join(self.children)}")

        # Documentation
        if self.docstring:
            # Truncate long docstrings
            doc = self.docstring.strip()
            if len(doc) > 200:
                doc = doc[:200] + "..."
            parts.append(f"doc: {doc}")

        # Source location
        if self.source:
            parts.append(f"source: {self.source}")
        if self.line_start:
            parts.append(f"line: {self.line_start}-{self.line_end}")

        return ". ".join(parts) + "."

    def to_dict(self) -> dict:
        """Serialize to a plain dict."""
        return {
            "kind": self.kind,
            "name": self.name,
            "signature": self.signature,
            "docstring": self.docstring,
            "parent": self.parent,
            "children": list(self.children),
            "line_start": self.line_start,
            "line_end": self.line_end,
            "source": self.source,
        }


@dataclass
class CodeAnalysisResult:
    """The result of a code analysis.

    Analogi: Laporan analisis Jin Soun tentang teknik bela diri —
    bukan hanya "ini teknik X", tapi juga "kelemahannya Y",
    "hubungannya dengan teknik Z", dan "kemungkinan perbaikan W".

    Attributes:
        query: The original analysis query.
        elements_found: Code elements relevant to the query.
        similar_code: Structurally similar code elements.
        anomalies: Potential bugs/issues detected.
        patterns: Code patterns identified.
        suggestions: Improvement suggestions.
        confidence: Overall confidence of the analysis.
        evidence_nodes: Graph nodes used as evidence.
    """

    query: str
    elements_found: list[dict] = field(default_factory=list)
    similar_code: list[dict] = field(default_factory=list)
    anomalies: list[dict] = field(default_factory=list)
    patterns: list[dict] = field(default_factory=list)
    suggestions: list[str] = field(default_factory=list)
    confidence: float = 0.0
    evidence_nodes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        """Serialize to a plain dict."""
        return {
            "query": self.query,
            "elements_found": list(self.elements_found),
            "similar_code": list(self.similar_code),
            "anomalies": list(self.anomalies),
            "patterns": list(self.patterns),
            "suggestions": list(self.suggestions),
            "confidence": self.confidence,
            "evidence_nodes": list(self.evidence_nodes),
        }


# ---------------------------------------------------------------------------
# Call graph data structures
# ---------------------------------------------------------------------------

@dataclass
class CallGraph:
    """A directed call graph representing caller → callee relationships.

    The CoderLayer can parse code structure (functions, classes) but cannot
    reason about who-calls-whom. This data structure captures behavioral
    relationships: for each function/method, which other functions it calls.

    Attributes:
        edges: List of (caller, callee) pairs representing call relationships.
        nodes: Dict mapping function name → set of functions it calls.
        entry_points: Functions not called by anyone (potential entry points).
        leaf_functions: Functions that don't call anything (leaves of the graph).
    """

    edges: list[tuple[str, str]] = field(default_factory=list)
    nodes: dict[str, set[str]] = field(default_factory=dict)
    entry_points: list[str] = field(default_factory=list)
    leaf_functions: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        """Serialize to a plain dict."""
        return {
            "edges": list(self.edges),
            "nodes": {k: sorted(v) for k, v in self.nodes.items()},
            "entry_points": list(self.entry_points),
            "leaf_functions": list(self.leaf_functions),
        }


# ---------------------------------------------------------------------------
# Code parsing functions
# ---------------------------------------------------------------------------

def parse_python_code(code: str, source: str = "code_snippet") -> list[CodeElement]:
    """Parse Python code using the ast module.

    Extracts functions, classes, methods, imports, and top-level variables
    as structured CodeElement objects.

    Analogi: Jin Soun membaca manual dengan sabar — mengidentifikasi
    setiap teknik, parameternya, dan hubungan antar teknik.
    AST parser = pembacaan yang sama, tapi untuk kode.

    Args:
        code: Python source code string.
        source: Source identifier for the code.

    Returns:
        List of CodeElement objects representing the code structure.
    """
    elements: list[CodeElement] = []

    try:
        tree = ast.parse(code)
    except SyntaxError as exc:
        logger.warning("Failed to parse Python code: %s", exc)
        # Fallback: try regex-based extraction
        return _parse_code_regex(code, source, language="python")

    for node in ast.iter_child_nodes(tree):
        if isinstance(node, ast.FunctionDef) or isinstance(node, ast.AsyncFunctionDef):
            # Function definition
            sig = _extract_function_signature(node)
            docstring = ast.get_docstring(node) or ""
            children_names = [
                child.name for child in ast.walk(node)
                if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef))
                and child is not node
            ]

            element = CodeElement(
                kind="function",
                name=node.name,
                signature=sig,
                docstring=docstring,
                children=children_names,
                line_start=node.lineno,
                line_end=node.end_lineno or node.lineno,
                source=source,
            )
            elements.append(element)

        elif isinstance(node, ast.ClassDef):
            # Class definition
            bases = [ast.dump(base) for base in node.bases]
            bases_simple = []
            for base in node.bases:
                if isinstance(base, ast.Name):
                    bases_simple.append(base.id)
                elif isinstance(base, ast.Attribute):
                    bases_simple.append(ast.dump(base))

            sig = f"class({', '.join(bases_simple)})" if bases_simple else "class"
            docstring = ast.get_docstring(node) or ""

            # Extract methods and nested classes
            method_names: list[str] = []
            for child in node.body:
                if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    method_names.append(child.name)
                    # Also create a method element
                    method_sig = _extract_function_signature(child)
                    method_doc = ast.get_docstring(child) or ""
                    method_element = CodeElement(
                        kind="method",
                        name=child.name,
                        signature=method_sig,
                        docstring=method_doc,
                        parent=node.name,
                        line_start=child.lineno,
                        line_end=child.end_lineno or child.lineno,
                        source=source,
                    )
                    elements.append(method_element)

            element = CodeElement(
                kind="class",
                name=node.name,
                signature=sig,
                docstring=docstring,
                children=method_names,
                line_start=node.lineno,
                line_end=node.end_lineno or node.lineno,
                source=source,
            )
            elements.append(element)

        elif isinstance(node, ast.Import):
            # Import statement
            for alias in node.names:
                element = CodeElement(
                    kind="import",
                    name=alias.name,
                    signature=f"import {alias.name}" + (f" as {alias.asname}" if alias.asname else ""),
                    line_start=node.lineno,
                    line_end=node.lineno,
                    source=source,
                )
                elements.append(element)

        elif isinstance(node, ast.ImportFrom):
            # From...import statement
            module = node.module or ""
            names = ", ".join(
                alias.name + (f" as {alias.asname}" if alias.asname else "")
                for alias in node.names
            )
            element = CodeElement(
                kind="import",
                name=module,
                signature=f"from {module} import {names}",
                line_start=node.lineno,
                line_end=node.lineno,
                source=source,
            )
            elements.append(element)

        elif isinstance(node, ast.Assign):
            # Top-level variable assignment
            for target in node.targets:
                if isinstance(target, ast.Name):
                    element = CodeElement(
                        kind="variable",
                        name=target.id,
                        line_start=node.lineno,
                        line_end=node.end_lineno or node.lineno,
                        source=source,
                    )
                    elements.append(element)

    return elements


def _extract_function_signature(node: ast.FunctionDef | ast.AsyncFunctionDef) -> str:
    """Extract a function signature string from an AST node."""
    args: list[str] = []

    # Positional args
    for arg in node.args.args:
        arg_str = arg.arg
        if arg.annotation:
            try:
                annotation = ast.unparse(arg.annotation)  # type: ignore[attr-defined]
                arg_str += f": {annotation}"
            except Exception:
                pass
        args.append(arg_str)

    # *args
    if node.args.vararg:
        arg_str = f"*{node.args.vararg.arg}"
        if node.args.vararg.annotation:
            try:
                annotation = ast.unparse(node.args.vararg.annotation)  # type: ignore[attr-defined]
                arg_str += f": {annotation}"
            except Exception:
                pass
        args.append(arg_str)

    # Keyword-only args
    for arg in node.args.kwonlyargs:
        arg_str = arg.arg
        if arg.annotation:
            try:
                annotation = ast.unparse(arg.annotation)  # type: ignore[attr-defined]
                arg_str += f": {annotation}"
            except Exception:
                pass
        args.append(arg_str)

    # **kwargs
    if node.args.kwarg:
        arg_str = f"**{node.args.kwarg.arg}"
        if node.args.kwarg.annotation:
            try:
                annotation = ast.unparse(node.args.kwarg.annotation)  # type: ignore[attr-defined]
                arg_str += f": {annotation}"
            except Exception:
                pass
        args.append(arg_str)

    # Defaults for positional args
    defaults = node.args.defaults
    if defaults:
        n_required = len(node.args.args) - len(defaults)
        for i, default in enumerate(defaults):
            idx = n_required + i
            if idx < len(args):
                try:
                    default_str = ast.unparse(default)  # type: ignore[attr-defined]
                    args[idx] += f"={default_str}"
                except Exception:
                    pass

    # Return type
    ret = ""
    if node.returns:
        try:
            ret = f" -> {ast.unparse(node.returns)}"  # type: ignore[attr-defined]
        except Exception:
            pass

    return f"({', '.join(args)}){ret}"


# ---------------------------------------------------------------------------
# Regex-based code parsing (fallback for non-Python languages)
# ---------------------------------------------------------------------------

# Regex patterns for common languages
_FUNCTION_PATTERNS: list[tuple[str, re.Pattern]] = [
    ("python", re.compile(
        r"^\s*(?:async\s+)?def\s+(\w+)\s*\(([^)]*)\)",
        re.MULTILINE
    )),
    ("javascript", re.compile(
        r"(?:function\s+(\w+)\s*\(([^)]*)\)|(?:const|let|var)\s+(\w+)\s*=\s*(?:async\s+)?(?:\([^)]*\)|[^=])\s*=>)",
        re.MULTILINE
    )),
    ("rust", re.compile(
        r"(?:pub\s+)?(?:async\s+)?fn\s+(\w+)\s*(?:<[^>]*>)?\s*\(([^)]*)\)",
        re.MULTILINE
    )),
    ("go", re.compile(
        r"func\s+(?:\([^)]*\)\s*)?(\w+)\s*\(([^)]*)\)",
        re.MULTILINE
    )),
    ("java", re.compile(
        r"(?:public|private|protected)?\s*(?:static\s+)?(?:\w+(?:<[^>]*>)?)\s+(\w+)\s*\(([^)]*)\)",
        re.MULTILINE
    )),
]

_CLASS_PATTERNS: list[tuple[str, re.Pattern]] = [
    ("python", re.compile(r"^\s*class\s+(\w+)(?:\(([^)]*)\))?", re.MULTILINE)),
    ("javascript", re.compile(r"class\s+(\w+)(?:\s+extends\s+(\w+))?", re.MULTILINE)),
    ("rust", re.compile(r"(?:pub\s+)?struct\s+(\w+)", re.MULTILINE)),
    ("go", re.compile(r"type\s+(\w+)\s+struct", re.MULTILINE)),
    ("java", re.compile(r"(?:public\s+)?class\s+(\w+)(?:\s+extends\s+(\w+))?", re.MULTILINE)),
]

_IMPORT_PATTERNS: list[tuple[str, re.Pattern]] = [
    ("python", re.compile(r"^\s*(?:from\s+(\w[\w.]*)\s+)?import\s+(.+)$", re.MULTILINE)),
    ("javascript", re.compile(r"(?:import\s+.+\s+from\s+['\"]([^'\"]+)['\"]|require\s*\(\s*['\"]([^'\"]+)['\"]\s*\))", re.MULTILINE)),
    ("rust", re.compile(r"use\s+([\w:]+)", re.MULTILINE)),
    ("go", re.compile(r'import\s+(?:"([^"]+)"|\(([^)]+)\))', re.MULTILINE)),
    ("java", re.compile(r"import\s+([\w.]+)", re.MULTILINE)),
]


def _parse_code_regex(code: str, source: str = "code_snippet", language: str = "python") -> list[CodeElement]:
    """Parse code using regex patterns (fallback for non-Python or unparseable code).

    Analogi: Kalau Jin Soun tidak bisa membaca bahasa aslinya,
    dia tetap bisa mengenali pola dari tanda baca dan struktur.
    Regex = pengenalan pola untuk kode.

    Args:
        code: Source code string.
        source: Source identifier.
        language: Programming language identifier.

    Returns:
        List of CodeElement objects from regex extraction.
    """
    elements: list[CodeElement] = []

    # Extract functions
    for lang, pattern in _FUNCTION_PATTERNS:
        if lang == language or language == "auto":
            for match in pattern.finditer(code):
                name = match.group(1) or match.group(3) if match.lastindex and match.lastindex >= 3 else match.group(1)
                sig_group = 2 if match.lastindex and match.lastindex >= 2 else None
                sig = f"({match.group(sig_group)})" if sig_group and match.group(sig_group) else "()"

                # Approximate line number
                line_start = code[:match.start()].count("\n") + 1
                line_end = line_start

                element = CodeElement(
                    kind="function",
                    name=name or "anonymous",
                    signature=sig,
                    line_start=line_start,
                    line_end=line_end,
                    source=source,
                )
                elements.append(element)

    # Extract classes
    for lang, pattern in _CLASS_PATTERNS:
        if lang == language or language == "auto":
            for match in pattern.finditer(code):
                name = match.group(1)
                base = match.group(2) if match.lastindex and match.lastindex >= 2 else ""
                sig = f"extends {base}" if base else ""

                line_start = code[:match.start()].count("\n") + 1

                element = CodeElement(
                    kind="class",
                    name=name,
                    signature=sig,
                    line_start=line_start,
                    line_end=line_start,
                    source=source,
                )
                elements.append(element)

    # Extract imports
    for lang, pattern in _IMPORT_PATTERNS:
        if lang == language or language == "auto":
            for match in pattern.finditer(code):
                # Get the first non-None group
                name = ""
                for group in match.groups():
                    if group:
                        name = group.strip()
                        break

                if name:
                    line_start = code[:match.start()].count("\n") + 1
                    element = CodeElement(
                        kind="import",
                        name=name,
                        signature=match.group(0).strip(),
                        line_start=line_start,
                        line_end=line_start,
                        source=source,
                    )
                    elements.append(element)

    return elements


# ---------------------------------------------------------------------------
# Language detection
# ---------------------------------------------------------------------------

def detect_language(code: str, filename: str = "") -> str:
    """Detect the programming language of code.

    Uses file extension first, then falls back to heuristic detection.

    Args:
        code: The source code string.
        filename: Optional filename for extension-based detection.

    Returns:
        Language identifier string (e.g., "python", "javascript").
    """
    # Extension-based detection
    if filename:
        ext = os.path.splitext(filename)[1].lower()
        for lang, extensions in DEFAULT_EXTENSIONS.items():
            if ext in extensions:
                return lang

    # Heuristic detection
    # Python indicators
    if re.search(r"^\s*def\s+\w+\s*\(", code, re.MULTILINE):
        return "python"
    if re.search(r"^\s*import\s+\w+", code, re.MULTILINE) and re.search(r"^\s*from\s+\w+", code, re.MULTILINE):
        return "python"
    if re.search(r"^\s*class\s+\w+.*:", code, re.MULTILINE):
        return "python"

    # JavaScript indicators
    if re.search(r"\b(const|let|var)\s+\w+\s*=", code):
        return "javascript"
    if re.search(r"=>\s*[{(]", code):
        return "javascript"
    if re.search(r"require\s*\(", code):
        return "javascript"

    # Rust indicators
    if re.search(r"\bfn\s+\w+\s*[<(]", code):
        return "rust"
    if re.search(r"\blet\s+mut\b", code):
        return "rust"

    # Go indicators
    if re.search(r"\bfunc\s+\w+\s*\(", code):
        return "go"
    if re.search(r"\bpackage\s+\w+", code):
        return "go"

    # Java indicators
    if re.search(r"\bpublic\s+class\s+\w+", code):
        return "java"

    # Default
    return "unknown"


# ---------------------------------------------------------------------------
# Call graph extraction
# ---------------------------------------------------------------------------

def extract_call_graph(
    code: str,
    source: str = "code_snippet",
    language: str = "python",
) -> CallGraph:
    """Extract a call graph from source code.

    For Python: walks the AST to find Call nodes within each function/method.
    For other languages (Rust, Go, JavaScript): uses regex-based heuristics.

    The call graph captures behavioral relationships — who calls whom —
    complementing the structural layout that parse_python_code provides.

    Args:
        code: Source code string.
        source: Source identifier.
        language: Programming language (default "python").

    Returns:
        A CallGraph with caller → callee relationships.
    """
    if language == "python":
        return _extract_call_graph_python(code, source)
    else:
        return _extract_call_graph_regex(code, source, language)


def _extract_call_graph_python(code: str, source: str) -> CallGraph:
    """Extract call graph from Python code using AST analysis.

    Walks each function/method body, finds all ast.Call nodes,
    and records the callee name. Handles direct calls (foo()),
    method calls (obj.method()), and attribute calls (module.func()).
    """
    edges: list[tuple[str, str]] = []
    nodes: dict[str, set[str]] = {}

    try:
        tree = ast.parse(code)
    except SyntaxError as exc:
        logger.warning("Failed to parse Python code for call graph: %s", exc)
        return CallGraph()

    # Collect all function/method definitions and their call targets
    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue

        # Determine the caller name (include parent class if method)
        caller = node.name
        # Walk parents to find enclosing class
        for parent in ast.walk(tree):
            if isinstance(parent, ast.ClassDef):
                for child in parent.body:
                    if child is node:
                        caller = f"{parent.name}.{node.name}"
                        break

        callees: set[str] = set()

        # Walk the function body for Call nodes
        for child in ast.walk(node):
            if isinstance(child, ast.Call):
                callee_name = _resolve_call_name(child)
                if callee_name and callee_name != caller:
                    callees.add(callee_name)

        nodes[caller] = callees
        for callee in callees:
            edges.append((caller, callee))

    # Compute entry points and leaf functions
    all_callees = {callee for _, callee in edges}
    all_callers = set(nodes.keys())

    entry_points = sorted(all_callers - all_callees)
    leaf_functions = sorted({c for c, calls in nodes.items() if not calls})

    return CallGraph(
        edges=edges,
        nodes=nodes,
        entry_points=entry_points,
        leaf_functions=leaf_functions,
    )


def _resolve_call_name(call_node: ast.Call) -> str | None:
    """Resolve the name of a function being called from an ast.Call node.

    Handles:
        - Direct calls: foo()  → "foo"
        - Attribute calls: obj.method()  → "obj.method"
        - Module calls: module.func()  → "module.func"
    """
    func = call_node.func
    if isinstance(func, ast.Name):
        return func.id
    elif isinstance(func, ast.Attribute):
        # Build dotted name: obj.method, self.process, etc.
        parts = []
        current = func
        while isinstance(current, ast.Attribute):
            parts.append(current.attr)
            current = current.value
        if isinstance(current, ast.Name):
            parts.append(current.id)
        parts.reverse()
        return ".".join(parts)
    return None


# ---------------------------------------------------------------------------
# Regex-based call graph extraction for non-Python languages
# ---------------------------------------------------------------------------

# Patterns to find function definitions and their body calls
_CALL_PATTERNS: dict[str, tuple[re.Pattern, re.Pattern]] = {
    "javascript": (
        # Function definition
        re.compile(
            r"(?:function\s+(\w+)\s*\([^)]*\)|(?:const|let|var)\s+(\w+)\s*=\s*(?:async\s+)?(?:\([^)]*\))\s*=>)",
            re.MULTILINE,
        ),
        # Call within a function body (simplified: name followed by parens)
        re.compile(r"(?<!\w)(\w+)\s*\(", re.MULTILINE),
    ),
    "rust": (
        # Function definition
        re.compile(
            r"(?:pub\s+)?(?:async\s+)?fn\s+(\w+)\s*(?:<[^>]*>)?\s*\([^)]*\)",
            re.MULTILINE,
        ),
        # Call within a function body
        re.compile(r"(?<!\w)(\w+)\s*\(", re.MULTILINE),
    ),
    "go": (
        # Function definition
        re.compile(
            r"func\s+(?:\([^)]*\)\s*)?(\w+)\s*\([^)]*\)",
            re.MULTILINE,
        ),
        # Call within a function body
        re.compile(r"(?<!\w)(\w+)\s*\(", re.MULTILINE),
    ),
}

# Language-specific keywords to exclude from call targets
_LANG_KEYWORDS: dict[str, set[str]] = {
    "javascript": {
        "if", "else", "for", "while", "do", "switch", "case", "break",
        "continue", "return", "try", "catch", "finally", "throw", "new",
        "typeof", "instanceof", "in", "of", "function", "const", "let",
        "var", "class", "extends", "import", "export", "default", "from",
        "async", "await", "yield", "true", "false", "null", "undefined",
        "this", "super", "delete", "void",
    },
    "rust": {
        "if", "else", "for", "while", "loop", "match", "break",
        "continue", "return", "let", "mut", "fn", "struct", "enum",
        "impl", "trait", "pub", "use", "mod", "crate", "self", "super",
        "where", "as", "in", "ref", "move", "async", "await", "unsafe",
        "extern", "type", "const", "static", "true", "false", "Some",
        "None", "Ok", "Err", "Vec", "Box", "String", "println", "format",
        "eprintln", "panic", "assert", "unwrap", "expect", "clone",
        "new", "default", "from", "into", "to_string", "as_ref", "as_mut",
    },
    "go": {
        "if", "else", "for", "range", "switch", "case", "break",
        "continue", "return", "func", "struct", "interface", "map",
        "chan", "go", "select", "defer", "fallthrough", "default",
        "package", "import", "type", "var", "const", "true", "false",
        "nil", "make", "new", "len", "cap", "append", "copy", "delete",
        "close", "panic", "recover", "print", "println", "error",
        "string", "int", "float64", "bool", "byte", "rune",
    },
}


def _extract_call_graph_regex(
    code: str,
    source: str,
    language: str,
) -> CallGraph:
    """Extract call graph from non-Python code using regex heuristics.

    This is less precise than AST-based extraction but works for Rust,
    Go, and JavaScript where we don't have a full parser.

    Strategy:
    1. Find all function definitions using language-specific patterns.
    2. For each function, approximate its body region (between this def
       and the next, or end of file).
    3. Find all call-like patterns in the body region.
    4. Filter out language keywords and the function's own name.
    """
    if language not in _CALL_PATTERNS:
        return CallGraph()

    def_pattern, call_pattern = _CALL_PATTERNS[language]
    keywords = _LANG_KEYWORDS.get(language, set())

    edges: list[tuple[str, str]] = []
    nodes: dict[str, set[str]] = {}

    # Find all function definitions with their positions
    func_defs: list[tuple[str, int, int]] = []  # (name, start, end)
    for match in def_pattern.finditer(code):
        name = match.group(1) or (match.group(2) if match.lastindex and match.lastindex >= 2 else None)
        if name:
            start = match.start()
            func_defs.append((name, start, match.end()))

    # Approximate function body regions and find calls
    for i, (func_name, func_start, func_header_end) in enumerate(func_defs):
        # Body extends to the next function definition or end of file
        if i + 1 < len(func_defs):
            body_end = func_defs[i + 1][1]
        else:
            body_end = len(code)

        body = code[func_header_end:body_end]
        callees: set[str] = set()

        for call_match in call_pattern.finditer(body):
            callee = call_match.group(1)
            if callee and callee not in keywords and callee != func_name:
                callees.add(callee)

        nodes[func_name] = callees
        for callee in callees:
            edges.append((func_name, callee))

    # Compute entry points and leaf functions
    all_callees = {callee for _, callee in edges}
    all_callers = set(nodes.keys())

    entry_points = sorted(all_callers - all_callees)
    leaf_functions = sorted({c for c, calls in nodes.items() if not calls})

    return CallGraph(
        edges=edges,
        nodes=nodes,
        entry_points=entry_points,
        leaf_functions=leaf_functions,
    )


# ---------------------------------------------------------------------------
# CoderLayer — the main class
# ---------------------------------------------------------------------------

class CoderLayer:
    """Coder Layer — Code understanding as structured knowledge.

    Like ContextLayer but specialized for code. Ingests code as
    structured knowledge into the RSVS graph, then provides code-aware
    analysis, similarity search, and anomaly detection.

    Analogi: Jin Soun bisa membaca manual teknik bela diri dan
    mengerti kelemahannya dari struktur — bukan dari teks saja.
    Coder Layer melakukan hal yang sama untuk kode: memahami
    struktur, menemukan kelemahan (bug), dan menyarankan perbaikan.

    Code becomes knowledge:
    - Functions/methods → atoms in the graph
    - Classes → compositions (collections of methods)
    - Imports/dependencies → edges between nodes
    - Variables → context atoms

    Analysis pipeline (like PatternOutput but for code):
    1. TRIGGER — identify code concepts in the query
    2. RECALL — find related code in the graph
    3. CROSS-REFERENCE — compare code structures
    4. ANOMALY — detect potential bugs/issues
    5. PATTERN — identify code patterns and anti-patterns
    6. OUTPUT — structured code analysis report

    Attributes:
        rsvs_available: Whether a working RSVS instance is connected.
        is_rust_core: Whether the Rust core is being used.
    """

    def __init__(
        self,
        bridge: Optional[RsvsBridge] = None,
        web_search: Any = None,
    ) -> None:
        """Initialize the Coder Layer.

        Args:
            bridge: Optional pre-built RsvsBridge instance. If None,
                the layer will obtain a bridge via get_bridge().
            web_search: Optional WebSearchEngine instance (reserved
                for future use — searching for code solutions online).
        """
        if bridge is not None:
            self._bridge = bridge
        else:
            self._bridge = get_bridge()

        self.rsvs_available = self._bridge.is_available
        self.is_rust_core = self._bridge.is_rust_core

        # Reserved for future: web search for code solutions
        self._web_search = web_search

        # Ingestion log — tracks what code has been ingested
        # Analogi: Jin Soun mencatat manual mana yang sudah dibaca.
        self._ingestion_log: list[dict] = []

        # Code element registry — maps element names to their details
        # This is a local cache for quick lookups
        self._elements: dict[str, CodeElement] = {}

        # File registry — maps file paths to their elements
        self._files: dict[str, list[str]] = {}

        if self.rsvs_available:
            if self.is_rust_core:
                logger.info("CoderLayer initialized with RSVS Rust core")
            else:
                logger.info("CoderLayer initialized with RSVS fallback graph")
        else:
            logger.info("CoderLayer initialized WITHOUT RSVS core (fallback mode)")

    # ==================================================================
    # Code ingestion
    # ==================================================================

    def ingest_code(
        self,
        code: str,
        language: str = "python",
        source: str = "code_snippet",
    ) -> dict:
        """Ingest code into the RSVS graph.

        Parses code into structural elements, then ingests each element
        as structured text into the RSVS graph. This creates nodes for
        functions, classes, imports, and variables, with edges representing
        their relationships.

        Analogi: Jin Soun tidak menghafal buku mentah — dia
        mengidentifikasi setiap teknik, parameternya, hubungannya
        dengan teknik lain, LALU menyimpan pemahaman terstruktur itu.

        Args:
            code: Source code string to ingest.
            language: Programming language (default: "python").
                Use "auto" for automatic detection.
            source: Source identifier (e.g., file path or "code_snippet").

        Returns:
            A dict with keys:
                - "success": bool — whether ingestion succeeded
                - "source": str — the source identifier
                - "language": str — detected/specified language
                - "elements": list[dict] — parsed code elements
                - "element_count": int — number of elements parsed
                - "ingest_stats": dict | None — RSVS ingest stats
                - "trust": float — trust score for the source type
        """
        trust = CODE_SOURCE_TRUST.get(source, CODE_SOURCE_TRUST["unknown"])

        # Detect language if auto
        if language == "auto":
            language = detect_language(code, filename=source)

        # Parse code into structural elements
        if language == "python":
            elements = parse_python_code(code, source=source)
        else:
            elements = _parse_code_regex(code, source=source, language=language)

        # Ingest each element into RSVS
        # Analogi: Jin Soun menyimpan setiap teknik sebagai entri terpisah
        # di Simhyeon Pavilion — bukan sebagai satu teks panjang.
        all_stats: list[dict] = []
        element_names: list[str] = []

        for element in elements:
            ingest_text = element.to_ingest_text()

            # Register element locally
            element_key = f"{element.kind}:{element.name}"
            if element.parent:
                element_key = f"{element.parent}.{element_key}"
            self._elements[element_key] = element
            element_names.append(element.name)

            # Ingest into RSVS graph
            if self.rsvs_available:
                try:
                    stats = self._bridge.ingest(ingest_text)
                    all_stats.append(stats)
                except Exception as exc:
                    logger.error("RSVS ingest failed for element '%s': %s", element.name, exc)
                    all_stats.append({"success": False, "error": str(exc)})
            else:
                all_stats.append({
                    "atoms_before": 0,
                    "atoms_after": 0,
                    "new_atoms": 0,
                    "fallback": True,
                })

        # Also ingest relationships between elements
        # Analogi: Jin Soun tidak hanya menghafal teknik individual —
        # dia juga menghafal HUBUNGAN antar teknik.
        if element_names:
            relationship_text = (
                f"code structure: {', '.join(element_names[:20])}. "
                f"source: {source}. language: {language}."
            )
            if self.rsvs_available:
                try:
                    stats = self._bridge.ingest(relationship_text)
                    all_stats.append(stats)
                except Exception as exc:
                    logger.debug("RSVS ingest failed for relationship text: %s", exc)

        # Register file in the file registry
        if source not in self._files:
            self._files[source] = []
        self._files[source].extend(element_names)

        # Record in ingestion log
        record = {
            "success": True,
            "source": source,
            "language": language,
            "element_count": len(elements),
            "elements": [e.to_dict() for e in elements],
            "ingest_stats": all_stats,
            "trust": trust,
            "timestamp": time.time(),
            "code_length": len(code),
        }
        self._ingestion_log.append(record)

        logger.info(
            "CoderLayer.ingest_code(): %d elements from %s (%s)",
            len(elements), source, language,
        )

        return record

    def ingest_file(self, file_path: str) -> dict:
        """Ingest a code file from disk.

        Reads the file, detects the language, and ingests the code.

        Analogi: Jin Soun mengambil manual dari perpustakaan dan
        membacanya — satu buku pada satu waktu.

        Args:
            file_path: Path to the code file.

        Returns:
            A dict with ingestion results (same format as ingest_code).
        """
        try:
            with open(file_path, "r", encoding="utf-8", errors="replace") as f:
                code = f.read()
        except FileNotFoundError:
            logger.error("File not found: %s", file_path)
            return {
                "success": False,
                "source": file_path,
                "language": "unknown",
                "element_count": 0,
                "elements": [],
                "ingest_stats": None,
                "trust": 0.0,
                "error": "file_not_found",
            }
        except Exception as exc:
            logger.error("Failed to read file %s: %s", file_path, exc)
            return {
                "success": False,
                "source": file_path,
                "language": "unknown",
                "element_count": 0,
                "elements": [],
                "ingest_stats": None,
                "trust": 0.0,
                "error": str(exc),
            }

        # Detect language from file extension
        language = detect_language(code, filename=file_path)
        source = os.path.basename(file_path)

        return self.ingest_code(code, language=language, source=source)

    def ingest_directory(
        self,
        dir_path: str,
        extensions: list[str] | None = None,
    ) -> dict:
        """Ingest all code files in a directory.

        Recursively walks the directory and ingests all files with
        matching extensions.

        Analogi: Jin Soun membaca seluruh perpustakaan suatu sekte —
        bukan hanya satu buku, tapi semua manual yang tersedia.

        Args:
            dir_path: Path to the directory.
            extensions: List of file extensions to include (e.g., [".py"]).
                If None, all supported extensions are used.

        Returns:
            A dict with keys:
                - "success": bool
                - "directory": str
                - "files_processed": int
                - "total_elements": int
                - "results": list[dict] — per-file ingestion results
        """
        if extensions is None:
            extensions = ALL_SUPPORTED_EXTENSIONS

        results: list[dict] = []
        total_elements = 0

        for root, dirs, files in os.walk(dir_path):
            # Skip hidden directories and __pycache__
            dirs[:] = [
                d for d in dirs
                if not d.startswith(".") and d != "__pycache__" and d != "node_modules"
            ]

            for filename in files:
                ext = os.path.splitext(filename)[1].lower()
                if ext in extensions:
                    file_path = os.path.join(root, filename)
                    result = self.ingest_file(file_path)
                    results.append(result)
                    total_elements += result.get("element_count", 0)

        logger.info(
            "CoderLayer.ingest_directory(): %d files, %d elements from %s",
            len(results), total_elements, dir_path,
        )

        return {
            "success": True,
            "directory": dir_path,
            "files_processed": len(results),
            "total_elements": total_elements,
            "results": results,
        }

    # ==================================================================
    # Code analysis
    # ==================================================================

    def analyze_code(
        self,
        query: str,
        context: list[str] | None = None,
    ) -> CodeAnalysisResult:
        """Analyze code based on a query.

        Like PatternOutput but specialized for code:
        1. TRIGGER — identify code concepts in the query
        2. RECALL — find related code in the graph
        3. CROSS-REFERENCE — compare code structures
        4. ANOMALY — detect potential bugs/issues
        5. PATTERN — identify code patterns and anti-patterns
        6. OUTPUT — structured code analysis report

        Analogi: Jin Soun mendengar "teki jari racun" dan langsung:
        (1) mengidentifikasi itu sebagai teknik tertentu,
        (2) mengingat semua teknik serupa,
        (3) membandingkan struktur,
        (4) menemukan kelemahan,
        (5) melihat pola serangan,
        (6) menghasilkan analisis lengkap.

        Args:
            query: The analysis query (e.g., "find bugs in authenticate()",
                "how is User related to Database", "similar code to calculate_price").
            context: Optional list of context atoms (e.g., ["bug", "security"]).

        Returns:
            A CodeAnalysisResult with the full analysis.
        """
        context = context or []
        result = CodeAnalysisResult(query=query)
        evidence_nodes: list[str] = []
        confidence = 0.3

        # ---- Step 1: TRIGGER — identify code concepts in query ----
        # Analogi: Jin Soun mendengar pertanyaan dan mengidentifikasi
        # konsep kunci: nama teknik, jenis serangan, dll.
        trigger_concepts = self._extract_code_concepts(query)
        evidence_nodes.extend(trigger_concepts)

        # ---- Step 2: RECALL — find related code in graph ----
        # Analogi: Jin Soun mengaktifkan semua kenangan tentang
        # teknik yang disebutkan — spreading activation.
        activated_nodes: list[str] = []
        for concept in trigger_concepts:
            if self.rsvs_available:
                try:
                    relate_result = self._bridge.relate(concept)
                    if relate_result:
                        related_labels = self._parse_node_labels(relate_result)
                        activated_nodes.extend(related_labels)
                        evidence_nodes.extend(related_labels)
                        confidence = max(confidence, 0.5)
                except Exception as exc:
                    logger.debug("relate() failed for '%s': %s", concept, exc)

                # Also try direct query
                try:
                    query_result = self._bridge.query(concept)
                    if query_result:
                        parsed = self._parse_query_labels(query_result)
                        activated_nodes.extend(parsed)
                        evidence_nodes.extend(parsed)
                        confidence = max(confidence, 0.5)
                except Exception as exc:
                    logger.debug("query() failed for '%s': %s", concept, exc)

        # Also check local element registry
        for concept in trigger_concepts:
            for key, element in self._elements.items():
                if concept.lower() in element.name.lower() or concept.lower() in key.lower():
                    result.elements_found.append(element.to_dict())
                    activated_nodes.append(element.name)
                    confidence = max(confidence, 0.6)

        # ---- Step 3: CROSS-REFERENCE — compare code structures ----
        # Analogi: Jin Soun membandingkan teknik yang diingat —
        # mana yang mirip, mana yang berbeda, apa yang bisa dipelajari.
        seen_pairs: set[tuple[str, str]] = set()
        for i, node_a in enumerate(activated_nodes[:15]):
            for node_b in activated_nodes[i + 1:15]:
                pair = tuple(sorted([node_a, node_b]))
                if pair in seen_pairs:
                    continue
                seen_pairs.add(pair)

                if self.rsvs_available:
                    try:
                        sim = self._bridge.structural_similarity(node_a, node_b)
                        if sim:
                            sim_value = sim.get("structural_similarity", 0.0) if isinstance(sim, dict) else 0.0
                            if sim_value > 0.1:
                                result.similar_code.append({
                                    "node_a": node_a,
                                    "node_b": node_b,
                                    "similarity": sim_value,
                                    "shared": sim.get("shared", []),
                                })
                                evidence_nodes.extend([node_a, node_b])
                                confidence = max(confidence, 0.65)
                    except Exception:
                        pass

        # ---- Step 4: ANOMALY — detect potential bugs/issues ----
        # Analogi: Jin Soun memeriksa — "Jika teknik ini benar,
        # seharusnya hasilnya A. Tapi hasilnya B. ANOMALI!"
        for concept in trigger_concepts[:5]:
            if self.rsvs_available:
                # Appraise a statement about the code
                statement = f"{concept} is correctly implemented without bugs"
                try:
                    appraise_result = self._bridge.appraise(statement)
                    if isinstance(appraise_result, dict):
                        verdict = appraise_result.get("verdict", "neutral")
                        if verdict == "disagree" or appraise_result.get("disagree_pct", 0) > 0.3:
                            result.anomalies.append({
                                "type": "code_anomaly",
                                "concept": concept,
                                "verdict": verdict,
                                "disagree_pct": appraise_result.get("disagree_pct", 0.0),
                                "description": (
                                    f"Code element '{concept}' may have issues: "
                                    f"appraise verdict={verdict}, "
                                    f"disagreement={appraise_result.get('disagree_pct', 0):.1%}"
                                ),
                            })
                            evidence_nodes.append(concept)
                            confidence = max(confidence, 0.7)
                except Exception as exc:
                    logger.debug("appraise() failed for '%s': %s", statement, exc)

        # Check for common code issues using local analysis
        for key, element in self._elements.items():
            if element.kind in ("function", "method"):
                # Functions with no docstring might need documentation
                if not element.docstring and element.kind == "function":
                    result.suggestions.append(
                        f"Function '{element.name}' lacks documentation. "
                        f"Consider adding a docstring."
                    )

        # ---- Step 5: PATTERN — identify code patterns ----
        # Analogi: Jin Soun melihat pola dari semua teknik yang dianalisis.
        if result.similar_code:
            result.patterns.append({
                "type": "structural_similarity",
                "description": (
                    f"Found {len(result.similar_code)} structurally similar "
                    f"code pairs, suggesting shared implementation patterns."
                ),
                "pair_count": len(result.similar_code),
            })
            confidence = max(confidence, 0.7)

        if result.anomalies:
            result.patterns.append({
                "type": "anomaly_pattern",
                "description": (
                    f"Found {len(result.anomalies)} code anomalies, "
                    f"suggesting potential bugs or structural issues."
                ),
                "anomaly_count": len(result.anomalies),
            })
            confidence = max(confidence, 0.75)

        # ---- Step 6: OUTPUT — finalize ----
        result.evidence_nodes = list(dict.fromkeys(evidence_nodes))
        result.confidence = min(0.95, confidence)

        logger.info(
            "CoderLayer.analyze_code(): query='%s', found=%d, similar=%d, "
            "anomalies=%d, patterns=%d, confidence=%.3f",
            query[:50], len(result.elements_found),
            len(result.similar_code), len(result.anomalies),
            len(result.patterns), result.confidence,
        )

        return result

    def find_similar_code(
        self,
        code_snippet: str,
        top_k: int = 5,
    ) -> list[dict]:
        """Find structurally similar code in the graph.

        Parses the input code, then uses RSVS structural_similarity
        to find matching code elements already in the graph.

        Analogi: Jin Soun melihat teknik baru dan langsung membandingkan
        dengan semua teknik yang pernah dilihat — "ini mirip dengan
        Teknik X dari Sekolah Y, tapi dengan variasi Z".

        Args:
            code_snippet: The code to search for similar code.
            top_k: Maximum number of results to return.

        Returns:
            A list of dicts, each with:
                - "name": str — the matching code element name
                - "kind": str — the type of element
                - "similarity": float — structural similarity score
                - "shared": list — shared structural elements
        """
        # Parse the input code
        language = detect_language(code_snippet)
        if language == "python":
            elements = parse_python_code(code_snippet, source="query")
        else:
            elements = _parse_code_regex(code_snippet, source="query", language=language)

        if not elements:
            return []

        # For each element in the input, find similar elements in the graph
        results: list[dict] = []
        seen_names: set[str] = set()

        for element in elements:
            # Check local registry
            for key, existing in self._elements.items():
                if existing.name in seen_names:
                    continue

                if self.rsvs_available:
                    try:
                        sim = self._bridge.structural_similarity(element.name, existing.name)
                        if sim and isinstance(sim, dict):
                            sim_value = sim.get("structural_similarity", 0.0)
                            if sim_value > 0.05:
                                results.append({
                                    "name": existing.name,
                                    "kind": existing.kind,
                                    "similarity": sim_value,
                                    "shared": sim.get("shared", []),
                                    "source": existing.source,
                                })
                                seen_names.add(existing.name)
                    except Exception:
                        pass

            # Also try relate() to find connected nodes
            if self.rsvs_available:
                try:
                    relate_result = self._bridge.relate(element.name)
                    if relate_result:
                        labels = self._parse_node_labels(relate_result)
                        for label in labels[:top_k]:
                            if label not in seen_names:
                                # Check if this is a known code element
                                matching = self._elements.get(label)
                                if matching:
                                    results.append({
                                        "name": matching.name,
                                        "kind": matching.kind,
                                        "similarity": 0.5,  # Default for related
                                        "shared": [],
                                        "source": matching.source,
                                    })
                                    seen_names.add(label)
                except Exception:
                    pass

        # Sort by similarity and return top_k
        results.sort(key=lambda r: r["similarity"], reverse=True)
        return results[:top_k]

    def detect_code_anomalies(self) -> list[dict]:
        """Detect potential bugs/issues using RSVS anomaly detection.

        Uses appraise() on code structure statements to find
        contradictions, missing implementations, and potential bugs.

        Analogi: Jin Soun memeriksa semua teknik yang diketahui —
        "Teknik X seharusnya punya kontra Y, tapi tidak ada.
        Teknik Z punya parameter yang tidak pernah digunakan. ANOMALI!"

        Returns:
            A list of dicts, each with:
                - "type": str — anomaly type
                "element": str — the affected code element
                - "description": str — human-readable description
                - "verdict": str — appraise verdict
                - "confidence": float — confidence of the anomaly
        """
        anomalies: list[dict] = []

        for key, element in self._elements.items():
            # Strategy 1: Check if element's implementation seems complete
            if element.kind in ("function", "method"):
                # Functions with empty signatures might be stubs
                if element.signature == "()" or element.signature.strip() == "()":
                    statement = f"{element.name} is a complete implementation"
                    if self.rsvs_available:
                        try:
                            appraise_result = self._bridge.appraise(statement)
                            if isinstance(appraise_result, dict):
                                verdict = appraise_result.get("verdict", "neutral")
                                if verdict != "agree":
                                    anomalies.append({
                                        "type": "potential_stub",
                                        "element": element.name,
                                        "description": (
                                            f"Function '{element.name}' has no parameters — "
                                            f"it may be a stub or incomplete implementation."
                                        ),
                                        "verdict": verdict,
                                        "confidence": appraise_result.get("disagree_pct", 0.3),
                                    })
                        except Exception:
                            pass

            # Strategy 2: Check class consistency
            if element.kind == "class":
                # Classes without methods might be data-only or incomplete
                if not element.children:
                    statement = f"{element.name} class has complete functionality"
                    if self.rsvs_available:
                        try:
                            appraise_result = self._bridge.appraise(statement)
                            if isinstance(appraise_result, dict):
                                verdict = appraise_result.get("verdict", "neutral")
                                if verdict != "agree":
                                    anomalies.append({
                                        "type": "incomplete_class",
                                        "element": element.name,
                                        "description": (
                                            f"Class '{element.name}' has no methods — "
                                            f"it may be a data class or incomplete abstraction."
                                        ),
                                        "verdict": verdict,
                                        "confidence": appraise_result.get("disagree_pct", 0.2),
                                    })
                        except Exception:
                            pass

            # Strategy 3: Check for missing docstrings (informational)
            if element.kind in ("function", "method", "class"):
                if not element.docstring:
                    anomalies.append({
                        "type": "missing_documentation",
                        "element": element.name,
                        "description": (
                            f"{element.kind.capitalize()} '{element.name}' "
                            f"lacks documentation (docstring)."
                        ),
                        "verdict": "neutral",
                        "confidence": 0.3,
                    })

        logger.info(
            "CoderLayer.detect_code_anomalies(): %d anomalies from %d elements",
            len(anomalies), len(self._elements),
        )

        return anomalies

    def get_code_summary(self, file_path: str | None = None) -> dict:
        """Get a summary of ingested code.

        If file_path is given, returns a summary of that specific file.
        Otherwise, returns a summary of all ingested code.

        Analogi: Jin Soun merangkum semua yang dibacanya —
        "Dari 10 manual, ada 50 teknik, 5 di antaranya tidak lengkap."

        Args:
            file_path: Optional file path to summarize.

        Returns:
            A dict with:
                - "total_elements": int
                - "by_kind": dict — count of elements by type
                - "by_source": dict — count of elements by source
                - "files": list[str] — list of ingested files
                - "elements": list[dict] — all elements (or file-specific)
        """
        elements = self._elements.values()

        if file_path:
            # Filter to specific file
            source_name = os.path.basename(file_path)
            file_element_keys = self._files.get(source_name, [])
            elements = [
                e for e in elements
                if e.name in file_element_keys or e.source == source_name
            ]

        by_kind: dict[str, int] = {}
        by_source: dict[str, int] = {}
        element_list: list[dict] = []

        for element in elements:
            by_kind[element.kind] = by_kind.get(element.kind, 0) + 1
            by_source[element.source] = by_source.get(element.source, 0) + 1
            element_list.append(element.to_dict())

        return {
            "total_elements": len(element_list),
            "by_kind": by_kind,
            "by_source": by_source,
            "files": list(self._files.keys()),
            "elements": element_list,
        }

    # ==================================================================
    # Call graph extraction
    # ==================================================================

    def extract_call_graph(
        self,
        code: str,
        language: str = "python",
        source: str = "code_snippet",
    ) -> CallGraph:
        """Extract and ingest a call graph from source code.

        Parses code to extract caller → callee relationships, then ingests
        those relationships into the RSVS graph as "calls" composition edges.
        This gives the CoderLayer behavioral understanding beyond structural
        layout — not just "what functions exist" but "who calls whom".

        Analogi: Jin Soun tidak hanya menghafal teknik dari buku — dia juga
        memahami URUTAN teknik mana yang mengarah ke teknik lain.

        Args:
            code: Source code string.
            language: Programming language (default "python").
                Use "auto" for automatic detection.
            source: Source identifier.

        Returns:
            A CallGraph object with caller → callee relationships.
        """
        # Detect language if auto
        if language == "auto":
            language = detect_language(code, filename=source)

        # Extract the call graph
        graph = extract_call_graph(code, source=source, language=language)

        # Ingest call relationships into RSVS as "calls" composition type
        if self.rsvs_available and graph.edges:
            for caller, callee in graph.edges:
                call_text = f"{caller} calls {callee}"
                try:
                    self._bridge.ingest(call_text)
                except Exception as exc:
                    logger.debug(
                        "RSVS ingest failed for call edge '%s → %s': %s",
                        caller, callee, exc,
                    )

            # Also ingest a summary of the call graph structure
            summary_parts = []
            if graph.entry_points:
                summary_parts.append(f"entry_points: {', '.join(graph.entry_points[:10])}")
            if graph.leaf_functions:
                summary_parts.append(f"leaf_functions: {', '.join(graph.leaf_functions[:10])}")
            if summary_parts:
                summary_text = f"call_graph: {'; '.join(summary_parts)}. source: {source}"
                try:
                    self._bridge.ingest(summary_text)
                except Exception as exc:
                    logger.debug("RSVS ingest failed for call graph summary: %s", exc)

        logger.info(
            "CoderLayer.extract_call_graph(): %d edges, %d nodes from %s (%s)",
            len(graph.edges), len(graph.nodes), source, language,
        )

        return graph

    # ==================================================================
    # Internal helpers
    # ==================================================================

    def _extract_code_concepts(self, text: str) -> list[str]:
        """Extract code-related concepts from a query string.

        Identifies function names, class names, and code-related terms
        from the query text.

        Args:
            text: The query text.

        Returns:
            List of extracted concept strings.
        """
        concepts: list[str] = []

        # Check for known element names in the query
        for key, element in self._elements.items():
            if element.name in text:
                concepts.append(element.name)
            # Also check for class.method patterns
            if element.parent and f"{element.parent}.{element.name}" in text:
                concepts.append(element.name)
                concepts.append(element.parent)

        # Extract quoted strings (likely code identifiers)
        for match in re.finditer(r'["\'](\w+)["\']', text):
            concepts.append(match.group(1))

        # Extract function-like identifiers (word followed by parens)
        for match in re.finditer(r'\b(\w+)\s*\(\)', text):
            concepts.append(match.group(1))

        # Extract CamelCase names (likely class names)
        for match in re.finditer(r'\b([A-Z][a-z]+(?:[A-Z][a-z]+)*)\b', text):
            concepts.append(match.group(1))

        # Extract snake_case names (likely function/variable names)
        for match in re.finditer(r'\b([a-z][a-z0-9]*(?:_[a-z0-9]+)+)\b', text):
            concepts.append(match.group(1))

        # If no concepts found, fall back to keyword extraction
        if not concepts:
            # Simple keyword extraction from the query
            stop_words = {
                "find", "how", "what", "where", "why", "is", "are",
                "the", "a", "an", "in", "on", "at", "to", "for",
                "bugs", "similar", "code", "related", "analyze",
                "check", "detect", "does", "can", "has", "have",
            }
            words = text.lower().replace(",", " ").replace("?", " ").split()
            concepts = [w for w in words if len(w) > 2 and w not in stop_words][:10]

        # Deduplicate while preserving order
        seen: set[str] = set()
        unique_concepts: list[str] = []
        for c in concepts:
            if c.lower() not in seen:
                seen.add(c.lower())
                unique_concepts.append(c)

        return unique_concepts

    def _parse_node_labels(self, relate_result: dict) -> list[str]:
        """Extract node labels from a relate() result."""
        labels: list[str] = []

        # Related nodes may be (label, score) tuples or (id, score) tuples
        related_nodes = relate_result.get("related_nodes", [])
        for item in related_nodes:
            if isinstance(item, tuple) and len(item) >= 1:
                label = item[0]
                if isinstance(label, str) and not label.startswith("_"):
                    labels.append(label)
            elif isinstance(item, str):
                labels.append(item)

        # Structural relations
        structural = relate_result.get("structural_relations", [])
        for item in structural:
            if isinstance(item, tuple) and len(item) >= 1:
                label = item[0]
                if isinstance(label, str):
                    labels.append(label)
            elif isinstance(item, str):
                labels.append(item)

        return labels

    def _parse_query_labels(self, query_result: dict) -> list[str]:
        """Extract concept labels from a query() result."""
        labels: list[str] = []

        # Atoms may be (label, score) tuples
        atoms = query_result.get("atoms", [])
        for item in atoms:
            if isinstance(item, tuple) and len(item) >= 1:
                label = item[0]
                if isinstance(label, str):
                    labels.append(label)
            elif isinstance(item, str):
                labels.append(item)

        # Compositions
        compositions = query_result.get("compositions", [])
        for item in compositions:
            if isinstance(item, tuple) and len(item) >= 1:
                label = item[0]
                if isinstance(label, str):
                    labels.append(label)
            elif isinstance(item, str):
                labels.append(item)

        return labels
