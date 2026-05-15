# RSVS Semantic Parsing Research Direction

## Objective

Design a **non-LLM semantic ingestion pipeline** for RSVS that can
transform raw natural language text into structured graph
representations suitable for compositional reasoning.

Core requirement:

-   No LLM intervention
-   Deterministic / auditable
-   Compatible with RSVS compositional graph architecture
-   Preserve semantic roles, relations, causality, and nested meaning

------------------------------------------------------------------------

## Problem Statement

A naive token/co-occurrence ingest pipeline loses structural meaning.

Example:

**Input:**

Raymond membuat aplikasi untuk kantor karena proses manual terlalu
lambat.

Naive token ingest:

``` text
Raymond, membuat, aplikasi, kantor, proses, manual, lambat
```

This loses:

-   who did the action
-   what received the action
-   why the action happened
-   purpose of the action
-   clause structure

RSVS can reason over relations, but only if relations are represented
structurally.

------------------------------------------------------------------------

## Research-Backed Architectural Direction

The most elegant architecture is:

``` text
Universal Dependencies
→ Semantic Role Labeling
→ AMR-style semantic graph
→ RSVS ingestion
```

------------------------------------------------------------------------

## 1. Universal Dependencies (UD)

Purpose:

Recover grammatical structure.

Detect:

-   nominal subject (`nsubj`)
-   object (`obj`)
-   oblique modifiers (`obl`)
-   case markers
-   dependency relations

Example:

``` text
Raymond membuat aplikasi
```

UD representation:

``` text
nsubj(membuat, Raymond)
obj(membuat, aplikasi)
```

This answers:

-   who is grammatical subject?
-   what is predicate?
-   what is grammatical object?

UD provides syntax skeleton.

However:

UD alone is insufficient because syntax ≠ semantic role.

Example:

``` text
Aplikasi dibuat oleh Raymond.
```

Grammatical subject:

``` text
aplikasi
```

Actual agent:

``` text
Raymond
```

Therefore syntax alone is not enough.

------------------------------------------------------------------------

## 2. Semantic Role Labeling (SRL / PropBank-style)

Purpose:

Recover semantic meaning roles.

Canonical roles:

-   ARG0 → agent / doer
-   ARG1 → patient / theme
-   ARG2 → beneficiary / attribute / instrument (verb-dependent)
-   TMP → temporal
-   LOC → location
-   CAU → cause
-   PNC → purpose
-   NEG → negation

Example:

Input:

``` text
Raymond membuat aplikasi untuk kantor karena proses manual terlalu lambat.
```

SRL frame:

``` text
Predicate: membuat
ARG0: Raymond
ARG1: aplikasi
PNC: kantor
CAU: proses manual terlalu lambat
```

This matches RSVS far better than subject-object parsing.

Because RSVS needs meaning roles, not only grammar.

------------------------------------------------------------------------

## 3. Abstract Meaning Representation (AMR)

Most structurally aligned with RSVS.

AMR transforms text into semantic graph.

Example:

``` text
The boy wants to go.
```

AMR-like structure:

``` text
want-01
  ARG0 → boy
  ARG1 → go-01
           ARG0 → boy
```

Meaning:

-   events become nodes
-   roles become edges
-   nested actions become nested graph structures

This is conceptually very close to RSVS.

------------------------------------------------------------------------

## Why OpenIE Alone Is Not Enough

Open Information Extraction usually yields:

``` text
(subject, relation, object)
```

Example:

``` text
(Raymond, membuat, aplikasi)
```

Useful but incomplete.

Missing:

-   cause
-   purpose
-   location
-   time
-   polarity
-   nested clauses
-   discourse relations

Therefore OpenIE can be a helper, but not the final ingest
representation.

------------------------------------------------------------------------

## Recommended RSVS Architecture

Final pipeline:

``` text
Raw Text
→ tokenizer
→ clause segmentation
→ dependency parsing
→ semantic role labeling
→ AMR-style event graph construction
→ RSVS compositional ingest
```

------------------------------------------------------------------------

## Target Structured Representation

Input:

``` text
Raymond membuat aplikasi untuk kantor karena proses manual terlalu lambat.
```

Structured output:

``` json
{
  "event_id": "e1",
  "predicate": "membuat",
  "ARG0_agent": "Raymond",
  "ARG1_patient": "aplikasi",
  "PNC_purpose": "kantor",
  "CAU_cause": "proses manual terlalu lambat",
  "polarity": "positive",
  "voice": "active"
}
```

------------------------------------------------------------------------

## RSVS Graph Mapping

Transform to graph:

``` text
event_e1
  predicate → membuat
  ARG0      → Raymond
  ARG1      → aplikasi
  PNC       → kantor
  CAU       → proses_manual_terlalu_lambat
```

Equivalent typed edges:

``` text
event_e1 --predicate--> membuat
event_e1 --ARG0--> Raymond
event_e1 --ARG1--> aplikasi
event_e1 --PNC--> kantor
event_e1 --CAU--> proses_manual_terlalu_lambat
```

------------------------------------------------------------------------

## Why This Fits RSVS

RSVS is compositional.

Meaning should not be ingested as flat tokens.

Bad:

``` text
{Raymond, membuat, aplikasi, kantor}
```

Good:

``` text
event composition:
predicate = membuat
agent = Raymond
patient = aplikasi
purpose = kantor
cause = proses_manual_lambat
```

This enables:

-   structural similarity
-   substitution analysis
-   grounded reasoning
-   contradiction checks
-   sense induction over relational meaning

------------------------------------------------------------------------

## Naming Suggestion

Internal subsystem name:

``` text
RSVS Semantic Frame Compiler
```

Responsibility:

Convert natural language into deterministic semantic event graphs.

Not "AI understanding".

Instead:

structured semantic compilation.

------------------------------------------------------------------------

## Final Recommendation

Best research-aligned architecture:

``` text
Universal Dependencies
+ Semantic Role Labeling
+ AMR-style semantic graph compilation
+ RSVS compositional ingestion
```

This is the most elegant non-LLM path.
