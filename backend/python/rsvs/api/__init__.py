"""RSVS API package — modular route architecture.

Splits the monolithic fastapi_server.py into:
  - schemas:   Pydantic request/response models
  - deps:      Shared dependencies (auth, rate limiter, get_rsvs_instance)
  - middleware: Request size limit
  - routes/core:         Core CRUD endpoints (run, ingest, query, etc.)
  - routes/analysis:     Structural similarity & context endpoints
  - routes/maintenance:  Consolidation, reflection, domain attention, etc.
"""
