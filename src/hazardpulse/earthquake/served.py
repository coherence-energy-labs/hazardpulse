"""Which stack on C0 is the published earthquake forecast (docs/EARTHQUAKE_FORECAST_PROGRAM.md sections 10 and 12).

Named once, here. The live scorer (``scripts/fetch_and_score_earthquake.py``), the served-model evidence
(``hazardpulse.verification.served_evidence``) and the model registry (``scripts/publish_model_registry.py``) all
read this, so what is served and what is described cannot point at different files. ``None`` means C0 alone is
served. A path that names a missing file is an error everywhere: it never falls back to C0.
"""
from __future__ import annotations

SERVED_STACK_RELPATH: str | None = "results/models/earthquake_gear1_stack_v1.json"
