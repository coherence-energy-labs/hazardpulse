"""Candidate C's causal per-cell features -- the implementation lives in
``hazardpulse.earthquake.operational_features`` (moved there unchanged after the CHOOSE
decision selected C0, so the served model and the evaluation share one copy). The code
the evaluation ran is byte-identical to that module's feature functions; see
docs/EARTHQUAKE_FORECAST_PROGRAM.md section 9.
"""

from hazardpulse.earthquake.operational_features import (  # noqa: F401
    CORE_FEATURES,
    NO_EVENT_DAYS,
    SEC_DAY,
    YEAR,
    CellCatalog,
    _last_event,
    active_cells,
    cell_catalog_from_arrays,
    core_features,
    nbr_reduce,
)
