"""Provenance certificate for a computation.

Every result may be accompanied by a :class:`Certificate` recording exactly how
it was produced. In ``float`` mode ``error_bound`` is ``None`` (discovery-tier,
no guarantee); ``certified`` mode (Phase 2) will populate it with a rigorous
interval bound.

Note: the current Certificate is result *metadata*, not a fully replayable
proof artifact.  It does not record the exact claim, theorem premises, canonical
input digest, or verifier result.  Certificate-v1 (with claim IR, input hash,
interval endpoints, and independent verifier) is a planned deliverable (P0-7).
Status is never a producer-declared PASS.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from typing import Any

try:
    from importlib.metadata import version as _pkg_version
    _HTF_VERSION: str = _pkg_version("htf")
except Exception:
    _HTF_VERSION = "unknown"


@dataclass
class Certificate:
    result: Any
    mode: str = "float"
    error_bound: float | None = None  # None in float mode: no guarantee
    backend: str = "numpy"
    chi: int | None = None            # bond dimension / truncation (the regulator)
    seed: int | None = None
    htf_version: str = _HTF_VERSION
    notes: str = ""

    def to_dict(self) -> dict:
        d = asdict(self)
        try:
            import numpy as np

            if isinstance(self.result, np.ndarray):
                d["result"] = self.result.tolist()
            elif isinstance(self.result, (np.floating, np.integer)):
                d["result"] = self.result.item()
        except Exception:
            pass
        return d

    def to_json(self, **kwargs) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, **kwargs)
