"""Provenance certificate for a computation.

Every result may be accompanied by a :class:`Certificate` recording exactly how
it was produced. In ``float`` mode ``error_bound`` is ``None`` (discovery-tier,
no guarantee); ``certified`` mode (Phase 2) will populate it with a rigorous
interval bound. The certificate is the trust artifact: it is meant to be
serialized to JSON, shared, and independently replayed — status is never a
producer-declared PASS.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from typing import Any


@dataclass
class Certificate:
    result: Any
    mode: str = "float"
    error_bound: float | None = None  # None in float mode: no guarantee
    backend: str = "numpy"
    chi: int | None = None            # bond dimension / truncation (the regulator)
    seed: int | None = None
    htf_version: str = "0.2.0"
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
