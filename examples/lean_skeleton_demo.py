"""Lean 4 proof-skeleton export demo.

Run:  PYTHONPATH=. python examples/lean_skeleton_demo.py

Demonstrates:
* Build a certified variational energy bound for TFIM.
* Compute gap bounds via Temple's inequality.
* Export a Lean 4 skeleton: valid syntax + sorry stubs for all theorems.
* Inspect the generated Lean 4 source.

Honest scope [研究]
-------------------
* The generated Lean 4 file is syntactically valid (Lean will accept it with
  import Mathlib) but all theorem proofs are sorry stubs.
* Completing the proofs requires a human Lean expert; full formalisation is [研究].
* HTF does not invoke the Lean server; use `lake build` externally to type-check.
* Certified bounds cover FP rounding only; bond-dimension bias is [OUT].
"""
import tempfile
from pathlib import Path

import numpy as np

from htf.certificate import Certificate
from htf.gap import gap_report
from htf.lean_export import (
    LeanExporter,
    certificate_to_lean,
    gap_report_to_lean,
    structure_report_to_lean,
)
from htf.mera import random_mera
from htf.structure import check_isometry
from htf.variational import optimize_mera, transverse_ising_ham, variational_bound

# ── Setup ──────────────────────────────────────────────────────────────────
n  = 4
H  = transverse_ising_ham(n=n, J=1.0, h=0.5)
E0 = float(np.linalg.eigvalsh(H)[0])

mera0     = random_mera(n, chi=2, seed=0)
mera_opt, _ = optimize_mera(H, mera0, n_iter=50, tol=1e-6)
cert        = variational_bound(H, mera_opt)

psi_gs = mera_opt.state_vector()
psi_es = random_mera(n, chi=2, seed=1).state_vector()
greport = gap_report(H, psi_gs, psi_es)

iso_rep = check_isometry(mera_opt.layers[0].disentanglers[0])

print(f"Certified E0 upper bound: {cert.result:.8f} ± {cert.error_bound:.2e}")
print(f"Exact E0:                 {E0:.8f}")
print(f"Temple lower bound:       {greport['temple_lb']:.8f}")
print(f"Gap exact:                {greport['gap_exact']:.8f}")

# ── Build Lean exporter ────────────────────────────────────────────────────
exp = LeanExporter(
    preamble=f"HTF certified results for {n}-site TFIM (J=1, h=0.5)"
)
exp.add_certificate(cert, "variational_E0")
exp.add_gap_report(greport, "spectral_gap")
exp.add_structure_report(iso_rep, "isometry_check")

# ── Write to a temp file and inspect ─────────────────────────────────────
with tempfile.NamedTemporaryFile(suffix=".lean", mode="w",
                                 delete=False, encoding="utf-8") as f:
    lean_path = Path(f.name)

exp.write(lean_path)
lean_src = lean_path.read_text(encoding="utf-8")

print(f"\nGenerated Lean 4 file: {lean_path}")
print(f"  Lines: {lean_src.count(chr(10))}")
print(f"  sorry stubs: {lean_src.count('sorry')}")
print(f"  theorem statements: {lean_src.count('theorem ')}")

print("\n─── Lean 4 source (first 60 lines) ───")
for i, line in enumerate(lean_src.splitlines()[:60], 1):
    print(f"{i:3d} │ {line}")

# ── Individual snippet helpers ─────────────────────────────────────────────
print("\n─── certificate_to_lean snippet ───")
print(certificate_to_lean(cert, "energy_upper"))

print("\n─── gap_report_to_lean snippet ───")
print(gap_report_to_lean(greport, "ising_gap"))

# ── Honest scope reminder ─────────────────────────────────────────────────
print("\nHonest scope:")
print("  [工程] Generated Lean 4 syntax is valid (loads with import Mathlib).")
print("  [研究] Completing sorry stubs requires a human Lean expert.")
print("  [OUT]  Does not produce a formal proof of continuum properties.")

lean_path.unlink(missing_ok=True)
