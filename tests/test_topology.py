"""Tests for HTF Layer 1 (topology) and Layer 3 (engine) — the Phase-1 skeleton."""
import numpy as np
import pytest

from htf import Box, TensorFunctor, Wire, contract


def test_type_mismatch_raises():
    a, b = Wire("a", 2), Wire("b", 3)
    f = Box("f", (a,), (b,))   # cod dim 3
    g = Box("g", (a,), (b,))   # dom dim 2 -> mismatch with f.cod
    with pytest.raises(TypeError):
        _ = f >> g


def test_hello_world_scalar():
    s = Wire("s", 2)
    psi = Box("psi", (), (s,))
    U = Box("U", (s,), (s,))
    phi = Box("phi", (s,), ())
    d = psi >> U >> phi
    assert d.dom == () and d.cod == ()
    F = TensorFunctor(
        {
            "psi": np.array([1.0, 0.0]),
            "U": np.array([[0.0, 1.0], [1.0, 0.0]]),  # swap
            "phi": np.array([0.0, 1.0]),
        }
    )
    val = contract(d, F)
    assert np.isclose(float(val), 1.0)  # <phi| U |psi> = 1


def test_matrix_compose():
    s = Wire("s", 2)
    psi = Box("psi", (), (s,))
    U = Box("U", (s,), (s,))
    d = psi >> U
    F = TensorFunctor({"psi": np.array([1.0, 0.0]), "U": np.array([[0.0, 1.0], [1.0, 0.0]])})
    out = contract(d, F)
    assert np.allclose(out, np.array([0.0, 1.0]))  # swap sends |0> -> |1>


def test_tensor_product_types():
    a, b = Wire("a", 2), Wire("b", 3)
    f = Box("f", (), (a,))
    g = Box("g", (), (b,))
    d = f @ g
    assert d.dom == () and d.cod == (a, b)


def test_shape_validation():
    s = Wire("s", 2)
    bad = Box("bad", (s,), (s,))
    F = TensorFunctor({"bad": np.array([1.0, 0.0])})  # wrong shape (should be 2x2)
    with pytest.raises(ValueError):
        contract(bad, F)


def test_certified_mode_works():
    """certified mode now returns a Certificate (Phase 2 deliverable)."""
    from htf import Certificate
    s = Wire("s", 2)
    psi = Box("psi", (), (s,))
    F = TensorFunctor({"psi": np.array([1.0, 0.0])})
    cert = contract(psi, F, mode="certified")
    assert isinstance(cert, Certificate)
    assert cert.mode == "certified"
    assert cert.error_bound is not None


def test_p0_3_regression_complex_input_raises_type_error():
    # Regression P0-3: complex tensors must be rejected with TypeError, not
    # silently truncated to float (which produced <1|S|1>=i → 0 ± 0 certified).
    s = Wire("spin", 2)
    S_gate = Box("S", (s,), (s,))          # phase gate diag(1, i)
    S_matrix_complex = np.array([[1.0, 0.0], [0.0, 1j]])  # complex dtype
    with pytest.raises(TypeError, match="complex"):
        TensorFunctor({"S": S_matrix_complex}).tensor(S_gate)


def test_p0_3_regression_real_float_still_accepted():
    # Counterpart: real tensors must still work normally.
    s = Wire("spin", 2)
    X_gate = Box("X", (s,), (s,))
    F = TensorFunctor({"X": np.array([[0.0, 1.0], [1.0, 0.0]])})
    t = F.tensor(X_gate)
    assert t.dtype == float


def test_wire_identity_same_dim_different_name_rejected():
    # P0-B regression: same dimension but different name must NOT compose.
    spin = Wire("spin", 2)
    charge = Wire("charge", 2)
    f = Box("f", (), (spin,))   # cod = (spin,)
    g = Box("g", (charge,), ()) # dom = (charge,)
    with pytest.raises(TypeError, match="type mismatch"):
        _ = f >> g


def test_wire_identity_same_name_same_dim_accepted():
    # Sanity: same name and same dim must still compose.
    s = Wire("spin", 2)
    f = Box("f", (), (s,))
    g = Box("g", (s,), ())
    d = f >> g
    assert d.dom == () and d.cod == ()


def test_engine_certified_outward_rounded():
    # P0-A regression: certified mode error_bound must cover the true rounding
    # error; a zero error_bound with a non-exact midpoint is a soundness failure.
    try:
        import flint  # noqa: F401
    except ImportError:
        pytest.skip("python-flint not installed")
    s = Wire("s", 1)
    psi = Box("psi", (), (s,))
    # Use 0.1 as the tensor value: its float64 representation is not exact.
    F = TensorFunctor({"psi": np.array([0.1])})
    cert = contract(psi, F, mode="certified")
    # The result is a scalar ≈ 0.1; the error_bound must be >= 0 and the
    # true value must lie within result ± error_bound.
    import math
    assert cert.error_bound >= 0.0
    # 0.1 exactly as a fraction is 1/10; check containment using Fraction.
    from fractions import Fraction
    result_val = float(np.asarray(cert.result).flat[0])
    err_val = float(cert.error_bound)
    exact = Fraction(1, 10)
    lo = Fraction(result_val) - Fraction(err_val)
    hi = Fraction(result_val) + Fraction(err_val)
    assert lo <= exact <= hi, (
        f"True value {float(exact)} not in [{float(lo)}, {float(hi)}]; "
        f"result={result_val}, error_bound={err_val}"
    )
