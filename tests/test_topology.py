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
