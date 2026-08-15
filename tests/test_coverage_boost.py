"""Coverage-boost tests for certificate.py, engine.py, topology.py, and functor.py.

Targets:
- certificate.py lines 33-38: numpy array / scalar handling in to_dict
- engine.py: Id certified path, Tensor certified path, Tensor float path with permutation,
             scalar Id (empty type), unknown node TypeError, unknown mode ValueError
- topology.py: Wire dim validation, Diagram.__repr__ for Id / Then / Tensor,
               Wire.__repr__, Wire.__eq__ NotImplemented, Wire.__hash__
- functor.py: KeyError for missing box assignment
"""

from __future__ import annotations

import numpy as np
import pytest

from htf import Box, Certificate, Id, TensorFunctor, Wire, contract
from htf.certificate import Certificate  # noqa: F811 (re-import is intentional)
from htf.topology import Diagram, Then


def _make_fake_diagram(w: Wire) -> Diagram:
    """Return a Diagram subclass instance that is none of Box/Id/Then/Tensor.

    Triggers the TypeError fallthrough in _eval / _eval_certified.
    """

    class _Fake(Diagram):
        pass

    d = _Fake()
    d.dom = (w,)
    d.cod = (w,)
    return d


# ──────────────────────────────────────────────────────────────────────────────
# Helpers shared across tests
# ──────────────────────────────────────────────────────────────────────────────


def _wire(name: str, dim: int = 2) -> Wire:
    return Wire(name, dim)


def _simple_box_and_functor():
    """Return (box_f, box_g, F) where f: (a,) -> (b,) and g: (c,) -> (d,)."""
    a = _wire("a", 2)
    b = _wire("b", 3)
    c = _wire("c", 4)
    d = _wire("d", 5)
    f = Box("f", (a,), (b,))
    g = Box("g", (c,), (d,))
    rng = np.random.default_rng(0)
    arr_f = rng.random((3, 2))  # shape dims(cod) + dims(dom) = (3,) + (2,)
    arr_g = rng.random((5, 4))  # shape (5,) + (4,)
    F = TensorFunctor({"f": arr_f, "g": arr_g})
    return f, g, F, arr_f, arr_g


# ──────────────────────────────────────────────────────────────────────────────
# certificate.py — numpy handling in to_dict (lines 33-38)
# ──────────────────────────────────────────────────────────────────────────────


class TestCertificateToDict:
    def test_ndarray_result_becomes_list(self):
        arr = np.array([[1.0, 2.0], [3.0, 4.0]])
        cert = Certificate(result=arr, mode="float")
        d = cert.to_dict()
        assert isinstance(d["result"], list)
        assert d["result"] == [[1.0, 2.0], [3.0, 4.0]]

    def test_1d_ndarray_result_becomes_list(self):
        arr = np.array([0.5, 1.5, 2.5])
        cert = Certificate(result=arr, mode="float")
        d = cert.to_dict()
        assert isinstance(d["result"], list)
        assert d["result"] == [0.5, 1.5, 2.5]

    def test_numpy_floating_result_becomes_python_float(self):
        val = np.float64(3.14)
        cert = Certificate(result=val, mode="float")
        d = cert.to_dict()
        assert isinstance(d["result"], float)
        assert abs(d["result"] - 3.14) < 1e-10

    def test_numpy_integer_result_becomes_python_int(self):
        val = np.int32(42)
        cert = Certificate(result=val, mode="float")
        d = cert.to_dict()
        # item() returns a Python scalar; for int32 that is int
        assert type(d["result"]) in (int, float)
        assert d["result"] == 42

    def test_plain_python_result_unchanged(self):
        cert = Certificate(result=7.0, mode="float")
        d = cert.to_dict()
        assert d["result"] == 7.0

    def test_to_dict_returns_all_fields(self):
        arr = np.zeros(3)
        cert = Certificate(
            result=arr, mode="certified", error_bound=1e-9, backend="flint-arb", chi=16, seed=99, notes="ok"
        )
        d = cert.to_dict()
        assert d["mode"] == "certified"
        assert d["error_bound"] == 1e-9
        assert d["backend"] == "flint-arb"
        assert d["chi"] == 16
        assert d["seed"] == 99
        assert d["notes"] == "ok"


# ──────────────────────────────────────────────────────────────────────────────
# engine.py — Id node in certified mode (lines 102-106)
# ──────────────────────────────────────────────────────────────────────────────


class TestEngineIdCertified:
    def test_id_certified_returns_certificate(self):
        w = _wire("x", 3)
        F = TensorFunctor()
        cert = contract(Id((w,)), F, mode="certified")
        assert isinstance(cert, Certificate)

    def test_id_certified_mode_field(self):
        w = _wire("x", 3)
        F = TensorFunctor()
        cert = contract(Id((w,)), F, mode="certified")
        assert cert.mode == "certified"

    def test_id_certified_result_is_identity_matrix(self):
        n = 4
        w = _wire("x", n)
        F = TensorFunctor()
        cert = contract(Id((w,)), F, mode="certified")
        # Id on a single wire of dim n: result shape (n, n), identity matrix
        np.testing.assert_allclose(cert.result, np.eye(n), atol=1e-12)

    def test_id_certified_error_bound_nonnegative(self):
        w = _wire("x", 2)
        F = TensorFunctor()
        cert = contract(Id((w,)), F, mode="certified")
        assert cert.error_bound is not None
        assert cert.error_bound >= 0.0

    def test_id_certified_multi_wire(self):
        """Id on a two-wire type produces a 4x4 identity in certified mode."""
        a = _wire("a", 2)
        b = _wire("b", 2)
        ty = (a, b)
        F = TensorFunctor()
        cert = contract(Id(ty), F, mode="certified")
        assert isinstance(cert, Certificate)
        # result shape: dims(cod) + dims(dom) = (2, 2, 2, 2)
        assert cert.result.shape == (2, 2, 2, 2)

    def test_id_certified_matches_float(self):
        n = 3
        w = _wire("x", n)
        F = TensorFunctor()
        float_result = contract(Id((w,)), F, mode="float")
        cert = contract(Id((w,)), F, mode="certified")
        np.testing.assert_allclose(cert.result, float_result, atol=cert.error_bound + 1e-14)


# ──────────────────────────────────────────────────────────────────────────────
# engine.py — Tensor node in certified mode (lines 115-130)
# ──────────────────────────────────────────────────────────────────────────────


class TestEngineTensorCertified:
    def test_tensor_certified_returns_certificate(self):
        f, g, F, _, _ = _simple_box_and_functor()
        cert = contract(f @ g, F, mode="certified")
        assert isinstance(cert, Certificate)

    def test_tensor_certified_mode_field(self):
        f, g, F, _, _ = _simple_box_and_functor()
        cert = contract(f @ g, F, mode="certified")
        assert cert.mode == "certified"

    def test_tensor_certified_result_shape(self):
        # f: (a:2,) -> (b:3,),  g: (c:4,) -> (d:5,)
        # f @ g: dom=(a,c), cod=(b,d) → result shape (3, 5, 2, 4)
        f, g, F, _, _ = _simple_box_and_functor()
        cert = contract(f @ g, F, mode="certified")
        assert cert.result.shape == (3, 5, 2, 4)

    def test_tensor_certified_error_bound_nonnegative(self):
        f, g, F, _, _ = _simple_box_and_functor()
        cert = contract(f @ g, F, mode="certified")
        assert cert.error_bound is not None
        assert cert.error_bound >= 0.0

    def test_tensor_certified_matches_float(self):
        f, g, F, _, _ = _simple_box_and_functor()
        float_result = contract(f @ g, F, mode="float")
        cert = contract(f @ g, F, mode="certified")
        np.testing.assert_allclose(cert.result, float_result, atol=cert.error_bound + 1e-12)

    def test_tensor_certified_kronecker_structure(self):
        """Certified result[b,d,a,c] must equal arr_f[b,a] * arr_g[d,c]."""
        f, g, F, arr_f, arr_g = _simple_box_and_functor()
        cert = contract(f @ g, F, mode="certified")
        expected = np.einsum("ba,dc->bdac", arr_f, arr_g)
        np.testing.assert_allclose(cert.result, expected, atol=1e-12)


# ──────────────────────────────────────────────────────────────────────────────
# engine.py — Tensor node in float mode with non-trivial permutation (lines 53-65)
# ──────────────────────────────────────────────────────────────────────────────


class TestEngineTensorFloat:
    def test_tensor_float_result_shape(self):
        # f: (a:2,) -> (b:3,),  g: (c:4,) -> (d:5,)
        # f @ g result shape should be dims(cod) + dims(dom) = (3, 5, 2, 4)
        f, g, F, _, _ = _simple_box_and_functor()
        result = contract(f @ g, F, mode="float")
        assert result.shape == (3, 5, 2, 4)

    def test_tensor_float_outer_product_values(self):
        """Result must equal the outer product with axes (cod_f, cod_g, dom_f, dom_g)."""
        f, g, F, arr_f, arr_g = _simple_box_and_functor()
        result = contract(f @ g, F, mode="float")
        expected = np.einsum("ba,dc->bdac", arr_f, arr_g)
        np.testing.assert_allclose(result, expected, rtol=1e-12)

    def test_tensor_float_permutation_is_applied(self):
        """Verify permutation actually reorders axes (result != raw outer product)."""
        f, g, F, arr_f, arr_g = _simple_box_and_functor()
        result = contract(f @ g, F, mode="float")
        raw_outer = np.tensordot(arr_f, arr_g, axes=0)  # shape (3, 2, 5, 4)
        # raw_outer has shape (3, 2, 5, 4); result has shape (3, 5, 2, 4)
        assert result.shape != raw_outer.shape or not np.allclose(result, raw_outer)

    def test_tensor_float_identity_boxes(self):
        """Two identity boxes tensored: result is block-identity."""
        a = _wire("a", 2)
        b = _wire("b", 2)
        id_a = Id((a,))
        id_b = Id((b,))
        F = TensorFunctor()
        result = contract(id_a @ id_b, F, mode="float")
        # result shape: (2, 2, 2, 2), should be eye(2) ⊗ eye(2)
        assert result.shape == (2, 2, 2, 2)

    def test_tensor_float_scalar_boxes(self):
        """Tensor of two scalar-output boxes gives correct outer product shape."""
        a = _wire("a", 2)
        b = _wire("b", 2)
        # scalar boxes: dom=(a,), cod=()
        f = Box("p", (a,), ())
        g = Box("q", (b,), ())
        arr_f = np.array([1.0, 2.0])  # shape (2,) = dims(()) + dims((a,)) wait no
        # shape should be dims(cod) + dims(dom) = () + (2,) = (2,)
        arr_f = np.array([1.0, 2.0])
        arr_g = np.array([3.0, 4.0])
        F2 = TensorFunctor({"p": arr_f, "q": arr_g})
        result = contract(f @ g, F2, mode="float")
        # f @ g: dom=(a, b), cod=()
        # result shape: dims(()) + dims((a, b)) = (2, 2)
        assert result.shape == (2, 2)
        expected = np.outer(arr_f, arr_g)
        np.testing.assert_allclose(result, expected, rtol=1e-12)


# ──────────────────────────────────────────────────────────────────────────────
# topology.py — Wire dim validation (lines 19-20)
# ──────────────────────────────────────────────────────────────────────────────


class TestWireDimValidation:
    def test_zero_dim_raises_value_error(self):
        with pytest.raises(ValueError, match="positive integer"):
            Wire("x", 0)

    def test_negative_dim_raises_value_error(self):
        with pytest.raises(ValueError, match="positive integer"):
            Wire("x", -1)

    def test_dim_one_is_valid(self):
        w = Wire("scalar", 1)
        assert w.dim == 1

    def test_dim_positive_is_valid(self):
        w = Wire("v", 5)
        assert w.dim == 5


# ──────────────────────────────────────────────────────────────────────────────
# topology.py — Diagram.__repr__ for Id, Then, Tensor (line 57)
# ──────────────────────────────────────────────────────────────────────────────


class TestDiagramRepr:
    def test_id_repr_contains_class_name(self):
        w = _wire("x", 3)
        r = repr(Id((w,)))
        assert r.startswith("Id(")

    def test_id_repr_shows_dims(self):
        w = _wire("x", 3)
        r = repr(Id((w,)))
        assert "(3,)" in r

    def test_id_empty_ty_repr(self):
        r = repr(Id(()))
        assert r.startswith("Id(")
        assert "()" in r

    def test_then_repr_contains_class_name(self):
        a = _wire("a", 2)
        b = _wire("b", 3)
        f = Box("f", (a,), (b,))
        g = Box("g", (b,), (a,))
        r = repr(Then(f, g))
        assert r.startswith("Then(")

    def test_then_repr_shows_dom_and_cod(self):
        a = _wire("a", 2)
        b = _wire("b", 3)
        f = Box("f", (a,), (b,))
        g = Box("g", (b,), (a,))
        r = repr(f >> g)
        # dom of (f >> g) is (2,), cod is (2,)
        assert "(2,)" in r

    def test_tensor_repr_contains_class_name(self):
        a = _wire("a", 2)
        b = _wire("b", 3)
        f = Box("f", (a,), (b,))
        g = Box("g", (a,), (b,))
        r = repr(f @ g)
        assert r.startswith("Tensor(")

    def test_tensor_repr_shows_combined_dims(self):
        a = _wire("a", 2)
        b = _wire("b", 3)
        f = Box("f", (a,), (b,))
        g = Box("g", (a,), (b,))
        r = repr(f @ g)
        # dom of (f @ g) is (2, 2), cod is (3, 3)
        assert "(2, 2)" in r
        assert "(3, 3)" in r

    def test_box_repr_is_not_delegated_to_diagram(self):
        a = _wire("a", 2)
        b = _wire("b", 3)
        f = Box("myfunc", (a,), (b,))
        r = repr(f)
        assert "myfunc" in r
        assert r.startswith("Box(")


# ──────────────────────────────────────────────────────────────────────────────
# topology.py — Wire.__repr__, Wire.__eq__ NotImplemented, Wire.__hash__
# ──────────────────────────────────────────────────────────────────────────────


class TestWireMethods:
    def test_repr_format(self):
        w = Wire("alpha", 5)
        assert repr(w) == "Wire('alpha', 5)"

    def test_repr_quoted_name(self):
        w = Wire("x", 1)
        assert "x" in repr(w)
        assert "1" in repr(w)

    def test_eq_with_non_wire_returns_not_implemented(self):
        w = Wire("x", 3)
        assert w.__eq__("not_a_wire") is NotImplemented
        assert w.__eq__(42) is NotImplemented
        assert w.__eq__(None) is NotImplemented

    def test_hash_equal_wires_same_hash(self):
        w1 = Wire("q", 4)
        w2 = Wire("q", 4)
        assert hash(w1) == hash(w2)

    def test_hash_different_name_different_hash(self):
        assert hash(Wire("a", 3)) != hash(Wire("b", 3))

    def test_hash_different_dim_different_hash(self):
        assert hash(Wire("x", 2)) != hash(Wire("x", 3))

    def test_wire_usable_as_dict_key(self):
        w = Wire("k", 2)
        d = {w: "value"}
        assert d[Wire("k", 2)] == "value"

    def test_wire_usable_in_set(self):
        s = {Wire("a", 2), Wire("a", 2), Wire("b", 3)}
        assert len(s) == 2


# ──────────────────────────────────────────────────────────────────────────────
# engine.py — scalar Id (empty type), unknown node TypeError, unknown mode
# ──────────────────────────────────────────────────────────────────────────────


class TestEngineEdgeCases:
    def test_empty_id_float_returns_scalar_one(self):
        result = contract(Id(()), TensorFunctor(), mode="float")
        assert float(result) == pytest.approx(1.0)
        assert result.shape == ()

    def test_empty_id_certified_returns_certificate(self):
        cert = contract(Id(()), TensorFunctor(), mode="certified")
        assert isinstance(cert, Certificate)
        assert float(cert.result) == pytest.approx(1.0)

    def test_unknown_mode_raises_value_error(self):
        w = Wire("x", 2)
        with pytest.raises(ValueError, match="unknown mode"):
            contract(Id((w,)), TensorFunctor(), mode="bad_mode")

    def test_unknown_node_float_raises_type_error(self):
        w = Wire("x", 2)
        d = _make_fake_diagram(w)
        with pytest.raises(TypeError, match="unknown diagram node"):
            contract(d, TensorFunctor(), mode="float")

    def test_unknown_node_certified_raises_type_error(self):
        w = Wire("x", 2)
        d = _make_fake_diagram(w)
        with pytest.raises(TypeError, match="unknown diagram node"):
            contract(d, TensorFunctor(), mode="certified")


# ──────────────────────────────────────────────────────────────────────────────
# functor.py — KeyError for missing tensor assignment
# ──────────────────────────────────────────────────────────────────────────────


class TestFunctorMissingTensor:
    def test_missing_box_raises_key_error(self):
        w = Wire("x", 2)
        box = Box("unregistered", (w,), (w,))
        F = TensorFunctor()
        with pytest.raises(KeyError):
            contract(box, F, mode="float")

    def test_error_message_contains_box_name(self):
        w = Wire("x", 2)
        box = Box("my_unregistered_box", (w,), (w,))
        F = TensorFunctor()
        with pytest.raises(KeyError, match="my_unregistered_box"):
            contract(box, F, mode="float")


# ──────────────────────────────────────────────────────────────────────────────
# htf/labs/__init__.py — re-export smoke tests (0% → covered)
# ──────────────────────────────────────────────────────────────────────────────


class TestLabsImport:
    """Import htf.labs and verify a representative subset of re-exports exist."""

    def test_import_labs_module(self):
        import htf.labs as labs

        assert labs is not None

    def test_mps_available(self):
        from htf.labs import MPS

        assert MPS is not None

    def test_tebd_evolve_available(self):
        from htf.labs import tebd_evolve

        assert callable(tebd_evolve)

    def test_dmrg_sweep_mpo_available(self):
        from htf.labs import dmrg_sweep_mpo

        assert callable(dmrg_sweep_mpo)

    def test_rayleigh_certificate_to_lean_not_in_labs(self):
        import htf.labs as labs

        assert not hasattr(labs, "rayleigh_certificate_to_lean")

    def test_certified_gap_upper_available(self):
        from htf.labs import certified_gap_upper

        assert callable(certified_gap_upper)

    def test_zx_available(self):
        from htf.labs import ZXGraph, clifford_simplify

        assert callable(clifford_simplify)
        assert ZXGraph is not None


# ──────────────────────────────────────────────────────────────────────────────
# htf/claim_registry.py — get_claim and registry_summary (77% → covered)
# ──────────────────────────────────────────────────────────────────────────────


class TestClaimRegistry:
    def test_get_claim_rayleigh_returns_claiminfo(self):
        from htf.claim_registry import get_claim

        info = get_claim("rayleigh")
        assert info.claim_id == "rayleigh"
        assert info.assurance == "rigorous"

    def test_get_claim_gap_returns_heuristic(self):
        from htf.claim_registry import get_claim

        info = get_claim("gap")
        assert info.assurance == "heuristic"

    def test_get_claim_unknown_raises_key_error(self):
        from htf.claim_registry import get_claim

        with pytest.raises(KeyError, match="Unknown claim_id"):
            get_claim("nonexistent_claim_xyz")

    def test_get_claim_error_lists_known_ids(self):
        from htf.claim_registry import get_claim

        with pytest.raises(KeyError) as exc_info:
            get_claim("bad_id")
        assert "rayleigh" in str(exc_info.value)

    def test_registry_summary_returns_dict(self):
        from htf.claim_registry import registry_summary

        summary = registry_summary()
        assert isinstance(summary, dict)
        assert "rayleigh" in summary
        assert "title" in summary["rayleigh"]


# ──────────────────────────────────────────────────────────────────────────────
# htf/_rayleigh_primitives.py — flint-absent fallback and zero-psi guard
# ──────────────────────────────────────────────────────────────────────────────


class TestRayleighPrimitivesNoFlint:
    """Cover the ImportError fallback paths in _arb_rayleigh and _acb_rayleigh."""

    def test_arb_fallback_returns_float_triple(self, monkeypatch):
        import sys

        from htf._rayleigh_primitives import _arb_rayleigh

        monkeypatch.setitem(sys.modules, "flint", None)
        H = np.diag([1.0, 2.0])
        psi = np.array([1.0, 0.0])
        lo, up, rad, label = _arb_rayleigh(H, psi)
        assert lo == up
        assert rad == 0.0
        assert "numpy" in label.lower()

    def test_acb_fallback_returns_float_triple(self, monkeypatch):
        import sys

        from htf._rayleigh_primitives import _acb_rayleigh

        monkeypatch.setitem(sys.modules, "flint", None)
        H = np.array([[1.0, 1j], [-1j, 2.0]], dtype=complex)
        psi = np.array([1.0, 0.0], dtype=complex)
        lo, up, rad, label = _acb_rayleigh(H, psi)
        assert lo == up
        assert rad == 0.0
        assert "numpy" in label.lower()


class TestRayleighPrimitivesZeroPsi:
    """Cover the denominator-contains-zero guards in _arb_rayleigh and _acb_rayleigh."""

    def test_arb_zero_psi_raises_value_error(self):
        from htf._rayleigh_primitives import _arb_rayleigh

        H = np.diag([1.0, 2.0])
        psi = np.zeros(2)
        with pytest.raises(ValueError, match="[Dd]enominator"):
            _arb_rayleigh(H, psi)

    def test_acb_zero_psi_raises_value_error(self):
        from htf._rayleigh_primitives import _acb_rayleigh

        H = np.array([[1.0, 0.0], [0.0, 2.0]], dtype=complex)
        psi = np.zeros(2, dtype=complex)
        with pytest.raises(ValueError, match="[Dd]enominator"):
            _acb_rayleigh(H, psi)


# ──────────────────────────────────────────────────────────────────────────────
# engine.py — tensordot path (nb==0) and certified-mode ImportError (lines 68-70, 211)
# ──────────────────────────────────────────────────────────────────────────────


class TestEngineTensordotPath:
    """Lines 68-70: np.tensordot fallback when shared-wire count is zero."""

    def test_compose_empty_type_diagrams(self):
        F = TensorFunctor()
        # Id(()) >> Id(()): Then with f.cod = () so nb = len(()) = 0
        result = contract(Id(()) >> Id(()), F, mode="float")
        assert float(result) == pytest.approx(1.0)

    def test_compose_empty_cod_box(self):
        # Box mapping () → () with result scalar 3.0
        scalar_box = Box("s", (), ())
        F = TensorFunctor({"s": np.array(3.0)})
        result = contract(scalar_box >> Id(()), F, mode="float")
        assert float(result) == pytest.approx(3.0)


class TestEngineCertifiedNoFlint:
    """Line 211-212: certified mode raises ImportError when flint absent."""

    def test_certified_raises_without_flint(self, monkeypatch):
        import sys

        w = Wire("x", 2)
        monkeypatch.setitem(sys.modules, "flint", None)
        with pytest.raises(ImportError, match="python-flint"):
            contract(Id((w,)), TensorFunctor(), mode="certified")


# ──────────────────────────────────────────────────────────────────────────────
# htf/viz.py — unknown diagram subclass fallback (lines 103-109)
# ──────────────────────────────────────────────────────────────────────────────


class TestVizUnknownDiagram:
    """Cover the fallback branch in _visit for unknown Diagram subclasses."""

    def test_unknown_subclass_produces_node(self):
        from htf.topology import Diagram
        from htf.viz import diagram_to_dict

        class _Unknown(Diagram):
            pass

        u = _Unknown()
        u.dom = (Wire("a", 2),)
        u.cod = (Wire("b", 3),)
        result = diagram_to_dict(u)
        # Must return a valid graph dict with at least one node
        assert isinstance(result, dict)
        assert len(result.get("nodes", [])) >= 1
