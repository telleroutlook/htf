"""Tests for htf/symmetric.py — U(1) symmetric / block-sparse tensors."""
import numpy as np
import pytest

from htf.symmetric import (
    BlockSparseTensor,
    ChargedBasis,
    block_sparse_matmul,
    check_u1_invariance,
    number_basis,
    project_to_u1,
    spin_half_basis,
    u1_blocks,
)

# ─────────── standard bases ───────────────────────────────────────────────

class TestStandardBases:

    def test_spin_half_dim(self):
        b = spin_half_basis()
        assert b.dim == 2

    def test_spin_half_charges(self):
        b = spin_half_basis()
        q = b.charge_array
        assert list(q) == [1, -1]

    def test_number_basis_dim(self):
        b = number_basis()
        assert b.dim == 2

    def test_number_basis_charges(self):
        b = number_basis()
        q = b.charge_array
        assert list(q) == [0, 1]


class TestChargedBasis:

    def test_dim_sum_of_sectors(self):
        b = ChargedBasis([(2, 0), (3, 1)])
        assert b.dim == 5

    def test_charge_array_length(self):
        b = ChargedBasis([(2, 0), (3, 1)])
        assert len(b.charge_array) == 5

    def test_charge_array_values(self):
        b = ChargedBasis([(2, 0), (3, 1)])
        assert list(b.charge_array) == [0, 0, 1, 1, 1]

    def test_single_sector(self):
        b = ChargedBasis([(4, 2)])
        assert b.dim == 4
        assert all(q == 2 for q in b.charge_array)


# ─────────── check_u1_invariance ─────────────────────────────────────────

class TestCheckU1Invariance:

    def test_z_operator_is_invariant(self):
        # Z operator: diagonal in spin-½ basis, preserves charge
        Z = np.diag([1.0, -1.0])
        b = spin_half_basis()
        r = check_u1_invariance(Z, dom_bases=[b], cod_bases=[b])
        assert r["is_invariant"]
        assert r["n_violations"] == 0

    def test_x_operator_is_not_invariant(self):
        # X = [[0,1],[1,0]] changes spin, violates U(1)
        X = np.array([[0.0, 1.0], [1.0, 0.0]])
        b = spin_half_basis()
        r = check_u1_invariance(X, dom_bases=[b], cod_bases=[b])
        assert not r["is_invariant"]
        assert r["n_violations"] > 0

    def test_identity_is_invariant(self):
        I = np.eye(2)
        b = spin_half_basis()
        r = check_u1_invariance(I, dom_bases=[b], cod_bases=[b])
        assert r["is_invariant"]

    def test_zero_tensor_is_invariant(self):
        T = np.zeros((2, 2))
        b = spin_half_basis()
        r = check_u1_invariance(T, dom_bases=[b], cod_bases=[b])
        assert r["is_invariant"]
        assert r["n_violations"] == 0

    def test_raising_operator_is_not_invariant(self):
        # S+ = [[0,1],[0,0]] raises spin, not invariant
        Sp = np.array([[0.0, 1.0], [0.0, 0.0]])
        b = spin_half_basis()
        r = check_u1_invariance(Sp, dom_bases=[b], cod_bases=[b])
        assert not r["is_invariant"]

    def test_charge_sectors_reported(self):
        Z = np.diag([1.0, -1.0])
        b = spin_half_basis()
        r = check_u1_invariance(Z, dom_bases=[b], cod_bases=[b])
        assert r["charge_sectors"] == [-1, 1]

    def test_number_operator_invariant(self):
        # n = diag(0, 1) in number basis — preserves particle number
        n_op = np.diag([0.0, 1.0])
        b = number_basis()
        r = check_u1_invariance(n_op, dom_bases=[b], cod_bases=[b])
        assert r["is_invariant"]


# ─────────── project_to_u1 ───────────────────────────────────────────────

class TestProjectToU1:

    def test_already_invariant_unchanged(self):
        Z = np.diag([1.0, -1.0])
        b = spin_half_basis()
        projected = project_to_u1(Z, [b], [b])
        assert np.allclose(projected, Z, atol=1e-12)

    def test_off_diagonal_zeroed(self):
        X = np.array([[0.0, 1.0], [1.0, 0.0]])
        b = spin_half_basis()
        projected = project_to_u1(X, [b], [b])
        assert np.allclose(projected, np.zeros((2, 2)), atol=1e-12)

    def test_mixed_tensor_partial_zeroing(self):
        # Diagonal preserved, off-diagonal zeroed
        A = np.array([[1.0, 0.5], [0.5, -1.0]])
        b = spin_half_basis()
        p = project_to_u1(A, [b], [b])
        # Diagonal stays
        assert abs(p[0, 0] - 1.0) < 1e-12
        assert abs(p[1, 1] + 1.0) < 1e-12
        # Off-diagonal zeroed
        assert abs(p[0, 1]) < 1e-12
        assert abs(p[1, 0]) < 1e-12

    def test_result_is_u1_invariant(self):
        A = np.random.default_rng(42).standard_normal((2, 2))
        b = spin_half_basis()
        p = project_to_u1(A, [b], [b])
        r = check_u1_invariance(p, [b], [b])
        assert r["is_invariant"]

    def test_shape_preserved(self):
        A = np.ones((2, 2))
        b = spin_half_basis()
        p = project_to_u1(A, [b], [b])
        assert p.shape == A.shape


# ─────────── u1_blocks ───────────────────────────────────────────────────

class TestU1Blocks:

    def test_z_operator_has_two_blocks(self):
        Z = np.diag([1.0, -1.0]).astype(complex)
        b = spin_half_basis()
        bst = u1_blocks(Z, [b], [b])
        assert len(bst.blocks) == 2

    def test_blocks_keys_are_charges(self):
        Z = np.diag([1.0, -1.0]).astype(complex)
        b = spin_half_basis()
        bst = u1_blocks(Z, [b], [b])
        assert set(bst.blocks.keys()) == {1, -1}

    def test_identity_blocks(self):
        I = np.eye(2).astype(complex)
        b = spin_half_basis()
        bst = u1_blocks(I, [b], [b])
        for blk in bst.blocks.values():
            assert np.allclose(blk, np.eye(blk.shape[0]), atol=1e-12)

    def test_to_dense_recovers_original(self):
        Z = np.diag([1.0, -1.0]).astype(complex)
        b = spin_half_basis()
        bst = u1_blocks(Z, [b], [b])
        recovered = bst.to_dense()
        assert np.allclose(recovered.reshape(2, 2), Z, atol=1e-12)

    def test_nnz_less_than_full(self):
        Z = np.diag([1.0, -1.0]).astype(complex)
        b = spin_half_basis()
        bst = u1_blocks(Z, [b], [b])
        assert bst.nnz() <= 4   # at most 4 total

    def test_sparsity_between_0_and_1(self):
        Z = np.diag([1.0, -1.0]).astype(complex)
        b = spin_half_basis()
        bst = u1_blocks(Z, [b], [b])
        assert 0.0 <= bst.sparsity() <= 1.0

    def test_number_operator_blocks(self):
        n_op = np.diag([0.0, 1.0]).astype(complex)
        b = number_basis()
        bst = u1_blocks(n_op, [b], [b])
        # Only sector Q=1 has a non-zero element (value 1)
        assert 1 in bst.blocks
        # The Q=0 block has value 0 (n_op[0,0]=0) → filtered out
        assert bst.blocks[1][0, 0] == pytest.approx(1.0)


# ─────────── block_sparse_matmul ──────────────────────────────────────────

class TestBlockSparseMatmul:

    def test_identity_times_z(self):
        I = np.eye(2).astype(complex)
        Z = np.diag([1.0, -1.0]).astype(complex)
        b = spin_half_basis()
        bst_I = u1_blocks(I, [b], [b])
        bst_Z = u1_blocks(Z, [b], [b])
        result = block_sparse_matmul(bst_I, bst_Z)
        dense = result.to_dense().reshape(2, 2)
        assert np.allclose(dense, Z, atol=1e-12)

    def test_z_times_z_is_identity(self):
        Z = np.diag([1.0, -1.0]).astype(complex)
        b = spin_half_basis()
        bst_Z = u1_blocks(Z, [b], [b])
        result = block_sparse_matmul(bst_Z, bst_Z)
        dense = result.to_dense().reshape(2, 2)
        assert np.allclose(dense, np.eye(2), atol=1e-12)

    def test_result_is_block_sparse_tensor(self):
        Z = np.diag([1.0, -1.0]).astype(complex)
        b = spin_half_basis()
        bst = u1_blocks(Z, [b], [b])
        result = block_sparse_matmul(bst, bst)
        assert isinstance(result, BlockSparseTensor)
