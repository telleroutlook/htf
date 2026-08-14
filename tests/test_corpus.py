"""Tests for htf/corpus.py — public benchmark corpus."""
import json

import pytest

from htf.corpus import (
    CORPUS,
    CorpusCase,
    CorpusReport,
    corpus_by_tag,
    run_corpus,
)
from htf.rayleigh_cert import RayleighCertificate

# ─────────────────── corpus structure ────────────────────────────────────────

class TestCorpusStructure:

    def test_corpus_nonempty(self):
        assert len(CORPUS) >= 8

    def test_all_cases_are_corpus_case(self):
        for c in CORPUS:
            assert isinstance(c, CorpusCase)

    def test_names_unique(self):
        names = [c.name for c in CORPUS]
        assert len(names) == len(set(names))

    def test_all_have_tags(self):
        for c in CORPUS:
            assert len(c.tags) >= 1

    def test_all_have_h_and_psi(self):
        for c in CORPUS:
            assert c.H.ndim == 2
            assert c.psi.ndim == 1

    def test_h_square(self):
        for c in CORPUS:
            assert c.H.shape[0] == c.H.shape[1]

    def test_psi_length_matches_h(self):
        for c in CORPUS:
            assert len(c.psi) == c.H.shape[0]

    def test_expected_upper_geq_expected_e0(self):
        for c in CORPUS:
            assert c.expected_upper >= c.expected_E0 - 1e-15


# ─────────────────── tag filtering ───────────────────────────────────────────

class TestCorpusByTag:

    def test_exact_cases_exist(self):
        assert len(corpus_by_tag("exact")) >= 2

    def test_complex_cases_exist(self):
        assert len(corpus_by_tag("complex")) >= 2

    def test_near_degenerate_cases_exist(self):
        assert len(corpus_by_tag("near-degenerate")) >= 2

    def test_ill_conditioned_cases_exist(self):
        assert len(corpus_by_tag("ill-conditioned")) >= 2

    def test_cross_platform_cases_exist(self):
        assert len(corpus_by_tag("cross-platform")) >= 2

    def test_physics_cases_exist(self):
        assert len(corpus_by_tag("physics")) >= 2

    def test_multi_tag_filter(self):
        exact_physics = corpus_by_tag("exact", "physics")
        assert all("exact" in c.tags and "physics" in c.tags for c in exact_physics)

    def test_empty_result_for_unknown_tag(self):
        assert corpus_by_tag("nonexistent-tag-xyz") == []


# ─────────────────── individual cases: certificate ───────────────────────────

class TestCorpusCaseCertificate:

    def test_trivial_2x2_certificate(self):
        c = next(x for x in CORPUS if x.name == "trivial_2x2")
        cert = c.certificate()
        assert isinstance(cert, RayleighCertificate)
        assert cert.upper <= c.expected_upper + 1e-15

    def test_near_degenerate_2x2_certificate(self):
        c = next(x for x in CORPUS if x.name == "near_degenerate_2x2")
        cert = c.certificate()
        assert cert.upper <= c.expected_upper + 1e-15

    def test_complex_hermitian_2x2_certificate(self):
        c = next(x for x in CORPUS if x.name == "complex_hermitian_2x2")
        cert = c.certificate()
        assert cert.upper <= c.expected_upper + 1e-9

    def test_ill_conditioned_scale_certificate(self):
        c = next(x for x in CORPUS if x.name == "ill_conditioned_scale_1e12")
        cert = c.certificate()
        assert cert.upper <= c.expected_upper + 1e-15

    def test_tfim_n4_certificate(self):
        c = next(x for x in CORPUS if x.name == "tfim_n4_exact")
        cert = c.certificate()
        assert cert.upper >= c.expected_E0 - 1e-12
        assert cert.upper <= c.expected_upper + 1e-9

    def test_xx_n4_certificate(self):
        c = next(x for x in CORPUS if x.name == "xx_n4_exact")
        cert = c.certificate()
        assert cert.upper >= c.expected_E0 - 1e-12

    def test_cross_platform_digest_is_64hex(self):
        for c in corpus_by_tag("cross-platform"):
            cert = c.certificate()
            assert len(cert.input_digest) == 64
            int(cert.input_digest, 16)

    def test_certificate_notes_include_corpus_prefix(self):
        c = CORPUS[0]
        cert = c.certificate()
        assert "corpus:" in cert.notes


# ─────────────────── individual cases: run() ─────────────────────────────────

class TestCorpusCaseRun:

    def test_run_trivial_passes(self):
        c = next(x for x in CORPUS if x.name == "trivial_2x2")
        result = c.run()
        assert result.passed is True
        assert result.error is None

    def test_run_returns_verified_digest(self):
        c = next(x for x in CORPUS if x.name == "trivial_2x2")
        result = c.run()
        assert len(result.input_digest) == 64

    def test_run_elapsed_positive(self):
        c = CORPUS[0]
        result = c.run()
        assert result.elapsed_s >= 0.0

    def test_run_to_dict_keys(self):
        c = CORPUS[0]
        d = c.run().to_dict()
        for key in ("name", "tags", "passed", "expected_upper", "upper", "backend"):
            assert key in d

    def test_run_complex_case_passes(self):
        c = next(x for x in CORPUS if x.name == "complex_hermitian_2x2")
        result = c.run()
        assert result.passed is True

    def test_run_ill_conditioned_passes(self):
        c = next(x for x in CORPUS if x.name == "ill_conditioned_scale_1e12")
        result = c.run()
        assert result.passed is True


# ─────────────────── run_corpus ──────────────────────────────────────────────

class TestRunCorpus:

    @pytest.fixture(scope="class")
    def report(self):
        return run_corpus()

    def test_report_type(self, report):
        assert isinstance(report, CorpusReport)

    def test_all_pass(self, report):
        failed = [r for r in report.results if not r.passed]
        assert failed == [], f"Failed cases: {[r.case.name for r in failed]}"

    def test_n_total_matches_corpus(self, report):
        assert report.n_total == len(CORPUS)

    def test_n_passed_equals_n_total(self, report):
        assert report.n_passed == report.n_total

    def test_n_failed_zero(self, report):
        assert report.n_failed == 0

    def test_summary_is_string(self, report):
        s = report.summary()
        assert isinstance(s, str)
        assert "PASS" in s
        assert "FAIL" not in s

    def test_to_dict_structure(self, report):
        d = report.to_dict()
        assert d["n_total"] == len(CORPUS)
        assert d["n_passed"] == len(CORPUS)
        assert d["n_failed"] == 0
        assert len(d["results"]) == len(CORPUS)

    def test_subset_run(self):
        exact = corpus_by_tag("exact")
        report = run_corpus(exact)
        assert report.n_total == len(exact)
        assert report.n_passed == report.n_total


# ─────────────────── cross-platform digest stability ─────────────────────────

class TestCrossplatformDigests:

    def test_two_runs_same_digest(self):
        """Same inputs → same SHA-256 digest on every run."""
        for case in corpus_by_tag("cross-platform"):
            cert1 = case.certificate()
            cert2 = case.certificate()
            assert cert1.input_digest == cert2.input_digest

    def test_different_cases_different_digests(self):
        """Different (H, psi) → different digests."""
        digests = [c.certificate().input_digest for c in corpus_by_tag("cross-platform")]
        assert len(digests) == len(set(digests))

    def test_cross_platform_uniform_digest_known(self):
        """Uniform psi on random-seed-0 H: digest known at build time."""
        from htf.rayleigh_cert import _canonical_digest
        case = next(c for c in CORPUS if c.name == "cross_platform_random_n4_uniform")
        expected = _canonical_digest(case.H, case.psi)
        cert = case.certificate()
        assert cert.input_digest == expected


# ─────────────────── serialisation ───────────────────────────────────────────

class TestCorpusSerialization:

    def test_to_full_json_roundtrip(self):
        from htf.verify import verify_from_dict
        case = next(c for c in CORPUS if c.name == "trivial_2x2")
        cert = case.certificate()
        full = json.loads(cert.to_full_json())
        result = verify_from_dict(full)
        assert result["verified"] is True

    def test_complex_to_full_json_roundtrip(self):
        from htf.verify import verify_from_dict
        case = next(c for c in CORPUS if c.name == "complex_hermitian_2x2")
        cert = case.certificate()
        full = json.loads(cert.to_full_json())
        result = verify_from_dict(full)
        assert result["verified"] is True
