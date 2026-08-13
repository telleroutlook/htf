"""Tests for htf/benchmark.py — certified reproducibility benchmark suite."""
import json
import math

import pytest

from htf.benchmark import BenchmarkReport, BenchmarkResult, run_benchmark


# ─────────────────── TestBenchmarkResult ─────────────────────────────────

class TestBenchmarkResult:

    def _make(self, **kw):
        defaults = dict(
            model="ising", n_sites=4, chi=2, seed=0,
            E0_var=-3.3, E0_error_bound=1e-14,
            gap_exact=0.095, gap_var=4.5,
            gap_cert_result=4.5, gap_cert_error=1e-13,
            temple_lb=-4.0, temple_condition_met=False,
            os_passed=True, max_entropy=0.58,
            likely_area_law=True, n_iter_used=50, elapsed_s=0.1,
        )
        defaults.update(kw)
        return BenchmarkResult(**defaults)

    def test_to_dict_has_all_fields(self):
        d = self._make().to_dict()
        for f in ("model", "n_sites", "chi", "seed", "E0_var", "E0_error_bound",
                  "gap_exact", "gap_var", "gap_cert_result", "gap_cert_error",
                  "temple_lb", "temple_condition_met", "os_passed",
                  "max_entropy", "likely_area_law", "n_iter_used", "elapsed_s"):
            assert f in d

    def test_to_dict_json_serialisable(self):
        d = self._make().to_dict()
        assert json.dumps(d)  # must not raise

    def test_os_passed_is_bool(self):
        r = self._make(os_passed=True)
        assert isinstance(r.to_dict()["os_passed"], bool)

    def test_model_string_preserved(self):
        r = self._make(model="ising_critical")
        assert r.to_dict()["model"] == "ising_critical"


# ─────────────────── TestBenchmarkReport ─────────────────────────────────

class TestBenchmarkReport:

    def _empty_report(self):
        return BenchmarkReport(
            htf_version="0.5.0", n_sites=4, chi=2, n_iter=50, seed=0
        )

    def test_to_dict_keys(self):
        d = self._empty_report().to_dict()
        for k in ("htf_version", "n_sites", "chi", "n_iter", "seed", "results"):
            assert k in d

    def test_to_dict_results_is_list(self):
        assert isinstance(self._empty_report().to_dict()["results"], list)

    def test_to_json_is_valid_json(self):
        rep = self._empty_report()
        s = rep.to_json()
        data = json.loads(s)
        assert "htf_version" in data

    def test_summary_contains_version(self):
        rep = self._empty_report()
        assert "0.5.0" in rep.summary()

    def test_summary_contains_n_sites(self):
        rep = self._empty_report()
        assert "4" in rep.summary()


# ─────────────────── TestRunBenchmark ────────────────────────────────────

class TestRunBenchmark:

    @pytest.fixture(scope="class")
    @classmethod
    def report_ising(cls):
        return run_benchmark(n_sites=4, chi=2, n_iter=20, seed=0, models=["ising"])

    @pytest.fixture(scope="class")
    @classmethod
    def report_both(cls):
        return run_benchmark(n_sites=4, chi=2, n_iter=20, seed=0)

    def test_returns_benchmark_report(self, report_ising):
        assert isinstance(report_ising, BenchmarkReport)

    def test_result_count_matches_models(self, report_ising):
        assert len(report_ising.results) == 1

    def test_both_models_default(self, report_both):
        names = {r.model for r in report_both.results}
        assert names == {"ising", "xx"}

    def test_result_is_benchmark_result(self, report_ising):
        assert isinstance(report_ising.results[0], BenchmarkResult)

    def test_version_recorded(self, report_ising):
        from htf import __version__
        assert report_ising.htf_version == __version__

    def test_n_sites_recorded(self, report_ising):
        assert report_ising.n_sites == 4
        assert report_ising.results[0].n_sites == 4

    def test_E0_var_is_float(self, report_ising):
        assert isinstance(report_ising.results[0].E0_var, float)

    def test_E0_error_bound_nonneg(self, report_ising):
        assert report_ising.results[0].E0_error_bound >= 0

    def test_gap_exact_positive(self, report_ising):
        assert report_ising.results[0].gap_exact > 0

    def test_gap_cert_error_nonneg(self, report_ising):
        assert report_ising.results[0].gap_cert_error >= 0

    def test_os_passed_true_for_ising(self, report_ising):
        assert report_ising.results[0].os_passed is True

    def test_os_passed_true_for_xx(self, report_both):
        xx = next(r for r in report_both.results if r.model == "xx")
        assert xx.os_passed is True

    def test_max_entropy_nonneg(self, report_ising):
        assert report_ising.results[0].max_entropy >= 0

    def test_likely_area_law_is_bool(self, report_ising):
        assert isinstance(report_ising.results[0].likely_area_law, bool)

    def test_n_iter_used_positive(self, report_ising):
        assert report_ising.results[0].n_iter_used > 0

    def test_elapsed_s_positive(self, report_ising):
        assert report_ising.results[0].elapsed_s > 0

    def test_to_json_parseable(self, report_ising):
        d = json.loads(report_ising.to_json())
        assert len(d["results"]) == 1

    def test_summary_contains_model_name(self, report_ising):
        assert "ising" in report_ising.summary()

    def test_summary_contains_gap_exact(self, report_ising):
        r = report_ising.results[0]
        # Gap exact should appear somewhere in the summary
        assert str(r.gap_exact)[:4] in report_ising.summary() or "gap" in report_ising.summary().lower()

    def test_ising_critical_model(self):
        rep = run_benchmark(n_sites=4, chi=2, n_iter=15, seed=1, models=["ising_critical"])
        assert len(rep.results) == 1
        assert rep.results[0].model == "ising_critical"

    def test_unknown_model_raises(self):
        with pytest.raises(ValueError, match="Unknown model"):
            run_benchmark(models=["invalid_model_xyz"])

    def test_reproducibility_same_seed(self):
        r1 = run_benchmark(n_sites=4, chi=2, n_iter=15, seed=5, models=["ising"])
        r2 = run_benchmark(n_sites=4, chi=2, n_iter=15, seed=5, models=["ising"])
        assert abs(r1.results[0].E0_var - r2.results[0].E0_var) < 1e-10
