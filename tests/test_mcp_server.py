"""Tests for htf/mcp_server.py — MCP server construction and tool listing."""
import json

import numpy as np
import pytest

from htf.mcp_server import HAS_MCP, _build_server


@pytest.mark.skipif(not HAS_MCP, reason="mcp package not installed")
class TestMcpServer:

    @pytest.fixture(scope="class")
    @classmethod
    def server(cls):
        return _build_server()

    def test_server_name(self, server):
        assert server.name == "htf"

    def test_server_version_matches_htf(self, server):
        from htf import __version__
        assert server.version == __version__

    def test_server_has_description(self, server):
        assert server.description is not None
        assert len(server.description) > 0

    def test_server_description_contains_out(self, server):
        assert "[OUT]" in server.description

    def test_import_has_mcp_flag_true(self):
        assert HAS_MCP is True

    def test_build_server_returns_instance(self, server):
        from mcp.server.mcpserver import MCPServer
        assert isinstance(server, MCPServer)


class TestMcpServerNoMcp:
    """Tests that run regardless of whether mcp is installed."""

    def test_has_mcp_is_bool(self):
        assert isinstance(HAS_MCP, bool)

    def test_module_importable_without_mcp(self):
        # If mcp is not present, the module should still import (HAS_MCP=False)
        import htf.mcp_server  # noqa: F401 — just check it doesn't crash

    def test_main_raises_system_exit_without_mcp(self, monkeypatch):
        import htf.mcp_server as ms
        monkeypatch.setattr(ms, "HAS_MCP", False)
        with pytest.raises(SystemExit):
            ms.main()

    def test_build_server_raises_import_error_without_mcp(self, monkeypatch):
        import htf.mcp_server as ms
        monkeypatch.setattr(ms, "HAS_MCP", False)
        with pytest.raises(ImportError):
            ms._build_server()


class TestVerifyBundleTool:
    """Tests for the htf_verify_bundle MCP tool (callable directly)."""

    def _make_full_cert_json(self) -> str:
        """Produce a real full certificate JSON using rayleigh_certificate."""
        pytest.importorskip("flint")
        from htf.rayleigh_cert import rayleigh_certificate, verify_rayleigh_certificate
        H = np.diag([-1.0, 0.0, 1.0])
        psi = np.array([1.0, 0.0, 0.0])
        cert = rayleigh_certificate(H, psi)
        cert = verify_rayleigh_certificate(cert)
        return cert.to_full_json()

    def test_verify_bundle_pass(self):
        from htf.mcp_server import _build_server
        pytest.importorskip("flint")
        pytest.importorskip("mcp")
        server = _build_server()
        cert_json = self._make_full_cert_json()
        # Call the tool function directly by looking it up
        tool_fn = None
        for tool in server._tool_manager._tools.values():
            if tool.name == "htf_verify_bundle":
                tool_fn = tool.fn
                break
        assert tool_fn is not None, "htf_verify_bundle tool not registered"
        result_str = tool_fn(cert_json=cert_json)
        result = json.loads(result_str)
        assert result["verified"] is True
        assert "PASS" in result["message"]

    def test_verify_bundle_bad_json(self):
        from htf.mcp_server import _build_server
        pytest.importorskip("mcp")
        server = _build_server()
        tool_fn = None
        for tool in server._tool_manager._tools.values():
            if tool.name == "htf_verify_bundle":
                tool_fn = tool.fn
                break
        assert tool_fn is not None
        result = json.loads(tool_fn(cert_json="not-json"))
        assert result["verified"] is False
        assert "JSON parse error" in result["message"]

    def test_verify_bundle_tampered_claim(self):
        from htf.mcp_server import _build_server
        pytest.importorskip("flint")
        pytest.importorskip("mcp")
        server = _build_server()
        cert_json = self._make_full_cert_json()
        cert_dict = json.loads(cert_json)
        cert_dict["claim"] = "E0 ≤ -999.0  [tampered]"
        tampered_json = json.dumps(cert_dict)
        tool_fn = None
        for tool in server._tool_manager._tools.values():
            if tool.name == "htf_verify_bundle":
                tool_fn = tool.fn
                break
        result = json.loads(tool_fn(cert_json=tampered_json))
        assert result["verified"] is False
