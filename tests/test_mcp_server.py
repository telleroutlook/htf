"""Tests for htf/mcp_server.py — MCP server construction and tool listing."""
import json
import pytest

from htf.mcp_server import _build_server, HAS_MCP


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
