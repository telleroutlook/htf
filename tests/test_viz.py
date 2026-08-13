"""Tests for htf/viz.py — string-diagram node-graph visualization."""
import json

from htf.topology import Box, Id, Wire
from htf.viz import diagram_to_dict, diagram_to_html, save_diagram_html

# ──────── helpers ─────────────────────────────────────────────────────────

def _two_wire() -> tuple[Wire, Wire]:
    return Wire("a", 2), Wire("b", 2)

def _simple_box(name="f") -> Box:
    w = Wire("x", 2)
    return Box(name, (w,), (w,))

def _bell_diagram():
    """H on q0, then CNOT(q0,q1) — a two-qubit circuit as an HTF diagram."""
    q0 = Wire("q0", 2)
    q1 = Wire("q1", 2)
    h  = Box("H",    (q0,),      (q0,))
    cx = Box("CX",   (q0, q1),   (q0, q1))
    # H ⊗ Id(q1) >> CX
    h_layer  = h @ Id((q1,))
    return h_layer >> cx


# ──────── TestDiagramToDict ───────────────────────────────────────────────

class TestDiagramToDict:

    def test_box_produces_node(self):
        b = _simple_box()
        d = diagram_to_dict(b)
        assert len(d["nodes"]) >= 1
        assert any(n["data"]["type"] == "box" for n in d["nodes"])

    def test_id_produces_wire_nodes(self):
        w = Wire("x", 2)
        d = diagram_to_dict(Id((w,)))
        assert any(n["data"]["type"] == "wire" for n in d["nodes"])

    def test_then_produces_edge(self):
        w = Wire("x", 2)
        f = Box("f", (w,), (w,))
        g = Box("g", (w,), (w,))
        d = diagram_to_dict(f >> g)
        assert len(d["edges"]) >= 1

    def test_tensor_has_no_cross_edges(self):
        w = Wire("x", 2)
        f = Box("f", (w,), (w,))
        g = Box("g", (w,), (w,))
        d = diagram_to_dict(f @ g)
        # Tensor product has no connecting edges
        assert len(d["edges"]) == 0

    def test_returns_dict_with_nodes_and_edges(self):
        d = diagram_to_dict(_simple_box())
        assert "nodes" in d
        assert "edges" in d

    def test_node_has_required_keys(self):
        d = diagram_to_dict(_simple_box())
        for node in d["nodes"]:
            assert "data" in node
            assert "id" in node["data"]
            assert "position" in node

    def test_edge_has_source_and_target(self):
        w = Wire("x", 2)
        d = diagram_to_dict(Box("f", (w,), (w,)) >> Box("g", (w,), (w,)))
        for edge in d["edges"]:
            assert "source" in edge["data"]
            assert "target" in edge["data"]

    def test_box_label_preserved(self):
        d = diagram_to_dict(_simple_box("MyGate"))
        labels = [n["data"]["label"] for n in d["nodes"]]
        assert any("MyGate" in l for l in labels)

    def test_bell_diagram_nodes_and_edges(self):
        d = diagram_to_dict(_bell_diagram())
        assert len(d["nodes"]) >= 2   # H box + CX box + wire nodes
        assert len(d["edges"]) >= 1

    def test_ids_are_unique(self):
        d = diagram_to_dict(_bell_diagram())
        ids = [n["data"]["id"] for n in d["nodes"]]
        assert len(ids) == len(set(ids))

    def test_empty_id_is_handled(self):
        d = diagram_to_dict(Id(()))
        assert isinstance(d, dict)

    def test_json_serializable(self):
        d = diagram_to_dict(_bell_diagram())
        serialized = json.dumps(d)
        assert isinstance(serialized, str)

    def test_position_has_x_and_y(self):
        d = diagram_to_dict(_simple_box())
        for node in d["nodes"]:
            assert "x" in node["position"]
            assert "y" in node["position"]

    def test_three_box_chain(self):
        w = Wire("x", 2)
        a = Box("A", (w,), (w,))
        b = Box("B", (w,), (w,))
        c = Box("C", (w,), (w,))
        d = diagram_to_dict(a >> b >> c)
        box_nodes = [n for n in d["nodes"] if n["data"]["type"] == "box"]
        assert len(box_nodes) == 3
        assert len(d["edges"]) >= 2


# ──────── TestDiagramToHtml ───────────────────────────────────────────────

class TestDiagramToHtml:

    def test_returns_string(self):
        html = diagram_to_html(_simple_box())
        assert isinstance(html, str)

    def test_contains_cytoscape(self):
        html = diagram_to_html(_simple_box())
        assert "cytoscape" in html.lower()

    def test_default_title_in_html(self):
        html = diagram_to_html(_simple_box())
        assert "HTF Diagram" in html

    def test_custom_title_in_html(self):
        html = diagram_to_html(_simple_box(), title="Bell State")
        assert "Bell State" in html

    def test_elements_json_embedded(self):
        html = diagram_to_html(_simple_box())
        assert "elements" in html

    def test_html_structure(self):
        html = diagram_to_html(_simple_box())
        assert "<html" in html
        assert "</html>" in html
        assert "<div id=\"cy\"" in html

    def test_bell_diagram_html(self):
        html = diagram_to_html(_bell_diagram(), title="Bell Prep")
        assert "Bell Prep" in html
        assert "cytoscape" in html.lower()


# ──────── TestSaveDiagramHtml ─────────────────────────────────────────────

class TestSaveDiagramHtml:

    def test_writes_file(self, tmp_path):
        path = str(tmp_path / "test.html")
        save_diagram_html(_simple_box(), path)
        with open(path, encoding="utf-8") as f:
            content = f.read()
        assert "cytoscape" in content.lower()

    def test_custom_title_in_file(self, tmp_path):
        path = str(tmp_path / "bell.html")
        save_diagram_html(_bell_diagram(), path, title="Bell Circuit")
        with open(path, encoding="utf-8") as f:
            content = f.read()
        assert "Bell Circuit" in content

    def test_file_is_utf8(self, tmp_path):
        path = str(tmp_path / "utf8.html")
        save_diagram_html(_simple_box(), path, title="Diàgram")
        with open(path, encoding="utf-8") as f:
            f.read()  # must not raise
