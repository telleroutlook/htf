"""HTF §7 — String-diagram node-graph visualization.

Converts an HTF ``Diagram`` to a Cytoscape.js-compatible JSON graph and
generates a self-contained HTML prototype for interactive exploration.

Provides
--------
* ``diagram_to_dict``  — serialize a Diagram to a ``{nodes, edges}`` dict.
* ``diagram_to_html``  — generate a standalone HTML string.
* ``save_diagram_html``— write the HTML to a file.

Honest scope [工程]
-------------------
* Layout is a simple depth-first left-to-right (Then) / top-to-bottom
  (Tensor) assignment; proper Sugiyama or force-directed layout is ``[研究]``.
* Rendering uses Cytoscape.js loaded from CDN; no build step required.
* Only ``Box``, ``Id``, ``Then``, ``Tensor`` nodes are handled.
"""
from __future__ import annotations

import json
from typing import Any

from .topology import Box, Diagram, Id, Tensor, Then

# ────────────────────── internal graph builder ────────────────────────────

_WIRE_SPACING = 80   # px between parallel wires
_BOX_STEP     = 220  # px between sequential boxes


def _visit(
    diag: Diagram,
    nodes: list[dict],
    edges: list[dict],
    counter: list[int],
    x: float,
    y: float,
) -> tuple[list[tuple[str, int]], list[tuple[str, int]], float, float]:
    """Recursively walk the diagram tree and populate *nodes* / *edges*.

    Returns
    -------
    in_ports  : list of (node_id, port_index) for each input wire.
    out_ports : list of (node_id, port_index) for each output wire.
    x_end     : x-coordinate just past the rightmost element placed.
    y_end     : y-coordinate just past the bottommost element placed.
    """

    def _new_id(prefix: str = "n") -> str:
        counter[0] += 1
        return f"{prefix}{counter[0]}"

    if isinstance(diag, Box):
        nid = _new_id("b")
        n_dom = len(diag.dom)
        n_cod = len(diag.cod)
        height = max(n_dom, n_cod, 1) * _WIRE_SPACING
        nodes.append({
            "data": {
                "id": nid,
                "label": diag.name,
                "type": "box",
                "dom": [w.name for w in diag.dom],
                "cod": [w.name for w in diag.cod],
            },
            "position": {"x": x, "y": y + height / 2},
            "classes": "box",
        })
        in_ports  = [(nid, i) for i in range(n_dom)]
        out_ports = [(nid, i) for i in range(n_cod)]
        return in_ports, out_ports, x + _BOX_STEP, y + height

    if isinstance(diag, Id):
        wires = list(diag.ty)
        ports = []
        for i, w in enumerate(wires):
            nid = _new_id("w")
            nodes.append({
                "data": {"id": nid, "label": w.name, "type": "wire"},
                "position": {"x": x, "y": y + i * _WIRE_SPACING},
                "classes": "wire",
            })
            ports.append((nid, 0))
        n = len(wires) or 1
        return ports, ports, x + _BOX_STEP // 2, y + n * _WIRE_SPACING

    if isinstance(diag, Then):
        in_f, out_f, x_mid, y_f = _visit(diag.f, nodes, edges, counter, x, y)
        in_g, out_g, x_end, y_g = _visit(diag.g, nodes, edges, counter, x_mid, y)
        # Connect out_f → in_g
        for (n1, p1), (n2, p2) in zip(out_f, in_g):
            eid = _new_id("e")
            edges.append({"data": {"id": eid, "source": n1, "target": n2}})
        return in_f, out_g, x_end, max(y_f, y_g)

    if isinstance(diag, Tensor):
        in_f, out_f, x_f, y_after_f = _visit(diag.f, nodes, edges, counter, x, y)
        in_g, out_g, x_g, y_after_g = _visit(diag.g, nodes, edges, counter, x, y_after_f)
        return in_f + in_g, out_f + out_g, max(x_f, x_g), y_after_g

    # Fallback: unknown subclass — treat as opaque box
    nid = _new_id("u")
    nodes.append({
        "data": {"id": nid, "label": type(diag).__name__, "type": "box"},
        "position": {"x": x, "y": y},
        "classes": "box",
    })
    return [(nid, 0)], [(nid, 0)], x + _BOX_STEP, y + _WIRE_SPACING


# ────────────────────── public API ────────────────────────────────────────

def diagram_to_dict(diagram: Diagram) -> dict[str, Any]:
    """Serialize a ``Diagram`` to a Cytoscape.js-compatible graph dict.

    Returns
    -------
    ``{"nodes": [...], "edges": [...]}`` where each element follows the
    Cytoscape.js element format (``data`` + ``position`` keys).
    """
    nodes: list[dict] = []
    edges: list[dict] = []
    counter = [0]
    _visit(diagram, nodes, edges, counter, x=100.0, y=50.0)
    return {"nodes": nodes, "edges": edges}


# ────────────────────── HTML template ─────────────────────────────────────

_HTML_TEMPLATE = """\
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>{title}</title>
<style>
  body  {{ margin: 0; background: #1a1a2e; font-family: monospace; color: #eee; }}
  #cy   {{ width: 100vw; height: 90vh; }}
  h1    {{ margin: 8px 16px; font-size: 1rem; color: #a0cfff; }}
</style>
</head>
<body>
<h1>HTF Diagram — {title}</h1>
<div id="cy"></div>
<script src="https://unpkg.com/cytoscape@3.28.1/dist/cytoscape.min.js"></script>
<script>
var elements = {elements_json};

var cy = cytoscape({{
  container: document.getElementById('cy'),
  elements: elements,
  style: [
    {{
      selector: '.box',
      style: {{
        'shape': 'rectangle',
        'background-color': '#2d6a9f',
        'label': 'data(label)',
        'color': '#fff',
        'font-size': '11px',
        'text-valign': 'center',
        'text-halign': 'center',
        'width': '120px',
        'height': '40px',
        'border-width': 2,
        'border-color': '#5bc0de',
      }}
    }},
    {{
      selector: '.wire',
      style: {{
        'shape': 'ellipse',
        'background-color': '#3a3a5c',
        'label': 'data(label)',
        'color': '#aaa',
        'font-size': '9px',
        'text-valign': 'center',
        'width': '60px',
        'height': '24px',
        'border-width': 1,
        'border-color': '#666',
      }}
    }},
    {{
      selector: 'edge',
      style: {{
        'width': 2,
        'line-color': '#5bc0de',
        'target-arrow-color': '#5bc0de',
        'target-arrow-shape': 'triangle',
        'curve-style': 'bezier',
      }}
    }}
  ],
  layout: {{ name: 'preset' }},
  userZoomingEnabled: true,
  userPanningEnabled: true,
}});
cy.fit(40);
</script>
</body>
</html>
"""


def diagram_to_html(diagram: Diagram, title: str = "HTF Diagram") -> str:
    """Generate a self-contained HTML string visualizing ``diagram``.

    The returned string can be saved to a ``.html`` file and opened in
    any browser — no build step or server required.

    Parameters
    ----------
    diagram : HTF :class:`~htf.topology.Diagram` to visualize.
    title   : page title and header text.

    Returns
    -------
    HTML string (UTF-8).
    """
    graph = diagram_to_dict(diagram)
    elements = graph["nodes"] + graph["edges"]
    return _HTML_TEMPLATE.format(
        title=title,
        elements_json=json.dumps(elements, indent=2),
    )


def save_diagram_html(diagram: Diagram, path: str, title: str = "HTF Diagram") -> None:
    """Write a self-contained HTML visualization of ``diagram`` to ``path``.

    Parameters
    ----------
    diagram : HTF :class:`~htf.topology.Diagram` to visualize.
    path    : file path to write (e.g. ``"out/diagram.html"``).
    title   : page title and header text.
    """
    html = diagram_to_html(diagram, title=title)
    with open(path, "w", encoding="utf-8") as f:
        f.write(html)
