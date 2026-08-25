"""Minimal dependency-free DXF R12 writer (CIRCLE / LINE / ARC)."""
from __future__ import annotations
from typing import List, Tuple

Point = Tuple[float, float]


class DXFWriter:
    def __init__(self, units: str = "inches"):
        self.entities: List[str] = []
        self.layers_used: set = set()
        self.units = units

    def _e(self, code: int, value) -> str:
        return f"{code}\n{value}\n"

    def _layer(self, layer: str) -> str:
        self.layers_used.add(layer)
        return self._e(8, layer)

    def circle(self, x: float, y: float, r: float, layer: str = "CUT") -> None:
        if r <= 0:
            raise ValueError(f"radius must be > 0, got {r}")
        self.entities.append(
            self._e(0, "CIRCLE")
            + self._layer(layer)
            + self._e(10, f"{x:.6f}")
            + self._e(20, f"{y:.6f}")
            + self._e(30, "0.0")
            + self._e(40, f"{r:.6f}")
        )

    def line(self, p1: Point, p2: Point, layer: str = "CUT") -> None:
        x1, y1 = p1
        x2, y2 = p2
        self.entities.append(
            self._e(0, "LINE")
            + self._layer(layer)
            + self._e(10, f"{x1:.6f}")
            + self._e(20, f"{y1:.6f}")
            + self._e(30, "0.0")
            + self._e(11, f"{x2:.6f}")
            + self._e(21, f"{y2:.6f}")
            + self._e(31, "0.0")
        )

    def arc(self, x: float, y: float, r: float, a0_deg: float, a1_deg: float, layer: str = "CUT") -> None:
        """CCW arc, angles in degrees from +X."""
        if r <= 0:
            raise ValueError(f"radius must be > 0, got {r}")
        self.entities.append(
            self._e(0, "ARC")
            + self._layer(layer)
            + self._e(10, f"{x:.6f}")
            + self._e(20, f"{y:.6f}")
            + self._e(30, "0.0")
            + self._e(40, f"{r:.6f}")
            + self._e(50, f"{a0_deg:.6f}")
            + self._e(51, f"{a1_deg:.6f}")
        )

    def rect(self, w: float, h: float, layer: str = "CUT") -> None:
        self.line((0, 0), (w, 0), layer)
        self.line((w, 0), (w, h), layer)
        self.line((w, h), (0, h), layer)
        self.line((0, h), (0, 0), layer)

    def to_string(self) -> str:
        insunits = 1 if self.units.lower().startswith("in") else 4
        header = (
            "0\nSECTION\n2\nHEADER\n9\n$ACADVER\n1\nAC1009\n"
            f"9\n$INSUNITS\n70\n{insunits}\n"
            "9\n$EXTMIN\n10\n-1e6\n20\n-1e6\n30\n0.0\n"
            "9\n$EXTMAX\n10\n1e6\n20\n1e6\n30\n0.0\n0\nENDSEC\n"
        )
        layers = sorted(self.layers_used) or ["CUT"]
        table = "0\nSECTION\n2\nTABLES\n0\nTABLE\n2\nLAYER\n70\n" + str(len(layers)) + "\n"
        for lyr in layers:
            table += f"0\nLAYER\n2\n{lyr}\n70\n0\n62\n7\n6\nCONTINUOUS\n"
        table += "0\nENDTAB\n0\nENDSEC\n"
        ents = "0\nSECTION\n2\nENTITIES\n" + "".join(self.entities) + "0\nENDSEC\n"
        return header + table + ents + "0\nEOF\n"

    def save(self, path: str) -> None:
        with open(path, "w", encoding="ascii", newline="\n") as f:
            f.write(self.to_string())

    def to_bytes(self) -> bytes:
        return self.to_string().encode("ascii")
