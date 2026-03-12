# -*- coding: utf-8 -*-
"""
gerber_parser.py — Gerber file and Altium stackup XML parsing

Provides:
  parse_gerber(path)        -> list[TraceSegment]
  parse_stackup_xml(path)   -> list[StackupLayer]
  find_clicked_trace(...)   -> TraceSegment | None
  default_stackup()         -> list[StackupLayer]
"""

from __future__ import annotations
import math
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path


# ─────────────────────────────────────────────────────────────────────────────
# Data structures
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class TraceSegment:
    x1: float          # mm, PCB coordinate space
    y1: float          # mm
    x2: float          # mm
    y2: float          # mm
    width: float       # mm
    layer: str         # e.g. 'F_Cu', 'B_Cu'
    aperture_shape: str = 'round'   # 'round' | 'rect'

    @property
    def length(self) -> float:
        return math.hypot(self.x2 - self.x1, self.y2 - self.y1)

    @property
    def angle_deg(self) -> float:
        return math.degrees(math.atan2(self.y2 - self.y1, self.x2 - self.x1))

    @property
    def is_horizontal(self) -> bool:
        a = abs(self.angle_deg % 180)
        return a < 1.0 or a > 179.0

    @property
    def is_vertical(self) -> bool:
        a = abs(self.angle_deg % 180)
        return abs(a - 90.0) < 1.0

    @property
    def is_diagonal(self) -> bool:
        return not (self.is_horizontal or self.is_vertical)

    @property
    def midpoint(self) -> tuple[float, float]:
        return ((self.x1 + self.x2) / 2, (self.y1 + self.y2) / 2)

    @property
    def center_x(self) -> float:
        return (self.x1 + self.x2) / 2

    @property
    def center_y(self) -> float:
        return (self.y1 + self.y2) / 2


@dataclass
class StackupLayer:
    name: str
    layer_type: str       # 'Conductor' | 'Dielectric'
    thickness_mm: float
    permittivity: float   # εr (1.0 for conductors)
    loss_tangent: float   # tan δ (0.0 for conductors)
    conductivity: float   # S/m  (5.8e7 copper, 0 dielectric)
    z_bottom_mm: float = field(default=0.0)
    z_top_mm:    float = field(default=0.0)

    @property
    def is_conductor(self) -> bool:
        return self.layer_type == 'Conductor'

    @property
    def is_dielectric(self) -> bool:
        return self.layer_type == 'Dielectric'


# ─────────────────────────────────────────────────────────────────────────────
# Gerber parsing
# ─────────────────────────────────────────────────────────────────────────────

def _gerber_to_mm(value: float, unit: str) -> float:
    """Convert a gerbonara coordinate/dimension to millimetres."""
    if unit == 'in':
        return value * 25.4
    if unit == 'mm':
        return value
    # 'm' would be unusual but handle it anyway
    if unit == 'm':
        return value * 1000.0
    return value   # unknown unit — pass through unchanged


def parse_gerber(path: str) -> list[TraceSegment]:
    """
    Parse a Gerber RS-274X file and return all line trace segments.

    gerbonara returns coordinates in the file's native unit (``obj.unit``
    is ``'in'`` or ``'mm'``).  We convert everything to mm.
    Zero-length flashes (pads) are skipped.

    Note: Altium legacy Gerbers sometimes wrap layers inside multiple nested
    ``%SRX1Y1I0J0*%`` step-repeat blocks (all no-ops: X=1,Y=1,I=0,J=0).
    If gerbonara raises a SyntaxError for this, all SR lines are stripped and
    the file is re-parsed from the raw string — the geometry is identical.
    """
    try:
        import gerbonara
    except ImportError:
        raise ImportError(
            "gerbonara is required: pip install gerbonara"
        )

    import re as _re

    layer_name = Path(path).stem

    def _open_gerber(raw_text: str | None = None):
        if raw_text is None:
            return gerbonara.GerberFile.open(str(path))
        return gerbonara.GerberFile.from_string(raw_text)

    import warnings as _warnings

    def _try_parse_with_fixes(exc_first: SyntaxError):
        """
        Two-stage workaround for non-conformant Altium Gerbers:

        Stage 1 — nested SR step-repeat:
          Altium legacy files wrap layers in no-op %SRX1Y1I0J0*% blocks that
          gerbonara rejects. Strip all %SR...% lines and re-parse.

        Stage 2 — bare / modal coordinate lines (ambiguous D-code):
          Altium also emits coordinate-only lines (no D01/D02/D03 suffix) in
          positions where the last explicit op was not D01.  gerbonara raises
          SyntaxError: "Ambiguous coordinate statement … This is garbage."
          Fix: scan the raw text and append the last-seen D-code to every
          bare coordinate line that lacks one, then re-parse.
        """
        raw = Path(path).read_text(encoding='utf-8', errors='replace')

        # Stage 1: remove nested SR blocks
        raw_fixed = _re.sub(r'%SR[^*]*\*%\r?\n?', '', raw)

        try:
            return _open_gerber(raw_fixed)
        except SyntaxError:
            pass

        # Stage 2: inject missing D-codes onto bare coord lines
        last_d = 'D01'
        in_region = False
        coord_only = _re.compile(
            r'^((?:X[0-9+\-]+)?(?:Y[0-9+\-]+)?)(\*)\s*$'
        )
        d_explicit = _re.compile(r'D0*([123])\s*\*')
        lines_out = []
        for line in raw_fixed.splitlines(keepends=True):
            stripped = line.strip()
            if 'G36' in stripped:
                in_region = True
            if 'G37' in stripped:
                in_region = False

            # Track last explicit D-code outside region blocks
            if not stripped.startswith('%') and not in_region:
                m = d_explicit.search(stripped)
                if m:
                    last_d = f'D0{m.group(1)}'

            # Append explicit D-code to bare coordinate lines outside regions
            bare = coord_only.match(stripped)
            if bare and bare.group(1) and not in_region:
                lines_out.append(bare.group(1) + last_d + '*\n')
                continue

            lines_out.append(line)

        try:
            return _open_gerber(''.join(lines_out))
        except SyntaxError as exc_final:
            raise SyntaxError(
                f"Could not repair non-conformant Gerber ({exc_final})"
            ) from exc_first

    try:
        gbr = _open_gerber()
    except SyntaxError as exc:
        gbr = _try_parse_with_fixes(exc)

    segments: list[TraceSegment] = []

    for obj in gbr.objects:
        # ── Line primitive ────────────────────────────────────────────────────
        # gerbonara exposes line strokes as objects with x1,y1,x2,y2 attrs.
        # Regions (polygon fills), Flashes (pads), and Arcs do not have x1.
        x1 = getattr(obj, 'x1', None)
        if x1 is None:
            continue

        unit = getattr(obj, 'unit', 'mm') or 'mm'
        x1_mm = _gerber_to_mm(obj.x1, unit)
        y1_mm = _gerber_to_mm(obj.y1, unit)
        x2_mm = _gerber_to_mm(obj.x2, unit)
        y2_mm = _gerber_to_mm(obj.y2, unit)

        # Skip zero-length
        if math.hypot(x2_mm - x1_mm, y2_mm - y1_mm) < 1e-6:
            continue

        # Determine width from aperture
        ap = getattr(obj, 'aperture', None)
        if ap is None:
            continue

        if hasattr(ap, 'diameter') and ap.diameter is not None:
            width_mm = _gerber_to_mm(ap.diameter, unit)
            shape = 'round'
        elif hasattr(ap, 'w') and hasattr(ap, 'h') and ap.w is not None:
            width_mm = _gerber_to_mm(min(ap.w, ap.h), unit)
            shape = 'rect'
        else:
            # Unknown aperture type — use a fallback
            size = getattr(ap, 'size', None)
            if size is not None:
                try:
                    width_mm = _gerber_to_mm(float(size), unit)
                except (TypeError, ValueError):
                    continue
            else:
                continue
            shape = 'round'

        if width_mm <= 0:
            continue

        segments.append(TraceSegment(
            x1=x1_mm, y1=y1_mm, x2=x2_mm, y2=y2_mm,
            width=width_mm, layer=layer_name,
            aperture_shape=shape,
        ))

    return segments


# ─────────────────────────────────────────────────────────────────────────────
# Stackup XML parsing
# ─────────────────────────────────────────────────────────────────────────────

def _parse_thickness(raw: str, default_mm: float = 0.0) -> float:
    """Parse a thickness string like '0.035mm', '1.4mil', '63' (mils)."""
    if not raw:
        return default_mm
    raw = raw.strip().lower()
    try:
        if raw.endswith('mm'):
            return float(raw[:-2].strip())
        elif raw.endswith('mil') or raw.endswith('mils'):
            mils = float(raw.rstrip('s').rstrip('mil').strip())
            return mils * 0.0254
        elif raw.endswith('um'):
            return float(raw[:-2].strip()) / 1000.0
        else:
            # Assume mils (common in legacy Altium exports)
            return float(raw) * 0.0254
    except ValueError:
        return default_mm


def _parse_float_attr(elem, attr: str, default: float = 0.0) -> float:
    raw = elem.get(attr, '')
    try:
        return float(raw)
    except ValueError:
        return default


def _assign_z_positions(layers: list[StackupLayer]) -> None:
    """
    Compute z_bottom_mm / z_top_mm for each layer.
    Altium lists layers top-down; we want z=0 at the bottom (ground plane side).
    Reverse so z accumulates from the bottom up.
    """
    z = 0.0
    for layer in reversed(layers):
        layer.z_bottom_mm = z
        z += layer.thickness_mm
        layer.z_top_mm = z


def parse_stackup_xml(path: str) -> list[StackupLayer]:
    """
    Parse an Altium stackup XML export.

    Handles two known schemas:
      Schema A (Altium 20+):  <Layer Type="Conductor/Dielectric"> with child elements
      Schema B (legacy):      <LAYER KIND="SIGNAL/DIELECTRIC" THICKNESS="..." ...>

    Returns layers ordered bottom-to-top with z positions assigned.
    """
    try:
        tree = ET.parse(path)
    except ET.ParseError as exc:
        raise ValueError(f"Cannot parse stackup XML: {exc}") from exc

    root = tree.getroot()
    layers: list[StackupLayer] = []

    # ── Schema A ──────────────────────────────────────────────────────────────
    for elem in root.iter('Layer'):
        name  = elem.get('Name', '') or elem.get('name', '')
        ltype = elem.get('Type', '') or elem.get('type', '')
        if not ltype:
            continue

        ltype_up = ltype.strip().title()   # normalise case

        # Thickness
        t_elem = elem.find('Thickness')
        thick_mm = _parse_thickness(
            t_elem.get('Value', '') if t_elem is not None else '',
            default_mm=0.035
        )

        if any(k in ltype_up for k in ('Conductor', 'Signal', 'Plane', 'Power')):
            layers.append(StackupLayer(
                name=name, layer_type='Conductor',
                thickness_mm=thick_mm,
                permittivity=1.0, loss_tangent=0.0,
                conductivity=5.8e7,
            ))
        elif 'Dielectric' in ltype_up or 'Prepreg' in ltype_up or 'Core' in ltype_up:
            er_elem   = elem.find('Permittivity')
            tand_elem = elem.find('LossTangent')
            er   = _parse_float_attr(er_elem,   'Value', 4.4)   if er_elem   is not None else 4.4
            tand = _parse_float_attr(tand_elem, 'Value', 0.02)  if tand_elem is not None else 0.02
            layers.append(StackupLayer(
                name=name, layer_type='Dielectric',
                thickness_mm=thick_mm,
                permittivity=er, loss_tangent=tand,
                conductivity=0.0,
            ))

    # ── Schema C (Altium 21+ StackupDocument / .stackupx) ─────────────────────
    # Root: <StackupDocument xmlns="http://altium.com/ns/LayerStackManager">
    # Layers:  <Layer TypeId="GUID" Name="..."><Properties>
    #            <Property Name="Thickness">1.4mil</Property>
    #            <Property Name="DielectricConstant">4.4</Property>
    #          </Properties></Layer>
    # TypeId GUIDs identify conductor vs dielectric.
    if not layers:
        # The namespace appears as a prefix on every tag in ElementTree
        _ns = '{http://altium.com/ns/LayerStackManager}'
        # Known conductor TypeId (signal/power copper layers)
        _COND_IDS = {'f4eccd87-2cfb-4f37-be50-4f3a272b4d01'}
        # Known dielectric TypeIds (prepreg and core)
        _DIEL_IDS = {
            '1a79611a-039d-4d40-a204-53c26c50f8b5',   # Prepreg
            '136c62ef-1fa6-4897-ae71-7e797b632b92',   # Core
        }

        for elem in root.iter(f'{_ns}Layer'):
            name    = elem.get('Name', '')
            type_id = (elem.get('TypeId', '') or '').lower()

            # Build a property dict from <Property Name="...">value</Property>
            props: dict[str, str] = {}
            for prop in elem.iter(f'{_ns}Property'):
                pname = prop.get('Name', '')
                if pname:
                    props[pname] = (prop.text or '').strip()

            thick_mm = _parse_thickness(props.get('Thickness', ''),
                                        default_mm=0.035)

            if type_id in _COND_IDS:
                layers.append(StackupLayer(
                    name=name, layer_type='Conductor',
                    thickness_mm=thick_mm,
                    permittivity=1.0, loss_tangent=0.0,
                    conductivity=5.8e7,
                ))
            elif type_id in _DIEL_IDS:
                try:
                    er = float(props.get('DielectricConstant', '4.4') or '4.4')
                except ValueError:
                    er = 4.4
                try:
                    tand = float(props.get('LossTangent', '0.02') or '0.02')
                except ValueError:
                    tand = 0.02
                layers.append(StackupLayer(
                    name=name, layer_type='Dielectric',
                    thickness_mm=thick_mm,
                    permittivity=er, loss_tangent=tand,
                    conductivity=0.0,
                ))

    # ── Schema B (legacy Altium PCBDoc XML) ───────────────────────────────────
    if not layers:
        for elem in root.iter('LAYER'):
            name  = elem.get('NAME', '') or elem.get('Name', '')
            kind  = elem.get('KIND', '').upper()
            raw_t = elem.get('THICKNESS', '0')
            thick_mm = _parse_thickness(raw_t, default_mm=0.035)

            if kind in ('SIGNAL', 'PLANE', 'MIXED'):
                layers.append(StackupLayer(
                    name=name, layer_type='Conductor',
                    thickness_mm=thick_mm,
                    permittivity=1.0, loss_tangent=0.0,
                    conductivity=5.8e7,
                ))
            elif kind == 'DIELECTRIC':
                er   = float(elem.get('DIELECTRICCONST',   '4.4'))
                tand = float(elem.get('DISSIPATIONFACTOR', '0.02'))
                layers.append(StackupLayer(
                    name=name, layer_type='Dielectric',
                    thickness_mm=thick_mm,
                    permittivity=er, loss_tangent=tand,
                    conductivity=0.0,
                ))

    if not layers:
        raise ValueError(
            "No layers found in stackup XML. "
            "Check that the file is an Altium layer stackup export."
        )

    _assign_z_positions(layers)
    return layers


def default_stackup() -> list[StackupLayer]:
    """Standard 2-layer FR4 board: 1.6mm core, 35µm copper top and bottom."""
    layers = [
        StackupLayer('Top Copper',  'Conductor',  0.035, 1.0, 0.0,  5.8e7),
        StackupLayer('FR4 Core',    'Dielectric', 1.6,   4.4, 0.02, 0.0),
        StackupLayer('Bot Copper',  'Conductor',  0.035, 1.0, 0.0,  5.8e7),
    ]
    _assign_z_positions(layers)
    return layers


# ─────────────────────────────────────────────────────────────────────────────
# Trace selection helpers
# ─────────────────────────────────────────────────────────────────────────────

def _dist_point_to_segment(px: float, py: float,
                            x1: float, y1: float,
                            x2: float, y2: float) -> float:
    """Minimum distance from point (px,py) to line segment (x1,y1)-(x2,y2)."""
    dx, dy = x2 - x1, y2 - y1
    if dx == 0 and dy == 0:
        return math.hypot(px - x1, py - y1)
    t = max(0.0, min(1.0, ((px - x1) * dx + (py - y1) * dy) / (dx * dx + dy * dy)))
    return math.hypot(px - (x1 + t * dx), py - (y1 + t * dy))


def find_clicked_trace(segments: list[TraceSegment],
                       click_x: float, click_y: float,
                       max_dist_mm: float = 0.5) -> TraceSegment | None:
    """
    Return the trace segment nearest to (click_x, click_y) in mm.
    The search radius is max(max_dist_mm, half the nearest trace width).
    """
    best: TraceSegment | None = None
    best_d = float('inf')

    for seg in segments:
        d = _dist_point_to_segment(click_x, click_y,
                                   seg.x1, seg.y1, seg.x2, seg.y2)
        if d < best_d:
            best_d = d
            best = seg

    if best is None:
        return None

    threshold = max(max_dist_mm, best.width / 2)
    return best if best_d <= threshold else None


# ─────────────────────────────────────────────────────────────────────────────
# Quick self-test (run as script)
# ─────────────────────────────────────────────────────────────────────────────

def _print_stackup(stack: list) -> None:
    for lyr in stack:
        extras = (f"  er={lyr.permittivity}  tand={lyr.loss_tangent}"
                  if lyr.is_dielectric else "")
        print(f"  {lyr.name:22s}  {lyr.layer_type:10s}  "
              f"t={lyr.thickness_mm:.4f}mm  "
              f"z=[{lyr.z_bottom_mm:.4f}, {lyr.z_top_mm:.4f}]mm"
              + extras)


if __name__ == '__main__':
    import sys
    import argparse

    ap = argparse.ArgumentParser(
        description=(
            'Parse a Gerber file and optionally an Altium stackup export.\n'
            'Both .xml and .stackupx (Altium 21+) are accepted for --stackup.'
        )
    )
    ap.add_argument('gerber',
                    help='Path to the signal copper Gerber (.gbr) file')
    ap.add_argument('--stackup', metavar='PATH',
                    help='Altium stackup export (.xml or .stackupx) — optional')
    ap.add_argument('--ground', metavar='PATH',
                    help='Ground-plane Gerber file — optional, parsed for info only')
    args = ap.parse_args()

    # ── Signal Gerber ──────────────────────────────────────────────────────────
    print(f"Parsing Gerber: {args.gerber}")
    try:
        segs = parse_gerber(args.gerber)
    except Exception as exc:
        print(f"  ERROR: {exc}")
        sys.exit(1)

    print(f"  Found {len(segs)} trace segments")
    if segs:
        widths = sorted(set(round(s.width, 3) for s in segs))
        all_x = [s.x1 for s in segs] + [s.x2 for s in segs]
        all_y = [s.y1 for s in segs] + [s.y2 for s in segs]
        print(f"  Unique widths (mm): {widths[:20]}")
        print(f"  X range: [{min(all_x):.2f}, {max(all_x):.2f}] mm")
        print(f"  Y range: [{min(all_y):.2f}, {max(all_y):.2f}] mm")
        horz = sum(1 for s in segs if s.is_horizontal)
        vert = sum(1 for s in segs if s.is_vertical)
        diag = sum(1 for s in segs if s.is_diagonal)
        print(f"  Horizontal: {horz}  Vertical: {vert}  Diagonal: {diag}")
    else:
        print("  (no line strokes found)")
        print("  Altium tip: re-export with 'Use Aperture & Flash Draws' enabled")
        print("  in File → Fabrication Outputs → Gerber Files → Advanced tab.")

    # ── Ground Gerber (info only) ──────────────────────────────────────────────
    if args.ground:
        print(f"\nParsing ground Gerber: {args.ground}")
        if not Path(args.ground).exists():
            print(f"  ERROR: file not found: {args.ground}")
        else:
            try:
                gnd_segs = parse_gerber(args.ground)
                print(f"  Found {len(gnd_segs)} segments")
            except Exception as exc:
                print(f"  ERROR: {exc}")

    # ── Stackup ────────────────────────────────────────────────────────────────
    if args.stackup:
        print(f"\nParsing stackup: {args.stackup}")
        stk_path = Path(args.stackup)
        if not stk_path.exists():
            print(f"  ERROR: file not found — {stk_path}")
            print(f"  Hint: in SampleGerbers the file is called 'stackup.xml'.")
            print("  Falling back to default FR4 stackup.")
            stack = default_stackup()
        else:
            try:
                stack = parse_stackup_xml(str(stk_path))
            except Exception as exc:
                print(f"  ERROR: {exc}")
                print("  Falling back to default FR4 stackup.")
                stack = default_stackup()
        _print_stackup(stack)
    else:
        print("\nNo --stackup specified — using default FR4 stackup:")
        _print_stackup(default_stackup())
