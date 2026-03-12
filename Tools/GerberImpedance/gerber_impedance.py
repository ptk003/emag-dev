# -*- coding: utf-8 -*-
"""
gerber_impedance.py — PCB trace impedance via openEMS FDTD simulation

Usage:
  python gerber_impedance.py F_Cu.gbr [options]

Options:
  --stackup stackup.xml   Altium stackup XML export (default: FR4 1.6mm)
  --ground  B_Cu.gbr      Ground plane Gerber (for display only)
  --f-max   10e9          Maximum simulation frequency in Hz (default 10 GHz)
  --sim-dir /tmp/sim      Simulation output directory (default: auto tempdir)
  --post-proc-only        Skip simulation; re-run post-processing only

Stackup export from Altium:
  Design → Layer Stack Manager → File → Export → save as .xml

Example:
  python gerber_impedance.py board_F_Cu.gbr --stackup stackup.xml --f-max 6e9
"""

from __future__ import annotations
import argparse
import os
import sys
import tempfile
from pathlib import Path

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection

# ── Local modules ─────────────────────────────────────────────────────────────
_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))

from gerber_parser import (
    TraceSegment, StackupLayer,
    parse_gerber, parse_stackup_xml, default_stackup,
    find_clicked_trace,
)
from openems_sim import (
    GerberImpedanceError,
    build_simulation,
    run_and_postprocess,
)
from plots import plot_results, plot_comparison


# ─────────────────────────────────────────────────────────────────────────────
# Colour palette for multi-trace highlighting
# ─────────────────────────────────────────────────────────────────────────────
_HIGHLIGHT_COLORS = [
    'cyan', '#ffff44', '#ff44ff', '#44ff44',
    '#ff8844', '#44ffff', '#ff4444', '#aaaaff',
]


# ─────────────────────────────────────────────────────────────────────────────
# Gerber viewer  (multi-select)
# ─────────────────────────────────────────────────────────────────────────────

def show_gerber_viewer(segments: list[TraceSegment],
                       ground_segments: list[TraceSegment] | None = None
                       ) -> list[TraceSegment]:
    """
    Display a matplotlib Gerber viewer for multi-trace selection.

    Controls
    --------
    - Left-click a trace to add it to the selection (unique colour + number).
    - Left-click a selected trace again to deselect it.
    - Press ESC to clear all selections.
    - Press ENTER (with >= 1 selection) or close the window to confirm.

    Returns a list of selected TraceSegments.  Diagonal segments are rejected
    with an error (the user must deselect them before confirming).
    """
    fig, ax = plt.subplots(figsize=(13, 9))
    ax.set_aspect('equal')
    ax.set_facecolor('#1a1a1a')
    fig.patch.set_facecolor('#111111')

    # ── Ground layer (dimmed) ─────────────────────────────────────────────────
    if ground_segments:
        gnd_lines = [[(s.x1, s.y1), (s.x2, s.y2)] for s in ground_segments]
        gnd_lc = LineCollection(gnd_lines, linewidths=1, colors='#334433', alpha=0.5)
        ax.add_collection(gnd_lc)

    # ── Signal layer ──────────────────────────────────────────────────────────
    if not segments:
        raise GerberImpedanceError("No trace segments found — wrong Gerber file?")

    all_x = [x for s in segments for x in (s.x1, s.x2)]
    all_y = [y for s in segments for y in (s.y1, s.y2)]
    x_span = max(all_x) - min(all_x) if all_x else 1
    y_span = max(all_y) - min(all_y) if all_y else 1
    scale = min(x_span, y_span)

    sig_lines  = [[(s.x1, s.y1), (s.x2, s.y2)] for s in segments]
    sig_widths = [max(0.5, s.width / scale * 200) for s in segments]
    sig_lc = LineCollection(sig_lines, linewidths=sig_widths,
                             colors='#B87333', alpha=0.85)
    ax.add_collection(sig_lc)
    ax.autoscale()
    ax.margins(0.05)

    ax.set_xlabel('X (mm)', color='white')
    ax.set_ylabel('Y (mm)', color='white')
    ax.tick_params(colors='white')
    for spine in ax.spines.values():
        spine.set_edgecolor('#444444')

    ax.set_title(
        'Click traces to select  |  click again to deselect  |  '
        'ENTER to simulate  |  ESC to clear\n'
        '(H/V segments only; multiple selections run as separate simulations)',
        color='white', fontsize=9
    )

    # ── Info box ──────────────────────────────────────────────────────────────
    info_box = ax.text(
        0.02, 0.97, 'Click a trace to select it',
        transform=ax.transAxes, va='top', ha='left',
        fontsize=9, color='white',
        bbox=dict(boxstyle='round,pad=0.4', facecolor='#111111',
                  edgecolor='#555555', alpha=0.9),
        zorder=11
    )

    # ── State ─────────────────────────────────────────────────────────────────
    selected: list[TraceSegment] = []
    highlight_artists: list = []          # line + label artists per selection

    # ── Helpers ───────────────────────────────────────────────────────────────
    def _redraw():
        for art in highlight_artists:
            art.remove()
        highlight_artists.clear()

        for i, seg in enumerate(selected):
            col = _HIGHLIGHT_COLORS[i % len(_HIGHLIGHT_COLORS)]
            # Highlight line
            line, = ax.plot(
                [seg.x1, seg.x2], [seg.y1, seg.y2],
                color=col, linewidth=3, zorder=10, solid_capstyle='round'
            )
            highlight_artists.append(line)
            # Number label at midpoint
            lbl = ax.text(
                (seg.x1 + seg.x2) / 2, (seg.y1 + seg.y2) / 2,
                f' {i + 1}', color=col, fontsize=8,
                fontweight='bold', zorder=12
            )
            highlight_artists.append(lbl)

        if selected:
            lines = []
            for i, seg in enumerate(selected):
                col = _HIGHLIGHT_COLORS[i % len(_HIGHLIGHT_COLORS)]
                orient = ('H' if seg.is_horizontal
                          else 'V' if seg.is_vertical
                          else 'DIAG!')
                flag = '  [!DIAGONAL - deselect]' if seg.is_diagonal else ''
                lines.append(
                    f'[{i + 1}] {orient}  w={seg.width:.3f} mm'
                    f'  L={seg.length:.3f} mm{flag}'
                )
            n = len(selected)
            lines.append(
                f'\n{n} trace{"s" if n > 1 else ""} selected'
                ' — press ENTER to simulate'
            )
            info_box.set_text('\n'.join(lines))
            info_box.set_color('cyan')
            info_box.get_bbox_patch().set_edgecolor('cyan')
        else:
            info_box.set_text('Click a trace to select it')
            info_box.set_color('white')
            info_box.get_bbox_patch().set_edgecolor('#555555')

        fig.canvas.draw_idle()

    def _on_click(event):
        if event.inaxes is not ax or event.button != 1:
            return
        seg = find_clicked_trace(segments, event.xdata, event.ydata)
        if seg is None:
            return
        # Toggle: deselect if already in list (compare by identity / coords)
        for i, s in enumerate(selected):
            if (s is seg or
                    (s.x1 == seg.x1 and s.y1 == seg.y1
                     and s.x2 == seg.x2 and s.y2 == seg.y2)):
                selected.pop(i)
                _redraw()
                return
        selected.append(seg)
        _redraw()

    def _on_key(event):
        if event.key == 'enter' and selected:
            plt.close(fig)
        elif event.key == 'escape':
            selected.clear()
            _redraw()

    fig.canvas.mpl_connect('button_press_event', _on_click)
    fig.canvas.mpl_connect('key_press_event', _on_key)

    plt.tight_layout()
    plt.show(block=True)

    if not selected:
        raise GerberImpedanceError("No trace selected. Exiting.")

    diag = [s for s in selected if s.is_diagonal]
    if diag:
        raise GerberImpedanceError(
            f"{len(diag)} diagonal trace(s) in selection "
            f"({', '.join(f'{s.angle_deg:.1f}deg' for s in diag)}).\n"
            "AddMSLPort requires axis-aligned traces. "
            "Re-open the viewer and deselect diagonal segments."
        )
    return selected


# ─────────────────────────────────────────────────────────────────────────────
# Post-processing helper (shared between normal and --post-proc-only paths)
# ─────────────────────────────────────────────────────────────────────────────

def _postproc_only(ports, sim_dir: str, f: np.ndarray,
                   trace: TraceSegment, stackup: list):
    """Re-run CalcPort on existing sim data and return SimResults."""
    from openems_sim import (SimResults, _find_microstrip_layers,
                             _effective_er, _compute_tdr)
    for p in ports:
        p.CalcPort(sim_dir, f, ref_impedance=50)

    s11 = ports[0].uf_ref / ports[0].uf_inc
    s21 = ports[1].uf_ref / ports[0].uf_inc
    Zin = ports[0].uf_tot / ports[0].if_tot
    Z0_freq = np.abs(ports[0].ZL)

    N = len(f)
    mid = slice(N // 4, 3 * N // 4)
    Z0_scalar = float(np.median(np.real(Z0_freq[mid])))

    _, substrate_layers, _ = _find_microstrip_layers(stackup)
    er    = substrate_layers[0].permittivity
    h_mm  = sum(l.thickness_mm for l in substrate_layers)
    er_eff = _effective_er(er, h_mm, trace.width)
    tdr_time, tdr_rho = _compute_tdr(s11, f, er_eff, trace.length)

    return SimResults(
        freq=f, s11=s11, s21=s21, Zin=Zin,
        Z0_scalar=Z0_scalar, Z0_freq=Z0_freq,
        tdr_time=tdr_time, tdr_rho=tdr_rho,
        trace=trace, stackup=stackup, sim_dir=sim_dir,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Main pipeline
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='PCB trace impedance simulation via openEMS FDTD',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument('gerber',
                        help='Copper layer Gerber file (.gbr)')
    parser.add_argument('--stackup', default=None,
                        help='Altium stackup XML export '
                             '(Design > Layer Stack Manager > File > Export)')
    parser.add_argument('--ground', default=None,
                        help='Ground layer Gerber (display only)')
    parser.add_argument('--f-max', type=float, default=10e9,
                        help='Max simulation frequency Hz (default: 10e9)')
    parser.add_argument('--sim-dir', default=None,
                        help='Simulation output directory (auto if omitted)')
    parser.add_argument('--post-proc-only', action='store_true',
                        help='Skip simulation; re-run post-processing only')
    args = parser.parse_args()

    # ── [1/5] Parse inputs ────────────────────────────────────────────────────
    print(f"\n[1/5] Parsing Gerber: {args.gerber}")
    try:
        segments = parse_gerber(args.gerber)
    except Exception as exc:
        print(f"ERROR: Cannot parse Gerber file: {exc}")
        sys.exit(1)

    if not segments:
        print("ERROR: No trace segments found in this Gerber file.\n"
              "       If you use Altium, re-export with 'Use Aperture & Flash\n"
              "       Draws' enabled — Altium's polygon-fill export produces\n"
              "       Region objects with no trace-width metadata.")
        sys.exit(1)
    print(f"      Found {len(segments)} trace segments.")

    ground_segments = None
    if args.ground:
        try:
            ground_segments = parse_gerber(args.ground)
            print(f"      Ground layer: {len(ground_segments)} segments.")
        except Exception as exc:
            print(f"      WARNING: Could not parse ground Gerber: {exc}")

    if args.stackup:
        print(f"      Parsing stackup: {args.stackup}")
        try:
            stackup = parse_stackup_xml(args.stackup)
            print(f"      Found {len(stackup)} stackup layers.")
        except Exception as exc:
            print(f"      WARNING: {exc}")
            print("      Falling back to default FR4 1.6mm stackup.")
            stackup = default_stackup()
    else:
        print("      No stackup provided — using default FR4 1.6mm stackup.")
        stackup = default_stackup()

    # Stackup summary (ASCII-safe)
    print("      Stackup (bottom -> top):")
    for lyr in stackup:
        if lyr.is_conductor:
            print(f"        Cu  {lyr.thickness_mm*1000:.0f}um"
                  f"  [{lyr.z_bottom_mm:.3f}-{lyr.z_top_mm:.3f}mm]"
                  f"  {lyr.name}")
        else:
            print(f"        er={lyr.permittivity}"
                  f"  h={lyr.thickness_mm:.3f}mm"
                  f"  [{lyr.z_bottom_mm:.3f}-{lyr.z_top_mm:.3f}mm]"
                  f"  {lyr.name}")

    # ── [2/5] Interactive trace selection ─────────────────────────────────────
    print("\n[2/5] Showing Gerber viewer — click traces, ENTER to confirm.")
    try:
        traces = show_gerber_viewer(segments, ground_segments)
    except GerberImpedanceError as exc:
        print(f"ERROR: {exc}")
        sys.exit(1)

    print(f"      Selected {len(traces)} trace(s):")
    for i, t in enumerate(traces):
        print(f"        [{i+1}] w={t.width:.3f}mm  L={t.length:.3f}mm  "
              f"{'horizontal' if t.is_horizontal else 'vertical'}")

    # ── [3/5]–[4/5] Build + run one simulation per trace ─────────────────────
    base_sim_dir = args.sim_dir or tempfile.mkdtemp(prefix='GerberImpedance_')
    f = np.linspace(1e6, args.f_max, 1601)
    all_results = []

    for i, trace in enumerate(traces):
        tag = f"[{i+1}/{len(traces)}]"

        # Each trace gets its own subdirectory when multiple are selected
        if len(traces) > 1:
            sim_dir = str(Path(base_sim_dir) / f'trace_{i+1:02d}')
        else:
            sim_dir = base_sim_dir

        print(f"\n[3/5] {tag} Building simulation"
              f"  w={trace.width:.3f}mm  L={trace.length:.3f}mm")
        print(f"       Output: {sim_dir}")
        try:
            FDTD, ports = build_simulation(trace, stackup, args.f_max, sim_dir)
        except GerberImpedanceError as exc:
            print(f"       ERROR: {exc}")
            continue
        print(f"       Geometry built.")

        print(f"\n[4/5] {tag} Running FDTD"
              f"  (f_max={args.f_max/1e9:.1f} GHz)...")
        if args.post_proc_only:
            print(f"       --post-proc-only: skipping FDTD, re-running post-processing.")
            try:
                results = _postproc_only(ports, sim_dir, f, trace, stackup)
            except Exception as exc:
                print(f"       ERROR: Post-processing failed: {exc}")
                continue
        else:
            try:
                results = run_and_postprocess(
                    FDTD, ports, sim_dir, f, trace, stackup
                )
            except Exception as exc:
                print(f"       ERROR: Simulation failed: {exc}")
                print(f"        Check files in: {sim_dir}")
                continue

        # ── Per-trace results ─────────────────────────────────────────────────
        s11_mid = results.s11[len(f) // 2]
        s21_mid = results.s21[len(f) // 2]
        print(f"\n[5/5] {tag} Results:")
        print(f"       Z0     = {results.Z0_scalar:.1f} ohm")
        print(f"       S11 @ {args.f_max/2e9:.1f} GHz = "
              f"{20*np.log10(abs(s11_mid)):.1f} dB")
        print(f"       S21 @ {args.f_max/2e9:.1f} GHz = "
              f"{20*np.log10(abs(s21_mid)):.1f} dB")

        all_results.append(results)
        plot_results(results)

    # ── Comparison summary (multiple traces) ──────────────────────────────────
    if len(all_results) > 1:
        print(f"\n{'='*60}")
        print(f"  Summary: {len(all_results)} traces simulated")
        print(f"  {'#':<4} {'Width':>8} {'Length':>8} {'Z0':>8}"
              f" {'S11@mid':>9} {'S21@mid':>9}")
        print(f"  {'-'*52}")
        for j, r in enumerate(all_results):
            sm11 = r.s11[len(f) // 2]
            sm21 = r.s21[len(f) // 2]
            print(f"  [{j+1}]"
                  f"  {r.trace.width:>7.3f}mm"
                  f"  {r.trace.length:>6.2f}mm"
                  f"  {r.Z0_scalar:>6.1f}ohm"
                  f"  {20*np.log10(abs(sm11)):>7.1f}dB"
                  f"  {20*np.log10(abs(sm21)):>7.1f}dB")
        print(f"{'='*60}")
        plot_comparison(all_results)

    elif not all_results:
        print("\nERROR: No simulations completed successfully.")
        sys.exit(1)


if __name__ == '__main__':
    main()
