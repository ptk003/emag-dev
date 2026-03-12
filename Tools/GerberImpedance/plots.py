# -*- coding: utf-8 -*-
"""
plots.py — Result visualisation for GerberImpedance

Provides:
  plot_results(results: SimResults) -> None
"""

from __future__ import annotations
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


def plot_results(results) -> None:
    """
    Produce a 4-panel figure:
      Top-left:     Z0(f) — characteristic impedance vs frequency
      Top-right:    S11 & S21 in dB
      Bottom-left:  Re{Zin} and Im{Zin} vs frequency
      Bottom-right: TDR step response (reflection vs time in ps)

    Saves a PNG to sim_dir/gerber_impedance_results.png and calls plt.show().
    """
    tr  = results.trace
    f   = results.freq
    f_ghz = f / 1e9

    fig = plt.figure(figsize=(15, 10))
    fig.suptitle(
        f"GerberImpedance — Layer: {tr.layer} | "
        f"Width: {tr.width:.3f} mm | Length: {tr.length:.3f} mm | "
        f"Z\u2080 = {results.Z0_scalar:.1f} \u03a9",
        fontsize=12, fontweight='bold'
    )

    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.38, wspace=0.32)

    # ── Panel 1: Z0(f) ────────────────────────────────────────────────────────
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(f_ghz, np.real(results.Z0_freq), 'b-', linewidth=2,
             label='Z\u2080(f)')
    ax1.axhline(results.Z0_scalar, color='r', linestyle='--', linewidth=1.5,
                label=f'Z\u2080 = {results.Z0_scalar:.1f} \u03a9')
    ax1.set_xlabel('Frequency (GHz)')
    ax1.set_ylabel('Characteristic Impedance (\u03a9)')
    ax1.set_title('Characteristic Impedance Z\u2080(f)')
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.4)
    # Reasonable y-limits
    z0_med = results.Z0_scalar
    ax1.set_ylim([max(0, z0_med - 30), z0_med + 30])

    # ── Panel 2: S-parameters ─────────────────────────────────────────────────
    ax2 = fig.add_subplot(gs[0, 1])
    s11_db = 20 * np.log10(np.abs(results.s11) + 1e-12)
    s21_db = 20 * np.log10(np.abs(results.s21) + 1e-12)
    ax2.plot(f_ghz, s11_db, 'k-',  linewidth=2, label='S\u2081\u2081')
    ax2.plot(f_ghz, s21_db, 'r--', linewidth=2, label='S\u2082\u2081')
    ax2.set_xlabel('Frequency (GHz)')
    ax2.set_ylabel('S-Parameter (dB)')
    ax2.set_title('S-Parameters vs Frequency')
    ax2.set_ylim([-40, 2])
    ax2.axhline(-3, color='grey', linestyle=':', linewidth=1, alpha=0.6)
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.4)

    # ── Panel 3: Input impedance Zin(f) ──────────────────────────────────────
    ax3 = fig.add_subplot(gs[1, 0])
    ax3.plot(f_ghz, np.real(results.Zin), 'b-',  linewidth=2, label='Re{Z\u1d62\u2099}')
    ax3.plot(f_ghz, np.imag(results.Zin), 'r--', linewidth=2, label='Im{Z\u1d62\u2099}')
    ax3.axhline(50,  color='grey', linestyle=':', linewidth=1, alpha=0.7, label='50 \u03a9')
    ax3.axhline(0,   color='k',    linestyle='-', linewidth=0.5)
    ax3.set_xlabel('Frequency (GHz)')
    ax3.set_ylabel('Impedance (\u03a9)')
    ax3.set_title('Input Impedance Z\u1d62\u2099(f)')
    ax3.legend(fontsize=9)
    ax3.grid(True, alpha=0.4)

    # ── Panel 4: TDR ──────────────────────────────────────────────────────────
    ax4 = fig.add_subplot(gs[1, 1])
    tdr_ps = results.tdr_time * 1e12   # seconds → picoseconds
    ax4.plot(tdr_ps, results.tdr_rho, 'g-', linewidth=2)
    ax4.axhline(0, color='k', linewidth=0.5)
    ax4.set_xlabel('Time (ps)')
    ax4.set_ylabel('Reflection Coefficient \u03c1(t)')
    ax4.set_title('TDR — Time-Domain Reflectometry')
    ax4.grid(True, alpha=0.4)

    # Annotate Z0 on TDR panel using the standard rho → Z0 conversion
    # For a uniform line: rho = (Z0 - Z_ref) / (Z0 + Z_ref)
    Z_ref = 50.0
    rho_z0 = (results.Z0_scalar - Z_ref) / (results.Z0_scalar + Z_ref)
    if len(tdr_ps) > 5:
        mid_t = tdr_ps[len(tdr_ps) // 2]
        ax4.annotate(
            f'  \u03c1 \u2192 Z\u2080\u2248{results.Z0_scalar:.0f}\u03a9',
            xy=(mid_t, rho_z0), fontsize=9, color='darkgreen',
            arrowprops=dict(arrowstyle='->', color='darkgreen'),
            xytext=(mid_t * 0.6, rho_z0 + 0.15)
        )

    # ── Stackup summary (text box) ────────────────────────────────────────────
    stack_lines = []
    for lyr in reversed(results.stackup):   # top-to-bottom display
        if lyr.is_conductor:
            stack_lines.append(f"  Cu  {lyr.thickness_mm*1000:.0f} µm  {lyr.name}")
        else:
            stack_lines.append(
                f"  εr={lyr.permittivity}  h={lyr.thickness_mm:.3f}mm  {lyr.name}"
            )
    stack_txt = "Stackup (top→bot):\n" + "\n".join(stack_lines)
    fig.text(0.01, 0.01, stack_txt, fontsize=7, family='monospace',
             verticalalignment='bottom',
             bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

    # ── Save and show ──────────────────────────────────────────────────────────
    out_png = Path(results.sim_dir) / 'gerber_impedance_results.png'
    plt.savefig(str(out_png), dpi=150, bbox_inches='tight')
    print(f"  Results saved: {out_png}")
    plt.show()


# ---------------------------------------------------------------------------
# Colour palette — same as _HIGHLIGHT_COLORS in gerber_impedance.py so the
# bar chart colours match the viewer highlights the user saw when selecting.
# ---------------------------------------------------------------------------
_TRACE_COLORS = [
    '#00bfff',   # cyan-ish
    '#ffff44',   # yellow
    '#ff44ff',   # magenta
    '#44ff44',   # green
    '#ff8844',   # orange
    '#44ffff',   # teal
    '#ff4444',   # red
    '#aaaaff',   # lavender
]


def plot_comparison(results_list) -> None:
    """
    3-panel comparison figure for multiple SimResults:

      Left:   Z0(f) overlay — one coloured line per trace, dashed scalar Z0
      Centre: S11 in dB overlay
      Right:  Bar chart of scalar Z0 per trace with value annotations

    PNG saved alongside the first sim_dir's parent as
    'gerber_impedance_comparison.png'.
    """
    if not results_list:
        return

    n = len(results_list)
    colours = [_TRACE_COLORS[i % len(_TRACE_COLORS)] for i in range(n)]

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle(
        f"GerberImpedance — Comparison of {n} Selected Trace{'s' if n != 1 else ''}",
        fontsize=12, fontweight='bold'
    )

    ax_z0, ax_s11, ax_bar = axes

    # ── Left panel: Z0(f) overlay ─────────────────────────────────────────────
    ax_z0.set_title('Characteristic Impedance Z\u2080(f)')
    ax_z0.set_xlabel('Frequency (GHz)')
    ax_z0.set_ylabel('Z\u2080 (\u03a9)')
    ax_z0.grid(True, alpha=0.4)

    z0_all = [r.Z0_scalar for r in results_list]
    y_centre = float(np.median(z0_all))

    for i, (r, col) in enumerate(zip(results_list, colours)):
        f_ghz = r.freq / 1e9
        label = (f"#{i+1} {r.trace.layer} "
                 f"w={r.trace.width:.2f}mm "
                 f"Z\u2080={r.Z0_scalar:.1f}\u03a9")
        ax_z0.plot(f_ghz, np.real(r.Z0_freq), color=col, linewidth=2, label=label)
        ax_z0.axhline(r.Z0_scalar, color=col, linestyle='--', linewidth=1.0, alpha=0.7)

    ax_z0.set_ylim([max(0, y_centre - 40), y_centre + 40])
    ax_z0.legend(fontsize=7, loc='best')

    # ── Centre panel: S11(f) overlay ─────────────────────────────────────────
    ax_s11.set_title('Return Loss S\u2081\u2081(f)')
    ax_s11.set_xlabel('Frequency (GHz)')
    ax_s11.set_ylabel('S\u2081\u2081 (dB)')
    ax_s11.grid(True, alpha=0.4)
    ax_s11.set_ylim([-50, 2])
    ax_s11.axhline(-10, color='grey', linestyle=':', linewidth=1, alpha=0.6,
                   label='\u221210 dB ref')
    ax_s11.axhline(-20, color='grey', linestyle=':', linewidth=1, alpha=0.4)

    for i, (r, col) in enumerate(zip(results_list, colours)):
        f_ghz = r.freq / 1e9
        s11_db = 20 * np.log10(np.abs(r.s11) + 1e-12)
        ax_s11.plot(f_ghz, s11_db, color=col, linewidth=2,
                    label=f"#{i+1} {r.trace.layer}")

    ax_s11.legend(fontsize=7, loc='best')

    # ── Right panel: scalar Z0 bar chart ─────────────────────────────────────
    ax_bar.set_title('Scalar Z\u2080 per Trace')
    ax_bar.set_xlabel('Trace #')
    ax_bar.set_ylabel('Z\u2080 (\u03a9)')
    ax_bar.grid(True, alpha=0.3, axis='y')

    x_pos = np.arange(n)
    bars = ax_bar.bar(x_pos, z0_all, color=colours, edgecolor='black',
                      linewidth=0.8, width=0.6)

    # Annotate each bar with its value
    for bar, z0_val in zip(bars, z0_all):
        ax_bar.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.5,
            f'{z0_val:.1f} \u03a9',
            ha='center', va='bottom', fontsize=9, fontweight='bold'
        )

    # Add 50-ohm reference line
    ax_bar.axhline(50, color='red', linestyle='--', linewidth=1.5,
                   label='50 \u03a9 ref', alpha=0.8)
    ax_bar.set_xticks(x_pos)
    ax_bar.set_xticklabels([f'#{i+1}' for i in range(n)])
    ax_bar.legend(fontsize=8)

    # Y range: centre on 50 ohm or the actual values, whichever spans more
    bar_min = min(min(z0_all) - 10, 30)
    bar_max = max(max(z0_all) + 15, 80)
    ax_bar.set_ylim([bar_min, bar_max])

    # ── Sub-title row: trace details ─────────────────────────────────────────
    detail_lines = []
    for i, r in enumerate(results_list):
        tr = r.trace
        detail_lines.append(
            f"  #{i+1}: layer={tr.layer}  w={tr.width:.3f}mm  "
            f"L={tr.length:.2f}mm  Z\u2080={r.Z0_scalar:.1f}\u03a9  "
            f"sim={r.sim_dir}"
        )
    fig.text(0.01, 0.01, "\n".join(detail_lines),
             fontsize=6.5, family='monospace', verticalalignment='bottom',
             bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.7))

    plt.tight_layout(rect=[0, 0.12, 1, 1])

    # ── Save PNG next to the first sim_dir ───────────────────────────────────
    first_sim_parent = Path(results_list[0].sim_dir).parent
    out_png = first_sim_parent / 'gerber_impedance_comparison.png'
    plt.savefig(str(out_png), dpi=150, bbox_inches='tight')
    print(f"  Comparison saved: {out_png}")
    plt.show()
