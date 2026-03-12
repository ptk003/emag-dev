# -*- coding: utf-8 -*-
"""
openems_sim.py — openEMS FDTD microstrip impedance simulation

Provides:
  build_simulation(trace, stackup, f_max, sim_dir) -> (FDTD, ports)
  run_and_postprocess(FDTD, ports, sim_dir, f, trace, stackup) -> SimResults
"""

from __future__ import annotations
import os
import math
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from gerber_parser import TraceSegment, StackupLayer


# ─────────────────────────────────────────────────────────────────────────────
# Ensure openEMS DLLs are found before any CSXCAD import
# ─────────────────────────────────────────────────────────────────────────────
_OPENEMS_PATH = Path(__file__).resolve().parent.parent.parent / 'openEMS'
if 'OPENEMS_INSTALL_PATH' not in os.environ:
    os.environ['OPENEMS_INSTALL_PATH'] = str(_OPENEMS_PATH)
os.add_dll_directory(os.environ['OPENEMS_INSTALL_PATH'])


# ─────────────────────────────────────────────────────────────────────────────
# Result dataclass
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class SimResults:
    freq:       np.ndarray   # Hz,  shape (N,)
    s11:        np.ndarray   # complex, shape (N,)
    s21:        np.ndarray   # complex, shape (N,)
    Zin:        np.ndarray   # complex Ω, shape (N,)
    Z0_scalar:  float        # median characteristic impedance Ω
    Z0_freq:    np.ndarray   # Z0(f), shape (N,)
    tdr_time:   np.ndarray   # s, shape (M,)
    tdr_rho:    np.ndarray   # dimensionless reflection vs time, shape (M,)
    trace:      TraceSegment
    stackup:    list
    sim_dir:    str


class GerberImpedanceError(RuntimeError):
    pass


# ─────────────────────────────────────────────────────────────────────────────
# Stackup helpers
# ─────────────────────────────────────────────────────────────────────────────

def _find_microstrip_layers(stackup: list[StackupLayer]):
    """
    Identify signal layer, substrate dielectric(s), and ground layer for a
    simple microstrip: signal copper / dielectric(s) / ground copper.

    Returns (signal_layer, substrate_layers, ground_layer).
    Assumes the first conductor from the top is the signal layer and the
    next conductor below is the ground plane.
    """
    # Stackup is ordered bottom-to-top (z_bottom of first layer = 0)
    conductors = [l for l in stackup if l.is_conductor]
    dielectrics = [l for l in stackup if l.is_dielectric]

    if len(conductors) < 2:
        raise GerberImpedanceError(
            "Stackup must have at least 2 conductor layers "
            "(signal + ground). Check your stackup XML."
        )
    if not dielectrics:
        raise GerberImpedanceError(
            "Stackup has no dielectric layers. Check your stackup XML."
        )

    # Top conductor = signal (highest z_top)
    signal_layer = max(conductors, key=lambda l: l.z_top_mm)
    # Bottom conductor = ground (lowest z_bottom)
    ground_layer = min(conductors, key=lambda l: l.z_bottom_mm)

    # Substrate = all dielectric layers between ground and signal
    substrate_layers = [
        l for l in dielectrics
        if l.z_bottom_mm >= ground_layer.z_top_mm
        and l.z_top_mm   <= signal_layer.z_bottom_mm
    ]
    if not substrate_layers:
        # Fall back: any dielectric
        substrate_layers = dielectrics

    return signal_layer, substrate_layers, ground_layer


# ─────────────────────────────────────────────────────────────────────────────
# Simulation builder
# ─────────────────────────────────────────────────────────────────────────────

def build_simulation(trace: TraceSegment,
                     stackup: list[StackupLayer],
                     f_max: float,
                     sim_dir: str):
    """
    Build an openEMS FDTD microstrip simulation for the given trace.

    Returns (FDTD, [port0, port1]).
    """
    from CSXCAD import ContinuousStructure
    from openEMS import openEMS
    from openEMS.physical_constants import C0

    # ── Validate ──────────────────────────────────────────────────────────────
    if trace.is_diagonal:
        raise GerberImpedanceError(
            f"Diagonal trace ({trace.angle_deg:.1f}°) cannot be simulated directly. "
            "Select a horizontal or vertical segment."
        )
    if trace.length < 0.1:
        raise GerberImpedanceError(
            f"Trace length {trace.length:.3f} mm is too short to simulate."
        )

    # ── Stackup geometry ──────────────────────────────────────────────────────
    signal_layer, substrate_layers, ground_layer = _find_microstrip_layers(stackup)

    substrate_er   = substrate_layers[0].permittivity
    substrate_tand = substrate_layers[0].loss_tangent
    substrate_thickness = sum(l.thickness_mm for l in substrate_layers)
    trace_z   = signal_layer.z_bottom_mm   # bottom of signal copper = trace surface
    ground_z  = ground_layer.z_top_mm      # top of ground plane

    # ── Simulation parameters ─────────────────────────────────────────────────
    unit = 1e-3   # all dimensions in mm; openEMS uses metres internally

    # Minimum resolution = lambda/50 in the substrate
    resolution = C0 / (f_max * math.sqrt(substrate_er)) / unit / 50

    # Minimum trace length: 3× wavelength in substrate at f_max
    lambda_sub_mm = C0 / f_max / math.sqrt(substrate_er) / unit
    min_sim_len = 3.0 * lambda_sub_mm
    if trace.length < min_sim_len:
        print(f"  WARNING: trace length {trace.length:.2f} mm < {min_sim_len:.2f} mm "
              f"(3x lambda at {f_max/1e9:.1f} GHz in substrate). "
              "Feed lines will be extended.")
    sim_length = max(trace.length, min_sim_len)

    w = trace.width

    # Propagation and excitation directions
    if trace.is_horizontal:
        prop_dir, exc_dir = 'x', 'z'
    else:  # vertical
        prop_dir, exc_dir = 'y', 'z'

    # Air above trace: 3× substrate thickness
    air_height = 3.0 * substrate_thickness
    sim_top_z  = trace_z + air_height

    # ── FDTD setup ────────────────────────────────────────────────────────────
    FDTD = openEMS()
    FDTD.SetGaussExcite(f_max / 2, f_max / 2)
    # PML on propagation axis ends; MUR on sides; PEC bottom (ground); MUR top
    if prop_dir == 'x':
        FDTD.SetBoundaryCond(['PML_8', 'PML_8', 'MUR', 'MUR', 'PEC', 'MUR'])
    else:
        FDTD.SetBoundaryCond(['MUR', 'MUR', 'PML_8', 'PML_8', 'PEC', 'MUR'])

    CSX = ContinuousStructure()
    FDTD.SetCSX(CSX)
    mesh = CSX.GetGrid()
    mesh.SetDeltaUnit(unit)

    # ── Mesh ──────────────────────────────────────────────────────────────────
    # "Third-mesh" technique: place lines fractionally inside each trace edge
    # to force FDTD cells to align with the metal boundary.
    # Use /2 (not /4) so edge_res = resolution/2, keeping the CFL timestep
    # within a factor of 2 of the bulk resolution and avoiding the "timestep
    # very small" warning that appears with /4.
    third = np.array([2 * resolution / 3, -resolution / 3]) / 2
    edge_res = resolution / 2   # fine near trace edges (was /4 — caused tiny dt)

    if prop_dir == 'x':
        # Propagation: X; transverse: Y
        mesh.AddLine('x', [-sim_length / 2, sim_length / 2])
        mesh.SmoothMeshLines('x', resolution)

        mesh.AddLine('y', 0)
        mesh.AddLine('y',  w / 2 + third)
        mesh.AddLine('y', -w / 2 - third)
        mesh.SmoothMeshLines('y', edge_res)
        mesh.AddLine('y', [-15 * w, 15 * w])
        mesh.SmoothMeshLines('y', resolution)
    else:
        # Propagation: Y; transverse: X
        mesh.AddLine('y', [-sim_length / 2, sim_length / 2])
        mesh.SmoothMeshLines('y', resolution)

        mesh.AddLine('x', 0)
        mesh.AddLine('x',  w / 2 + third)
        mesh.AddLine('x', -w / 2 - third)
        mesh.SmoothMeshLines('x', edge_res)
        mesh.AddLine('x', [-15 * w, 15 * w])
        mesh.SmoothMeshLines('x', resolution)

    # Z mesh: resolve substrate with ≥5 cells, add air layer above
    z_cells = max(5, int(substrate_thickness / edge_res))
    mesh.AddLine('z', np.linspace(ground_z, trace_z, z_cells + 1))
    mesh.AddLine('z', sim_top_z)
    mesh.SmoothMeshLines('z', resolution)

    # ── Materials ─────────────────────────────────────────────────────────────
    pec = CSX.AddMetal('PEC')

    # Board extents depend on propagation direction:
    #   prop_dir='x' → board_half is along x, transverse is ±15*w along y
    #   prop_dir='y' → board_half is along y, transverse is ±15*w along x
    board_half = sim_length / 2 + 10 * w
    if prop_dir == 'x':
        bx0, bx1 = -board_half, board_half
        by0, by1 = -15 * w,     15 * w
    else:
        bx0, bx1 = -15 * w,     15 * w
        by0, by1 = -board_half, board_half

    # Substrate dielectric layer(s)
    f0_loss = f_max / 2
    for lyr in substrate_layers:
        kappa = lyr.loss_tangent * 2 * math.pi * f0_loss * 8.854e-12 * lyr.permittivity
        sub = CSX.AddMaterial(f'sub_{lyr.name.replace(" ", "_")}',
                               epsilon=lyr.permittivity,
                               kappa=kappa)
        sub.AddBox(
            start=[bx0, by0, lyr.z_bottom_mm],
            stop= [bx1, by1, lyr.z_top_mm],
            priority=0
        )

    # Ground plane
    gnd = CSX.AddMetal('gnd')
    gnd.AddBox(
        start=[bx0, by0, ground_z],
        stop= [bx1, by1, ground_z],
        priority=10
    )
    FDTD.AddEdges2Grid(dirs='xy', properties=gnd)

    # ── MSL Ports ─────────────────────────────────────────────────────────────
    # Both ports span the FULL sim_length — exactly as in MSL_NotchFilter.py.
    # z-convention: start.z = ground_z (low), stop.z = trace_z (high).
    # excite=-1 → excitation at the min-prop_dir face (left/bottom).
    # excite=0  → port 1 is purely a measurement port at the max face.
    # The port creates the trace metal via `pec`; having both ports reference
    # the same full box just means the metal is defined twice (harmless).
    L    = sim_length
    port = [None, None]

    if prop_dir == 'x':
        ps = [-L / 2, -w / 2, ground_z]
        pe = [ L / 2,  w / 2, trace_z]
    else:
        ps = [-w / 2, -L / 2, ground_z]
        pe = [ w / 2,  L / 2, trace_z]

    port[0] = FDTD.AddMSLPort(
        1, pec, ps, pe, prop_dir, exc_dir,
        excite=-1,
        FeedShift=10 * resolution,
        MeasPlaneShift=L / 3,
        priority=10
    )
    port[1] = FDTD.AddMSLPort(
        2, pec, ps, pe, prop_dir, exc_dir,
        MeasPlaneShift=L / 3,
        priority=10
    )

    return FDTD, port


# ─────────────────────────────────────────────────────────────────────────────
# Post-processing
# ─────────────────────────────────────────────────────────────────────────────

def _effective_er(er: float, h_mm: float, w_mm: float) -> float:
    """Microstrip effective dielectric constant (Hammerstad/Wheeler)."""
    ratio = h_mm / w_mm
    return (er + 1) / 2 + (er - 1) / 2 / math.sqrt(1 + 12 * ratio)


def _compute_tdr(s11: np.ndarray, freq: np.ndarray,
                 er_eff: float, trace_length_mm: float) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute TDR (time-domain reflectometry) step response from S11.

    Returns (tdr_time_s, tdr_rho_normalised).
    """
    N = len(freq)
    N_pad = int(2 ** math.ceil(math.log2(N * 16)))   # 16× zero-pad

    f_full = np.linspace(0, freq[-1], N_pad // 2 + 1)
    s11_r = np.interp(f_full, freq, s11.real)
    s11_i = np.interp(f_full, freq, s11.imag)
    s11_pad = s11_r + 1j * s11_i

    # Conjugate-symmetric for real IFFT → full spectrum
    s11_sym = np.concatenate([s11_pad, np.conj(s11_pad[-2:0:-1])])
    window   = np.hanning(len(s11_sym))
    h_t      = np.fft.ifft(s11_sym * window).real

    dt = 1.0 / (2 * freq[-1])
    tdr_time = np.arange(len(h_t)) * dt

    # Step response via cumulative sum
    tdr_rho = np.cumsum(h_t)
    peak = np.max(np.abs(tdr_rho))
    if peak > 0:
        tdr_rho /= peak

    # Trim to 2× electrical length of trace
    v_phase = 3e8 / math.sqrt(er_eff)
    t_max = 2.0 * trace_length_mm * 1e-3 / v_phase
    mask = tdr_time <= t_max * 1.5   # small margin
    if mask.sum() < 10:
        mask = np.ones(len(tdr_time), dtype=bool)

    return tdr_time[mask], tdr_rho[mask]


def run_and_postprocess(FDTD, ports,
                        sim_dir: str,
                        f: np.ndarray,
                        trace: TraceSegment,
                        stackup: list) -> SimResults:
    """
    Run the FDTD simulation and extract S-parameters, Z0, Zin, and TDR.
    """
    sim_path = Path(sim_dir)
    if sim_path.exists():
        shutil.rmtree(sim_path)
    sim_path.mkdir(parents=True)

    print(f"  Running FDTD in: {sim_dir}")
    FDTD.Run(str(sim_dir), cleanup=False, verbose=1)

    # ── Port post-processing ──────────────────────────────────────────────────
    print("  Post-processing ports...")
    for p in ports:
        p.CalcPort(str(sim_dir), f, ref_impedance=50)

    s11 = ports[0].uf_ref / ports[0].uf_inc
    s21 = ports[1].uf_ref / ports[0].uf_inc
    Zin = ports[0].uf_tot / ports[0].if_tot

    # ── Characteristic impedance ──────────────────────────────────────────────
    # MSLPort does not expose a .ZL attribute; compute Z0 from the ratio of
    # incident voltage to incident current (valid for TEM / quasi-TEM modes).
    Z0_freq = np.abs(ports[0].uf_inc / ports[0].if_inc)

    N = len(f)
    mid = slice(N // 4, 3 * N // 4)
    Z0_scalar = float(np.median(np.real(Z0_freq[mid])))

    # ── TDR ───────────────────────────────────────────────────────────────────
    _, substrate_layers, _ = _find_microstrip_layers(stackup)
    er     = substrate_layers[0].permittivity
    h_mm   = sum(l.thickness_mm for l in substrate_layers)
    er_eff = _effective_er(er, h_mm, trace.width)

    tdr_time, tdr_rho = _compute_tdr(s11, f, er_eff, trace.length)

    return SimResults(
        freq=f, s11=s11, s21=s21, Zin=Zin,
        Z0_scalar=Z0_scalar, Z0_freq=Z0_freq,
        tdr_time=tdr_time, tdr_rho=tdr_rho,
        trace=trace, stackup=stackup,
        sim_dir=str(sim_dir),
    )


# ─────────────────────────────────────────────────────────────────────────────
# Quick self-test: simulate a known 50 Ω microstrip
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    import sys
    sys.path.insert(0, str(Path(__file__).parent))
    from gerber_parser import TraceSegment, default_stackup

    print("Self-test: 50 ohm microstrip (w=2.9 mm, h=1.6 mm, er=4.4)")
    print("Analytical Z0 ~ 50 ohm  (Wheeler formula)\n")

    trace = TraceSegment(
        x1=-15.0, y1=0.0, x2=15.0, y2=0.0,
        width=2.9, layer='test', aperture_shape='round'
    )
    stackup = default_stackup()
    # 3 GHz: lambda/50 mesh = 0.95 mm → dt ~1 ps → sim done in ~2 min.
    # Use 10 GHz only when you need frequency data above 3 GHz.
    f_max = 3e9
    sim_dir = tempfile.mkdtemp(prefix='GI_test_')

    print(f"Building simulation in: {sim_dir}")
    FDTD, ports = build_simulation(trace, stackup, f_max, sim_dir)

    f = np.linspace(1e6, f_max, 801)
    results = run_and_postprocess(FDTD, ports, sim_dir, f, trace, stackup)

    print(f"\nResult: Z0 = {results.Z0_scalar:.1f} ohm")
    print(f"        S11 at mid-band: {20*np.log10(np.abs(results.s11[len(f)//2])):.1f} dB")
    print(f"        S21 at mid-band: {20*np.log10(np.abs(results.s21[len(f)//2])):.1f} dB")
