import os
import tempfile
import numpy as np
import matplotlib.pyplot as plt

# Must be set before importing CSXCAD/openEMS so the DLLs can be located
os.environ['OPENEMS_INSTALL_PATH'] = r'C:\Users\paul\Documents\GitHub\emag-dev\openEMS'

from CSXCAD import ContinuousStructure
from openEMS import openEMS

# -----------------------------------------------------------
# Simulation output folder & run control
# -----------------------------------------------------------
Sim_Path = os.path.join(tempfile.gettempdir(), 'patch_msl_sim')
post_proc_only = False  # set True to skip the run and just re-plot

# -----------------------------------------------------------
# 1. Initialize the Simulation Environment
# -----------------------------------------------------------
CSX = ContinuousStructure()
FDTD = openEMS(NrTS=50000, EndCriteria=1e-4)
FDTD.SetCSX(CSX)

f0 = 2.5e9   # centre frequency
fc = 0.5e9   # bandwidth

FDTD.SetGaussExcite(f0, fc)
# Absorbing boundaries on all six faces
FDTD.SetBoundaryCond(['MUR', 'MUR', 'MUR', 'MUR', 'MUR', 'MUR'])

# -----------------------------------------------------------
# 2. Define the Materials
# -----------------------------------------------------------
FR4 = CSX.AddMaterial('FR4', epsilon=4.4)   # dielectric substrate
PEC = CSX.AddMetal('PEC')                   # perfect electric conductor

# -----------------------------------------------------------
# 3. Draw the Geometry
# -----------------------------------------------------------
FR4.AddBox(priority=1, start=[-20, -20, 0],  stop=[20, 20, 1.5])   # substrate
PEC.AddBox(priority=2, start=[-20, -20, 0],  stop=[20, 20, 0])     # ground plane
PEC.AddBox(priority=2, start=[-10, -8, 1.5], stop=[10, 8, 1.5])    # radiating patch

# -----------------------------------------------------------
# 4. Define the Mesh  (MUST come before AddMSLPort)
# -----------------------------------------------------------
feed_width = 2.9   # ~50 Ω feedline on FR4 h=1.5 mm

mesh = CSX.GetGrid()
mesh.SetDeltaUnit(1e-3)
mesh.AddLine('x', [-25, -feed_width/2, 0, feed_width/2, 25])
mesh.AddLine('y', [-25, -20, -17, -14, -11, -8, 0, 8, 25])
mesh.AddLine('z', [-15, 0, 0.5, 1.0, 1.5, 15])

# -----------------------------------------------------------
# 5. Add the RF Port (Microstrip Line)
# -----------------------------------------------------------
port = FDTD.AddMSLPort(
    1,                          # port number
    PEC,                        # metal property for the feedline trace
    [-feed_width/2, -20, 0],    # start  (X spans feedline width)
    [ feed_width/2, -8, 1.5],   # stop
    'y',                        # prop_dir  – wave travels in Y
    'z',                        # exc_dir   – E-field in Z through substrate
    excite=1,
    FeedShift=10,
    MeasPlaneShift=5,
    priority=10
)

# -----------------------------------------------------------
# 6. Run the Simulation
# -----------------------------------------------------------
if not post_proc_only:
    FDTD.Run(Sim_Path, verbose=0, cleanup=True)

# -----------------------------------------------------------
# 7. Post-processing
# -----------------------------------------------------------
f = np.linspace(f0 - fc, f0 + fc, 401)
port.CalcPort(Sim_Path, f, ref_impedance=50)

s11     = port.uf_ref / port.uf_inc
s11_dB  = 20 * np.log10(np.abs(s11))
Zin     = port.uf_tot / port.if_tot

# ── find resonance (deepest dip below -10 dB) ──────────────
res_idx = np.argmin(s11_dB)
f_res   = f[res_idx]
below_10dB = s11_dB[res_idx] < -10

# -----------------------------------------------------------
# 8. Plots
# -----------------------------------------------------------

# ── Figure 1: 3D geometry preview (shown before simulation results) ──────────
def _box_faces(start, stop):
    """Return Poly3DCollection-ready face list for a box (or flat sheet)."""
    x0, y0, z0 = start
    x1, y1, z1 = stop
    if z0 == z1:                        # zero-thickness — single face
        return [[[x0,y0,z0],[x1,y0,z0],[x1,y1,z0],[x0,y1,z0]]]
    return [
        [[x0,y0,z0],[x1,y0,z0],[x1,y1,z0],[x0,y1,z0]],  # bottom
        [[x0,y0,z1],[x1,y0,z1],[x1,y1,z1],[x0,y1,z1]],  # top
        [[x0,y0,z0],[x1,y0,z0],[x1,y0,z1],[x0,y0,z1]],  # front
        [[x0,y1,z0],[x1,y1,z0],[x1,y1,z1],[x0,y1,z1]],  # back
        [[x0,y0,z0],[x0,y1,z0],[x0,y1,z1],[x0,y0,z1]],  # left
        [[x1,y0,z0],[x1,y1,z0],[x1,y1,z1],[x1,y0,z1]],  # right
    ]

from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from matplotlib.patches import Patch as _Patch

fig3d = plt.figure(figsize=(10, 6))
ax3d  = fig3d.add_subplot(111, projection='3d')
fig3d.suptitle('Patch Antenna - 3D Geometry', fontsize=13, fontweight='bold')

fw = feed_width / 2

# Substrate height is only 1.5 mm vs 40 mm board — exaggerate Z for clarity
Z_SCALE = 12   # 1 mm in Z displayed as 12 mm visually

def _scaled_faces(start, stop, zs=Z_SCALE):
    s = [start[0], start[1], start[2] * zs]
    e = [stop[0],  stop[1],  stop[2]  * zs]
    return _box_faces(s, e)

# draw substrate walls only (no top/bottom) so metal layers are visible
sub_walls = [
    [[-20,-20, 0],[20,-20, 0],[20,-20, 1.5*Z_SCALE],[-20,-20, 1.5*Z_SCALE]],  # front
    [[-20, 20, 0],[20, 20, 0],[20, 20, 1.5*Z_SCALE],[-20, 20, 1.5*Z_SCALE]],  # back
    [[-20,-20, 0],[-20,20, 0],[-20,20, 1.5*Z_SCALE],[-20,-20, 1.5*Z_SCALE]],  # left
    [[ 20,-20, 0],[ 20,20, 0],[ 20,20, 1.5*Z_SCALE],[ 20,-20, 1.5*Z_SCALE]],  # right
    [[-20,-20, 0],[ 20,-20, 0],[ 20, 20, 0],[-20, 20, 0]],                     # bottom
    [[-20,-20, 1.5*Z_SCALE],[20,-20,1.5*Z_SCALE],[20,20,1.5*Z_SCALE],[-20,20,1.5*Z_SCALE]], # top
]
ax3d.add_collection3d(Poly3DCollection(
    sub_walls, alpha=0.12, facecolor='#90EE90', edgecolor='#888888', linewidth=0.5))

layers = [
    # (start, stop, facecolor, alpha, label)
    ([-20,-20, 0],  [20, 20,  0],   '#B87333', 0.90, 'Ground Plane (PEC)'),
    ([-fw, -20, 1.5], [fw,  -8, 1.5], '#B87333', 0.90, 'Feedline (PEC)'),
    ([-10,  -8, 1.5], [10,   8, 1.5], '#DAA520', 0.95, 'Patch (PEC)'),
]
for start, stop, color, alpha, label in layers:
    faces = _scaled_faces(start, stop)
    ax3d.add_collection3d(Poly3DCollection(
        faces, alpha=alpha, facecolor=color, edgecolor='#333333', linewidth=0.5))

ax3d.set_xlim(-22, 22)
ax3d.set_ylim(-22, 22)
ax3d.set_zlim(-2, 1.5 * Z_SCALE + 4)
ax3d.set_xlabel('X (mm)')
ax3d.set_ylabel('Y (mm)')
ax3d.set_zlabel('Z (scaled)')
ax3d.set_box_aspect([44, 44, 1.5 * Z_SCALE + 6])
ax3d.view_init(elev=28, azim=-45)

legend_handles = [
    _Patch(facecolor='#90EE90', alpha=0.3, edgecolor='#888888', label='FR4 Substrate'),
] + [
    _Patch(facecolor=c, alpha=a, edgecolor='#333333', label=lbl)
    for _, _, c, a, lbl in layers
]
ax3d.legend(handles=legend_handles, loc='upper left', fontsize=8)
ax3d.set_title('(Z axis exaggerated x{} for visibility)'.format(Z_SCALE),
               fontsize=8, color='grey')
plt.tight_layout()
plt.savefig(os.path.join(Sim_Path, 'geometry.png'), dpi=150)
print(f'Geometry plot saved to {Sim_Path}/geometry.png')

# ── Figure 2: simulation results ─────────────────────────────────────────────
fig, axes = plt.subplots(3, 1, figsize=(9, 11))
fig.suptitle('Patch Antenna - Simulation Results', fontsize=13, fontweight='bold')

# ── Plot 1: S11 return loss ─────────────────────────────────
ax = axes[0]
ax.plot(f / 1e9, s11_dB, linewidth=2, label='$S_{11}$')
ax.axhline(-10, color='r', linestyle='--', linewidth=1, label='−10 dB')
if below_10dB:
    ax.axvline(f_res / 1e9, color='g', linestyle=':', linewidth=1,
               label=f'Resonance {f_res/1e9:.3f} GHz')
ax.set_xlabel('Frequency (GHz)')
ax.set_ylabel('S11 (dB)')
ax.set_title('Return Loss')
ax.legend()
ax.grid(True)

# ── Plot 2: Input impedance ─────────────────────────────────
ax = axes[1]
ax.plot(f / 1e9, np.real(Zin), linewidth=2, label=r'$\Re\{Z_{in}\}$')
ax.plot(f / 1e9, np.imag(Zin), linewidth=2, linestyle='--',
        label=r'$\Im\{Z_{in}\}$')
ax.axhline(50, color='grey', linestyle=':', linewidth=1, label='50 Ω')
ax.set_xlabel('Frequency (GHz)')
ax.set_ylabel('Impedance (Ω)')
ax.set_title('Input Impedance')
ax.legend()
ax.grid(True)

# ── Plot 3: Smith chart (normalised to 50 Ω) ───────────────
ax = axes[2]
z0  = 50
zn  = Zin / z0                     # normalised impedance
gam = (zn - 1) / (zn + 1)          # reflection coefficient

# draw Smith chart background circles
theta = np.linspace(0, 2 * np.pi, 360)
# unit circle
ax.plot(np.cos(theta), np.sin(theta), 'k-', linewidth=0.8)
ax.axhline(0, color='k', linewidth=0.5)
# constant-resistance circles  r = 0, 0.5, 1, 2
for r in (0, 0.5, 1, 2):
    cx, rad = r / (1 + r), 1 / (1 + r)
    ax.plot(cx + rad * np.cos(theta), rad * np.sin(theta),
            color='lightgrey', linewidth=0.6)
# constant-reactance arcs  x = ±0.5, ±1, ±2
for x in (0.5, 1, 2):
    for sgn in (1, -1):
        cx, cy, rad = 1, sgn / x, 1 / x
        arc = theta[(np.cos(theta) * rad + cx) ** 2 +
                    (np.sin(theta) * rad + cy) ** 2 <= 1.01]
        ax.plot(cx + rad * np.cos(arc), cy + rad * np.sin(arc),
                color='lightgrey', linewidth=0.6)

# plot the trace, colour-coded by frequency
sc = ax.scatter(np.real(gam), np.imag(gam),
                c=f / 1e9, cmap='viridis', s=10, zorder=3)
plt.colorbar(sc, ax=ax, label='Frequency (GHz)')
if below_10dB:
    ax.plot(np.real(gam[res_idx]), np.imag(gam[res_idx]),
            'r*', markersize=12, label=f'{f_res/1e9:.3f} GHz', zorder=4)
    ax.legend(loc='lower right')
ax.set_xlim(-1.1, 1.1)
ax.set_ylim(-1.1, 1.1)
ax.set_aspect('equal')
ax.set_xlabel('Re{Γ}')
ax.set_ylabel('Im{Γ}')
ax.set_title('Smith Chart (Z₀ = 50 Ω)')
ax.grid(False)

plt.tight_layout()
plt.savefig(os.path.join(Sim_Path, 'results.png'), dpi=150)
print(f'Plots saved to {Sim_Path}/results.png')
plt.show()
