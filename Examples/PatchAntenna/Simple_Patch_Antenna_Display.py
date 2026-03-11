# -*- coding: utf-8 -*-
"""
Simple Patch Antenna - E-Field Time Animation for ParaView

Runs the patch antenna FDTD simulation with time-domain E-field recording,
writes VTK time-series files and a pvpython visualisation script, then
optionally launches ParaView.

Outputs (all in Examples/PatchAntenna/Visualizations/):
  sim_data/Et_3D/   - 3D E-field VTK time-series  (.vtr per timestep)
  sim_data/Et_xy/   - 2D XY slice at z = substrate_thickness
  sim_data/Et_xz/   - 2D XZ slice at y = 0
  Et_3D.pvd         - ParaView time-series collection for 3D data
  Et_xy.pvd         - collection for XY slice
  Et_xz.pvd         - collection for XZ slice
  visualize.py      - pvpython script; run to produce state file + screenshots
"""

import os
import re
import subprocess
import numpy as np
from pathlib import Path

# ── Must be set before importing CSXCAD so the DLLs can be found ─────────────
os.environ['OPENEMS_INSTALL_PATH'] = r'C:\Users\paul\Documents\GitHub\emag-dev\openEMS'

from CSXCAD import ContinuousStructure
from openEMS import openEMS
from openEMS.physical_constants import *

# ── Paths ─────────────────────────────────────────────────────────────────────
SCRIPT_DIR = Path(__file__).resolve().parent
VIZ_DIR    = SCRIPT_DIR / 'Visualizations'
SIM_DATA   = VIZ_DIR / 'sim_data'
PARAVIEW   = Path(r'C:\Program Files\ParaView 6.1.0\bin\paraview.exe')
PVPYTHON   = Path(r'C:\Program Files\ParaView 6.1.0\bin\pvpython.exe')

VIZ_DIR.mkdir(exist_ok=True)
SIM_DATA.mkdir(exist_ok=True)

# ── Run control ───────────────────────────────────────────────────────────────
post_proc_only  = False  # True = skip simulation, regenerate ParaView files only
launch_paraview = True   # True = open ParaView after generating files

# ── Geometry parameters (matching Simple_Patch_Antenna.py) ───────────────────
patch_width        = 32       # mm, resonant dimension in x
patch_length       = 40       # mm, resonant dimension in y
substrate_epsR     = 3.38
substrate_kappa    = 1e-3 * 2*np.pi*2.45e9 * EPS0 * substrate_epsR
substrate_width    = 60       # mm
substrate_length   = 60       # mm
substrate_thickness= 1.524    # mm
substrate_cells    = 4
feed_pos           = -6       # mm, feed offset in x
feed_R             = 50       # Ω
SimBox             = np.array([200, 200, 150])   # mm
f0 = 2e9    # Hz - centre frequency
fc = 1e9    # Hz - bandwidth

# ── FDTD setup ────────────────────────────────────────────────────────────────
# 5000 timesteps gives ~130 dump frames (one every ~38 steps) — enough to see
# the full pulse excite, propagate across the patch, and ring down.
NrTS_anim = 5000

FDTD = openEMS(NrTS=NrTS_anim, EndCriteria=1e-6)
FDTD.SetGaussExcite(f0, fc)
FDTD.SetBoundaryCond(['MUR', 'MUR', 'MUR', 'MUR', 'MUR', 'MUR'])

CSX = ContinuousStructure()
FDTD.SetCSX(CSX)
mesh = CSX.GetGrid()
mesh.SetDeltaUnit(1e-3)
mesh_res = C0 / (f0 + fc) / 1e-3 / 20

# ── Mesh ──────────────────────────────────────────────────────────────────────
mesh.AddLine('x', [-SimBox[0]/2,  SimBox[0]/2])
mesh.AddLine('y', [-SimBox[1]/2,  SimBox[1]/2])
mesh.AddLine('z', [-SimBox[2]/3,  SimBox[2]*2/3])

# ── Patch ─────────────────────────────────────────────────────────────────────
patch = CSX.AddMetal('patch')
start = [-patch_width/2,  -patch_length/2,  substrate_thickness]
stop  = [ patch_width/2,   patch_length/2,  substrate_thickness]
patch.AddBox(priority=10, start=start, stop=stop)
FDTD.AddEdges2Grid(dirs='xy', properties=patch, metal_edge_res=mesh_res/2)

# ── Substrate ─────────────────────────────────────────────────────────────────
substrate = CSX.AddMaterial('substrate', epsilon=substrate_epsR, kappa=substrate_kappa)
sub_start = [-substrate_width/2,  -substrate_length/2,  0]
sub_stop  = [ substrate_width/2,   substrate_length/2,  substrate_thickness]
substrate.AddBox(priority=0, start=sub_start, stop=sub_stop)
mesh.AddLine('z', np.linspace(0, substrate_thickness, substrate_cells + 1))

# ── Ground plane ──────────────────────────────────────────────────────────────
gnd = CSX.AddMetal('gnd')
gnd.AddBox(start=[-substrate_width/2, -substrate_length/2, 0],
           stop= [ substrate_width/2,  substrate_length/2, 0],
           priority=10)
FDTD.AddEdges2Grid(dirs='xy', properties=gnd)

# ── Lumped port ───────────────────────────────────────────────────────────────
port = FDTD.AddLumpedPort(
    1, feed_R,
    [feed_pos, 0, 0], [feed_pos, 0, substrate_thickness],
    'z', 1.0, priority=5, edges2grid='xy'
)

mesh.SmoothMeshLines('all', mesh_res, 1.4)

# ── E-field time-domain dumps ─────────────────────────────────────────────────
# dump_type=0  → electric field (E)
# dump_mode=0  → time domain (one .vtr per timestep)
# file_type=0  → VTK rectilinear grid (.vtr)
#
# 3D dump: volume around the antenna. No spatial subsampling so the field
# detail on the patch edges is fully resolved.
Et_3D = CSX.AddDump('Et_3D', dump_type=0, dump_mode=0, file_type=0)
Et_3D.AddBox(
    start=[-substrate_width/2 - 5,  -substrate_length/2 - 5,  -5],
    stop= [ substrate_width/2 + 5,   substrate_length/2 + 5,  15]
)

# 2D XY slice: horizontal plane at the top of the substrate (patch level)
Et_xy = CSX.AddDump('Et_xy', dump_type=0, dump_mode=0, file_type=0)
Et_xy.AddBox(
    start=[-substrate_width/2,  -substrate_length/2,  substrate_thickness],
    stop= [ substrate_width/2,   substrate_length/2,  substrate_thickness]
)

# 2D XZ slice: vertical plane through the centre of the antenna at y = 0
Et_xz = CSX.AddDump('Et_xz', dump_type=0, dump_mode=0, file_type=0)
Et_xz.AddBox(
    start=[-substrate_width/2,  0,  -5],
    stop= [ substrate_width/2,  0,  15]
)

# ── Run simulation ────────────────────────────────────────────────────────────
if not post_proc_only:
    import shutil
    if SIM_DATA.exists():
        shutil.rmtree(SIM_DATA)
    SIM_DATA.mkdir(parents=True)
    print(f'\nRunning FDTD simulation ({NrTS_anim} timesteps) ...')
    print(f'Output → {SIM_DATA}\n')
    FDTD.Run(str(SIM_DATA), verbose=1, cleanup=False)

# ── Write PVD time-series index files ─────────────────────────────────────────
def write_pvd(pvd_path: Path, sim_dir: Path, prefix: str):
    """Write a ParaView .pvd collection file for files named {prefix}_NNNN.vtr."""
    files = sorted(
        sim_dir.glob(f'{prefix}_*.vtr'),
        key=lambda f: int(f.stem.split('_')[-1])
    )
    if not files:
        print(f'  WARNING: no VTK files found for {prefix} in {sim_dir}')
        return
    with open(pvd_path, 'w') as fh:
        fh.write('<?xml version="1.0"?>\n')
        fh.write('<VTKFile type="Collection" version="0.1" byte_order="LittleEndian">\n')
        fh.write('  <Collection>\n')
        for i, vf in enumerate(files):
            rel = os.path.relpath(vf, pvd_path.parent).replace('\\', '/')
            fh.write(f'    <DataSet timestep="{i}" group="" part="0" file="{rel}"/>\n')
        fh.write('  </Collection>\n')
        fh.write('</VTKFile>\n')
    print(f'  Written: {pvd_path.name}  ({len(files)} timesteps)')

print('\nWriting PVD index files ...')
for name in ('Et_3D', 'Et_xy', 'Et_xz'):
    write_pvd(VIZ_DIR / f'{name}.pvd', SIM_DATA, name)

# ── Generate pvpython visualisation script ────────────────────────────────────
PVPYTHON_SCRIPT = r'''# -*- coding: utf-8 -*-
"""
E-Field Time Animation — pvpython visualisation script
Generated by Simple_Patch_Antenna_Display.py

Run in batch mode (no GUI, saves screenshots + state):
    pvpython visualize.py

Open in the ParaView GUI instead:
    paraview --script=visualize.py
"""

import os, sys
from pathlib import Path

# ── Auto-relaunch with pvpython if run under regular Python ──────────────────
try:
    from paraview import simple as pv
except ModuleNotFoundError:
    import subprocess
    PVPYTHON = Path(r'C:\Program Files\ParaView 6.1.0\bin\pvpython.exe')
    if not PVPYTHON.exists():
        import shutil
        found = shutil.which('pvpython')
        if found:
            PVPYTHON = Path(found)
        else:
            print('ERROR: paraview module not found and pvpython.exe not found.')
            print('Run this script with:  pvpython visualize.py')
            sys.exit(1)
    print(f'Relaunching with pvpython: {PVPYTHON}')
    result = subprocess.run([str(PVPYTHON), __file__] + sys.argv[1:])
    sys.exit(result.returncode)

VIZ_DIR    = Path(__file__).resolve().parent
SHOTS_DIR  = VIZ_DIR / 'screenshots'
STATE_FILE = str(VIZ_DIR / 'patch_efield.pvsm')
SHOTS_DIR.mkdir(exist_ok=True)

# ── Helper: load a PVD time series ───────────────────────────────────────────
def load_pvd(name):
    pvd = str(VIZ_DIR / f'{name}.pvd')
    if not os.path.exists(pvd):
        print(f'WARNING: {pvd} not found, skipping')
        return None
    return pv.PVDReader(FileName=pvd)

reader_3d = load_pvd('Et_3D')
reader_xy = load_pvd('Et_xy')
reader_xz = load_pvd('Et_xz')

if reader_3d is None and reader_xy is None:
    print('ERROR: No PVD files found. Run the simulation first.')
    sys.exit(1)

# ── Compute global E-field magnitude range across ALL timesteps ───────────────
print('Computing global E-field range...')
primary = reader_3d or reader_xy
global_max = 0.0
for ts in primary.TimestepValues:
    pv.UpdatePipeline(ts, primary)
    rng = primary.PointData['E-Field'].GetRange(-1)
    if rng[1] > global_max:
        global_max = rng[1]
print(f'  E-field magnitude peak = {global_max:.3e} V/m')
pv.GetAnimationScene().AnimationTime = primary.TimestepValues[0]

# ── Layout ────────────────────────────────────────────────────────────────────
layout = pv.CreateLayout('E-Field Layout')

view3d = pv.CreateRenderView()
pv.AssignViewToLayout(view=view3d, layout=layout, hint=0)
right_loc = layout.SplitViewHorizontal(view3d, 0.5)
view3d.Background = [0.1, 0.1, 0.15]

view_xy = pv.CreateRenderView()
pv.AssignViewToLayout(view=view_xy, layout=layout, hint=right_loc)
bot_loc = layout.SplitViewVertical(view_xy, 0.5)
view_xy.Background = [0.05, 0.05, 0.1]

view_xz = pv.CreateRenderView()
pv.AssignViewToLayout(view=view_xz, layout=layout, hint=bot_loc)
view_xz.Background = [0.05, 0.05, 0.1]

# ── Shared colormap (fixed global range) ─────────────────────────────────────
def setup_colormap(display, view, global_max):
    pv.ColorBy(display, ('POINTS', 'E-Field'))
    lut = pv.GetColorTransferFunction('E-Field')
    lut.ApplyPreset('Rainbow Desaturated', True)
    lut.RescaleTransferFunction(0.0, global_max)
    display.SetScalarBarVisibility(view, True)
    pv.GetScalarBar(lut, view).Title = 'E-Field (V/m)'

# ── 3D view: two crossed slices at the antenna ────────────────────────────────
if reader_3d:
    sl_xy = pv.Slice(Input=reader_3d)
    sl_xy.SliceType = 'Plane'
    sl_xy.SliceType.Normal = [0, 0, 1]
    sl_xy.SliceType.Origin = [0, 0, 1.524]
    d_xy = pv.Show(sl_xy, view3d)
    d_xy.Representation = 'Surface'
    setup_colormap(d_xy, view3d, global_max)

    sl_xz = pv.Slice(Input=reader_3d)
    sl_xz.SliceType = 'Plane'
    sl_xz.SliceType.Normal = [0, 1, 0]
    sl_xz.SliceType.Origin = [0, 0, 0]
    d_xz = pv.Show(sl_xz, view3d)
    d_xz.Representation = 'Surface'
    setup_colormap(d_xz, view3d, global_max)

    view3d.ResetCamera()
    view3d.CameraPosition   = [120, -200, 100]
    view3d.CameraFocalPoint = [0, 0, 5]
    view3d.CameraViewUp     = [0, 0, 1]

# ── XY plane view ─────────────────────────────────────────────────────────────
if reader_xy:
    disp_xy = pv.Show(reader_xy, view_xy)
    disp_xy.Representation = 'Surface'
    setup_colormap(disp_xy, view_xy, global_max)
    view_xy.ResetCamera()
    view_xy.CameraParallelProjection = 1
    view_xy.CameraViewUp   = [0, 1, 0]
    view_xy.CameraPosition = [0, 0, 500]
    view_xy.CameraFocalPoint = [0, 0, 1.524]
    text_xy = pv.Text()
    text_xy.Text = 'XY plane - patch level'
    pv.Show(text_xy, view_xy).FontSize = 10

# ── XZ plane view ─────────────────────────────────────────────────────────────
if reader_xz:
    disp_xz = pv.Show(reader_xz, view_xz)
    disp_xz.Representation = 'Surface'
    setup_colormap(disp_xz, view_xz, global_max)
    view_xz.ResetCamera()
    view_xz.CameraParallelProjection = 1
    view_xz.CameraViewUp   = [0, 0, 1]
    view_xz.CameraPosition = [0, -500, 5]
    view_xz.CameraFocalPoint = [0, 0, 5]
    text_xz = pv.Text()
    text_xz.Text = 'XZ plane - cross-section'
    pv.Show(text_xz, view_xz).FontSize = 10

# ── Animate and save screenshots ──────────────────────────────────────────────
anim = pv.GetAnimationScene()
anim.PlayMode = 'Snap To TimeSteps'
timesteps = list(primary.TimestepValues)
print(f'Saving {len(timesteps)} screenshots to {SHOTS_DIR} ...')
for ts in timesteps:
    anim.AnimationTime = ts
    pv.Render(view3d)
    pv.SaveScreenshot(
        str(SHOTS_DIR / f'Et_t{int(ts):04d}.png'),
        view=view3d,
        ImageResolution=[1200, 900]
    )
print('Screenshots done.')

# ── Build animated GIF from screenshots ──────────────────────────────────────
GIF_FILE = str(SHOTS_DIR.parent / 'efield_animation.gif')
try:
    from PIL import Image
    frames = []
    for ts in timesteps:
        png = SHOTS_DIR / f'Et_t{int(ts):04d}.png'
        if png.exists():
            img = Image.open(str(png))
            w, h = img.size
            frames.append(img.resize((w // 2, h // 2), Image.LANCZOS))
    if frames:
        frames[0].save(
            GIF_FILE,
            save_all=True,
            append_images=frames[1:],
            duration=100,
            loop=0
        )
        print(f'GIF saved: {GIF_FILE}  ({len(frames)} frames)')
    else:
        print('No screenshot frames found for GIF.')
except ImportError:
    print('Pillow not available — skipping GIF export.')

# ── Save ParaView state file (.pvsm) ──────────────────────────────────────────
pv.SaveState(STATE_FILE)
print(f'State saved: {STATE_FILE}')
print(f'To reopen:   paraview --state="{STATE_FILE}"')
'''

pvscript_path = VIZ_DIR / 'visualize.py'
pvscript_path.write_text(PVPYTHON_SCRIPT, encoding='utf-8')
print(f'\nParaView script written: {pvscript_path}')

# ── Launch ParaView ───────────────────────────────────────────────────────────
if launch_paraview:
    if not PARAVIEW.exists():
        print(f'\nWARNING: ParaView not found at {PARAVIEW}')
        print('Open ParaView manually and load the .pvd files from:')
        print(f'  {VIZ_DIR}')
    else:
        pvd_3d = VIZ_DIR / 'Et_3D.pvd'
        if pvd_3d.exists():
            print(f'\nLaunching ParaView with {pvd_3d.name} ...')
            subprocess.Popen([str(PARAVIEW), str(pvd_3d)])
        else:
            print(f'\nLaunching ParaView ...')
            subprocess.Popen([str(PARAVIEW)])

print('\nDone.')
print(f'Visualizations folder: {VIZ_DIR}')
print(f'  Et_3D.pvd   — 3D E-field time series')
print(f'  Et_xy.pvd   — XY slice at patch level')
print(f'  Et_xz.pvd   — XZ cross-section slice')
print(f'  visualize.py — run with pvpython for screenshots + state file')
