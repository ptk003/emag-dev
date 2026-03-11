"""
openEMS Windows Installer
=========================
Automates installation of openEMS (electromagnetic field solver) on Windows.

Usage:
    python install_openEMS.py [--openems-dir PATH] [--python PATH] [--no-path]

Options:
    --openems-dir PATH   Path to extracted openEMS directory
                         (default: same folder as this script, looks for 'openEMS' subdir)
    --python PATH        Path to python.exe to use (default: auto-detect)
    --no-path            Skip adding openEMS to system PATH

Requirements:
    - Python 3.13 or 3.14 (wheel files are included for these versions)
    - The extracted openEMS directory (containing openEMS.dll, openEMS.exe, etc.)
    - The python/ subdirectory inside openEMS with .whl files

What this script does:
    1. Locates the openEMS installation directory and Python executable
    2. Installs Python prerequisites: numpy, h5py, matplotlib, cython
    3. Installs CSXCAD and openEMS Python wheel files
    4. Sets OPENEMS_INSTALL_PATH user environment variable
    5. Adds openEMS directory to user PATH
    6. Verifies the installation by importing the modules
"""

import argparse
import os
import subprocess
import sys
import glob
from pathlib import Path


# ── helpers ────────────────────────────────────────────────────────────────────

def run(cmd, check=True, **kwargs):
    """Run a command and print it."""
    print(f"  > {' '.join(str(c) for c in cmd)}")
    result = subprocess.run(cmd, **kwargs)
    if check and result.returncode != 0:
        print(f"  ERROR: command failed with exit code {result.returncode}")
        sys.exit(1)
    return result


def find_python():
    """Locate a Python 3.13/3.14 executable on Windows."""
    candidates = []

    # Check common install locations
    appdata = os.environ.get("LOCALAPPDATA", "")
    if appdata:
        for ver in ("3.14", "3.13"):
            for pattern in (
                f"{appdata}/Python/pythoncore-{ver}-64/python.exe",
                f"{appdata}/Programs/Python/Python{ver.replace('.','')}/python.exe",
            ):
                candidates.extend(glob.glob(pattern))

    # py launcher
    for ver in ("3.14", "3.13"):
        candidates.append(f"py -{ver}")

    # Whatever is on PATH
    candidates.append("python")

    for candidate in candidates:
        parts = candidate.split()
        try:
            result = subprocess.run(
                parts + ["--version"],
                capture_output=True, text=True
            )
            if result.returncode == 0 and "Python 3" in result.stdout + result.stderr:
                version = (result.stdout + result.stderr).strip()
                print(f"  Found: {candidate}  ({version})")
                return parts
        except FileNotFoundError:
            continue

    return None


def find_openems_dir(script_dir):
    """Try to locate the openEMS installation directory."""
    # Look for openEMS subdir next to this script
    candidate = script_dir / "openEMS"
    if (candidate / "openEMS.exe").exists():
        return candidate

    # Also check the script's parent
    candidate = script_dir.parent / "openEMS"
    if (candidate / "openEMS.exe").exists():
        return candidate

    return None


def get_python_version_tuple(python_cmd):
    """Return (major, minor) for the given python command list."""
    result = subprocess.run(
        python_cmd + ["-c", "import sys; print(sys.version_info.major, sys.version_info.minor)"],
        capture_output=True, text=True
    )
    major, minor = result.stdout.strip().split()
    return int(major), int(minor)


def find_wheel(python_dir, prefix, major, minor):
    """Find the best matching wheel file for the detected Python version."""
    tag = f"cp{major}{minor}"
    pattern = str(python_dir / f"{prefix}-*-{tag}-{tag}-win_amd64.whl")
    matches = glob.glob(pattern)
    if matches:
        return matches[0]

    # Fallback: any wheel with the right prefix
    fallback = glob.glob(str(python_dir / f"{prefix}-*.whl"))
    if fallback:
        print(f"  WARNING: No wheel for {tag}, falling back to: {fallback[0]}")
        return fallback[0]

    return None


def set_user_env(name, value):
    """Set a Windows user environment variable via PowerShell."""
    run([
        "powershell.exe", "-NoProfile", "-Command",
        f"[System.Environment]::SetEnvironmentVariable('{name}', '{value}', 'User')"
    ])


def add_to_path(directory):
    """Add a directory to the user PATH if not already present."""
    result = subprocess.run(
        ["powershell.exe", "-NoProfile", "-Command",
         "[System.Environment]::GetEnvironmentVariable('PATH', 'User')"],
        capture_output=True, text=True
    )
    current = result.stdout.strip()
    if str(directory) in current:
        print(f"  Already in PATH: {directory}")
        return

    new_path = f"{current};{directory}" if current else str(directory)
    run([
        "powershell.exe", "-NoProfile", "-Command",
        f"[System.Environment]::SetEnvironmentVariable('PATH', '{new_path}', 'User')"
    ])
    print(f"  Added to user PATH: {directory}")


def verify_install(python_cmd, openems_dir):
    """Import CSXCAD and openEMS to confirm the installation works."""
    code = (
        "import sys; "
        f"sys.path.insert(0, r'{openems_dir}\\python'); "
        "import CSXCAD, openEMS; "
        "print('CSXCAD OK'); print('openEMS OK')"
    )
    result = subprocess.run(
        python_cmd + ["-c", code],
        capture_output=True, text=True,
        env={**os.environ, "OPENEMS_INSTALL_PATH": str(openems_dir)}
    )
    if result.returncode == 0:
        for line in result.stdout.strip().splitlines():
            print(f"  {line}")
        return True
    else:
        print(f"  FAILED:\n{result.stderr}")
        return False


# ── main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Install openEMS and its Python bindings on Windows"
    )
    parser.add_argument(
        "--openems-dir",
        help="Path to the extracted openEMS directory (containing openEMS.exe)"
    )
    parser.add_argument(
        "--python",
        help="Path to python.exe (default: auto-detect)"
    )
    parser.add_argument(
        "--no-path",
        action="store_true",
        help="Skip adding openEMS to user PATH"
    )
    args = parser.parse_args()

    script_dir = Path(__file__).resolve().parent

    print("\n=== openEMS Windows Installer ===\n")

    # ── 1. Locate openEMS directory ──────────────────────────────────────────
    print("[1/5] Locating openEMS directory...")
    if args.openems_dir:
        openems_dir = Path(args.openems_dir).resolve()
    else:
        openems_dir = find_openems_dir(script_dir)

    if not openems_dir or not (openems_dir / "openEMS.exe").exists():
        print(
            "  ERROR: Could not find openEMS directory.\n"
            "  Please extract the openEMS zip first, then either:\n"
            "    - Place the openEMS folder next to this script, or\n"
            "    - Pass --openems-dir PATH\n"
            "  Download from: https://github.com/thliebig/openEMS-Project/releases"
        )
        sys.exit(1)

    print(f"  Found: {openems_dir}")
    python_dir = openems_dir / "python"

    # ── 2. Locate Python ────────────────────────────────────────────────────
    print("\n[2/5] Locating Python executable...")
    if args.python:
        python_cmd = [args.python]
        result = subprocess.run(
            python_cmd + ["--version"], capture_output=True, text=True
        )
        if result.returncode != 0:
            print(f"  ERROR: '{args.python}' is not a valid Python executable")
            sys.exit(1)
        print(f"  Using: {args.python}  ({(result.stdout + result.stderr).strip()})")
    else:
        python_cmd = find_python()
        if not python_cmd:
            print(
                "  ERROR: No Python 3.13/3.14 found.\n"
                "  Install Python from https://www.python.org/downloads/\n"
                "  or pass --python PATH"
            )
            sys.exit(1)

    major, minor = get_python_version_tuple(python_cmd)
    print(f"  Python version: {major}.{minor}")

    if (major, minor) not in ((3, 13), (3, 14)):
        print(
            f"  WARNING: Python {major}.{minor} detected.\n"
            f"  The included wheel files are built for Python 3.13 and 3.14.\n"
            f"  Installation may fail. Consider using a supported Python version."
        )

    # ── 3. Install Python prerequisites ────────────────────────────────────
    print("\n[3/5] Installing Python prerequisites...")
    prereqs = ["numpy", "h5py", "matplotlib", "cython"]
    run(python_cmd + ["-m", "pip", "install", "--upgrade"] + prereqs)

    # ── 4. Install wheel files ──────────────────────────────────────────────
    print("\n[4/5] Installing openEMS Python wheels...")
    for prefix in ("csxcad", "openems"):
        wheel = find_wheel(python_dir, prefix, major, minor)
        if not wheel:
            print(f"  ERROR: No wheel found for '{prefix}' in {python_dir}")
            sys.exit(1)
        print(f"  Installing: {Path(wheel).name}")
        run(python_cmd + ["-m", "pip", "install", wheel, "--force-reinstall"])

    # ── 5. Set environment variables ────────────────────────────────────────
    print("\n[5/5] Configuring environment variables...")
    set_user_env("OPENEMS_INSTALL_PATH", str(openems_dir))
    print(f"  OPENEMS_INSTALL_PATH = {openems_dir}")

    if not args.no_path:
        add_to_path(openems_dir)

    # ── Verify ───────────────────────────────────────────────────────────────
    print("\n[Verify] Testing Python imports...")
    ok = verify_install(python_cmd, openems_dir)

    print()
    if ok:
        print("Installation complete!")
        print()
        print("Quick start:")
        print(f"  cd {openems_dir / 'python' / 'Tutorials'}")
        print(f"  {python_cmd[-1]} Simple_Patch_Antenna.py")
        print()
        print("Note: Restart your terminal/IDE for PATH changes to take effect.")
    else:
        print("Installation finished but verification failed.")
        print("Try restarting your terminal and running:")
        print(f"  python -c \"import CSXCAD, openEMS; print('OK')\"")
        sys.exit(1)


if __name__ == "__main__":
    main()
