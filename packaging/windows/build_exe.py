"""Build a self-contained single-file Windows EXE (`dist/TensorFlow_KI_Studio.exe`).

Uses the native Windows .NET Framework C# compiler (`csc.exe`) and bundles:
  - Native Windows instant splash screen & taskbar icon manager (`launcher_stub.cs`)
  - The complete `src/ki_studio/` package and static web UI
  - Factory `.keras` models + metadata (`models/`)
  - Offline MNIST dataset & app icons (`assets/`)
  - Bundled sample images (`samples/`)
"""

from __future__ import annotations

import hashlib
from pathlib import Path
import struct
import subprocess
import sys
import time
import zipfile

BUNDLED_FILES: tuple[str, ...] = (
    "models/zahlen_netz.keras",
    "models/zahlen_netz_meta.json",
    "models/smiley_netz.keras",
    "models/smiley_netz_meta.json",
    "assets/mnist.npz",
    "assets/keras.json",
    "assets/app_icon.ico",
    "assets/icon_256.png",
)

BOOTSTRAP_SCRIPT = '''"""Runtime bootstrap entrypoint inside the extracted Windows bundle."""

from pathlib import Path
import sys

SRC_DIR = Path(__file__).resolve().parent / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from ki_studio.cli.app import main

if __name__ == "__main__":
    main()
'''


def _collect_payload_files(project_dir: Path) -> list[tuple[Path, str]]:
    pack_list: list[tuple[Path, str]] = []

    for rel in BUNDLED_FILES:
        full_path = project_dir / rel
        if not full_path.is_file():
            raise FileNotFoundError(f"Required bundle file missing: {full_path}")
        pack_list.append((full_path, Path(rel).as_posix()))

    for folder in ("src/ki_studio", "samples"):
        base = project_dir / folder
        if not base.is_dir():
            raise FileNotFoundError(f"Required bundle directory missing: {base}")
        for entry in sorted(base.rglob("*")):
            if not entry.is_file():
                continue
            if "__pycache__" in entry.parts or entry.suffix in (".pyc", ".pyo"):
                continue
            pack_list.append((entry, entry.relative_to(project_dir).as_posix()))

    return pack_list


def build_single_exe(project_dir: Path, work_dir: Path, output_exe: Path) -> None:
    """Compile the native splash stub and append the compressed application payload."""
    t0 = time.perf_counter()
    csc_exe = Path(r"C:\Windows\Microsoft.NET\Framework64\v4.0.30319\csc.exe")
    if not csc_exe.is_file():
        raise FileNotFoundError(f"Windows C# compiler not found: {csc_exe}")

    work_dir.mkdir(parents=True, exist_ok=True)
    output_exe.parent.mkdir(parents=True, exist_ok=True)

    assets_dir = project_dir / "assets"
    pkg_dir = project_dir / "packaging" / "windows"
    cs_template = (pkg_dir / "launcher_stub.cs").read_text(encoding="utf-8")

    bootstrap_file = work_dir / "run_studio.pyw"
    bootstrap_file.write_text(BOOTSTRAP_SCRIPT, encoding="utf-8")

    pack_list = _collect_payload_files(project_dir) + [(bootstrap_file, "run_studio.pyw")]

    hasher = hashlib.sha256()
    hasher.update(str(time.time_ns()).encode("utf-8"))
    for full_path, arc_name in pack_list:
        hasher.update(f"{arc_name}:{full_path.stat().st_size}".encode("utf-8"))
    build_id = "TFKILITE_" + hasher.hexdigest()[:16]

    cs_generated = work_dir / "Launcher_Light_Generated.cs"
    cs_generated.write_text(
        cs_template.replace("__BUILD_ID_PLACEHOLDER__", build_id),
        encoding="utf-8-sig",
    )

    stub_exe = work_dir / "LauncherLightStub.exe"
    if stub_exe.exists():
        stub_exe.unlink()

    print(f"[1/3] Compiling native Windows splash launcher (Build ID: {build_id})...")
    cmd = [
        str(csc_exe),
        "/nologo",
        "/target:winexe",
        "/platform:x64",
        "/optimize+",
        f"/out:{stub_exe}",
        f"/win32icon:{assets_dir / 'app_icon.ico'}",
        f"/resource:{assets_dir / 'splash.png'},splash.png",
        f"/resource:{assets_dir / 'app_icon.ico'},app_icon.ico",
        "/r:System.dll",
        "/r:System.Drawing.dll",
        "/r:System.Windows.Forms.dll",
        "/r:System.IO.Compression.dll",
        str(cs_generated),
    ]
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0:
        raise RuntimeError(f"csc.exe compilation failed:\nSTDOUT: {res.stdout}\nSTDERR: {res.stderr}")
    print(f"      LauncherLightStub.exe built ({stub_exe.stat().st_size / 1024:.1f} KB)")

    payload_zip = work_dir / "light_payload.zip"
    if payload_zip.exists():
        payload_zip.unlink()

    print(f"[2/3] Packing {len(pack_list)} app, model, and offline MNIST files...")
    with zipfile.ZipFile(
        payload_zip,
        mode="w",
        compression=zipfile.ZIP_DEFLATED,
        allowZip64=False,
        compresslevel=9,
    ) as zf:
        for full_path, arc_name in pack_list:
            zf.write(full_path, arcname=arc_name)

    zip_size = payload_zip.stat().st_size
    print(f"      light_payload.zip ready ({zip_size / (1024 * 1024):.2f} MB)")

    print(f"[3/3] Creating single-file EXE -> {output_exe} ...")
    temp_out = work_dir / "TensorFlow_KI_Studio_light.exe"
    stub_bytes = stub_exe.read_bytes()
    zip_offset = len(stub_bytes)

    with open(temp_out, "wb") as out_f:
        out_f.write(stub_bytes)
        out_f.write(payload_zip.read_bytes())
        out_f.write(struct.pack("<qq8s", zip_offset, zip_size, b"TFKILITE"))

    if output_exe.exists():
        output_exe.unlink()
    temp_out.replace(output_exe)

    total_mb = output_exe.stat().st_size / (1024 * 1024)
    print(f"DONE! {output_exe.name} built ({total_mb:.2f} MB in {time.perf_counter() - t0:.2f}s).")


if __name__ == "__main__":
    proj = Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else Path(__file__).resolve().parents[2]
    work = Path(sys.argv[2]).resolve() if len(sys.argv) > 2 else (proj / "_build_tmp")
    out = Path(sys.argv[3]).resolve() if len(sys.argv) > 3 else (proj / "dist" / "TensorFlow_KI_Studio.exe")
    build_single_exe(proj, work, out)
