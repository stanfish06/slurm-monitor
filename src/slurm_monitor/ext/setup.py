"""Build slurm_monitor_ext against the installed pyslurm and the host's Slurm.

Run on a Slurm host, from this directory, with the venv that has pyslurm installed:

    CC=gcc SLURM_INCLUDE_DIR=/usr/include SLURM_LIB_DIR=/usr/lib64 \\
        python setup.py build_ext --build-lib <site-packages>

The main slurm-monitor package never builds this; bootstrap runs it on the cluster. Include and
library discovery mirror pyslurm's own setup.py: headers at $SLURM_INCLUDE_DIR/slurm/*.h and
libslurmfull.so under $SLURM_LIB_DIR, $SLURM_LIB_DIR/slurm, or $SLURM_LIB_DIR/slurm-wlm.
"""

from __future__ import annotations

import os
from pathlib import Path

from Cython.Build import cythonize
from setuptools import Extension, setup

HERE = Path(__file__).resolve().parent
SLURM_LIB = "slurmfull"


def pyslurm_site_dir() -> Path:
    """Directory containing the installed pyslurm package (its .pxd files are cimported)."""
    import pyslurm

    return Path(pyslurm.__file__).resolve().parent.parent


def find_lib_dir(base: Path) -> Path:
    for candidate in (base, base / "slurm", base / "slurm-wlm"):
        if (candidate / f"lib{SLURM_LIB}.so").exists():
            return candidate
    raise RuntimeError(
        f"cannot find lib{SLURM_LIB}.so under {base} (set SLURM_LIB_DIR to the directory "
        "that contains slurm/libslurmfull.so)"
    )


def main() -> None:
    inc_dir = Path(os.environ.get("SLURM_INCLUDE_DIR", "/usr/include"))
    if not (inc_dir / "slurm" / "slurm.h").exists():
        raise RuntimeError(
            f"cannot find {inc_dir / 'slurm' / 'slurm.h'} (set SLURM_INCLUDE_DIR to the "
            "directory that contains slurm/slurm.h)"
        )
    lib_dir = find_lib_dir(Path(os.environ.get("SLURM_LIB_DIR", "/usr/lib64")))
    site_dir = pyslurm_site_dir()

    # Relative source path keeps cythonize's build_dir layout flat (build/slurm_monitor_ext.c).
    source = os.path.relpath(HERE / "slurm_monitor_ext.pyx", os.getcwd())
    ext = Extension(
        "slurm_monitor_ext",
        [source],
        include_dirs=[str(inc_dir), str(site_dir)],
        library_dirs=[str(lib_dir)],
        libraries=[SLURM_LIB],
        runtime_library_dirs=[str(lib_dir)],
    )
    setup(
        name="slurm_monitor_ext",
        version="0.1.0",
        ext_modules=cythonize(
            [ext],
            include_path=[str(site_dir)],
            compiler_directives={"language_level": 3},
            # Emit the generated .c under build/, not next to the .pyx in site-packages.
            build_dir="build",
        ),
    )


if __name__ == "__main__":
    main()
