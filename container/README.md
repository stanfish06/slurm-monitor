# Linux development container

macOS cannot build pyslurm or the Cython extension (nixpkgs Slurm is Linux-only). This directory
runs the flake's devShell inside an aarch64 Linux VM via Apple `container` (1.1+). The image bakes
in the Slurm 25.11.6 closure, so `nix develop` against the mounted repo starts without fetching.

    container system start                     # once per boot; also `container builder start` if builds time out
    mise run container:build                   # == container/dev.sh build
    mise run container:shell                   # == container/dev.sh shell -> interactive devShell at /work

`dev.sh build` copies `flake.nix` and `flake.lock` into a temporary build context with the
Containerfile, so the repo checkout (with `.venv`, `.git`) is never sent to the builder. Rebuild
after changing `flake.lock`.

One-off commands, non-interactively:

    container run --rm -v "$PWD:/work" -w /work slurm-monitor-dev nix develop -c \
        python -c "import ctypes; ctypes.CDLL('libslurmfull.so')"
    container run --rm -v "$PWD:/work" -w /work slurm-monitor-dev nix develop -c \
        sh -c 'grep SLURM_VERSION_STRING "$SLURM_INCLUDE_DIR/slurm/slurm_version.h"'

or `container/dev.sh run <cmd...>` for the same thing.

Inside the shell: `SLURM_INCLUDE_DIR`, `SLURM_LIB_DIR`, `CC=gcc` and `LD_LIBRARY_PATH` (for
`libslurmfull.so`) are exported by the flake's shellHook; `UV_PROJECT_ENVIRONMENT=/work/.venv-linux`
keeps the Linux venv apart from the macOS `.venv` on the bind mount. Build the agent side with
`uv sync --extra agent --group dev`.

Note the container is aarch64 while the cluster is x86_64: a build here validates the source, not
the artifact that ships. `slurm-monitor bootstrap` compiles the deployed extension on the cluster.
