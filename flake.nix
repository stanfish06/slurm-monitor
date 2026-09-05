{
  description = "slurm-monitor development environment (Slurm 25.11.6 from nixpkgs on Linux)";

  # nixpkgs revision whose `slurm` is 25-11-6-1; nixpkgs-unstable has moved to 26.05.x.
  inputs.nixpkgs.url = "github:NixOS/nixpkgs/d57af924f160a5084293c71c2043f058bd1cdb60";

  outputs =
    { self, nixpkgs }:
    let
      systems = [
        "aarch64-darwin"
        "x86_64-darwin"
        "aarch64-linux"
        "x86_64-linux"
      ];
      forAllSystems = f: nixpkgs.lib.genAttrs systems (system: f nixpkgs.legacyPackages.${system});
    in
    {
      devShells = forAllSystems (
        pkgs:
        let
          inherit (pkgs) lib stdenv;
          slurmLib = lib.getLib pkgs.slurm;
          slurmDev = lib.getDev pkgs.slurm;
        in
        {
          default = pkgs.mkShell {
            packages =
              with pkgs;
              [
                python312
                uv
                ruff
                mise
              ]
              # Slurm is Linux-only in nixpkgs; pyslurm and the Cython extension build only here.
              ++ lib.optionals stdenv.isLinux [
                gcc
                pkg-config
                slurm
                slurm.dev
                python312Packages.cython
                python312Packages.setuptools
              ];

            shellHook = lib.optionalString stdenv.isLinux ''
              # pyslurm's setup.py appends /slurm to both paths itself; it looks for
              # $SLURM_LIB_DIR/slurm/libslurmfull.so and $SLURM_INCLUDE_DIR/slurm/slurm_version.h.
              export SLURM_INCLUDE_DIR=${slurmDev}/include
              export SLURM_LIB_DIR=${slurmLib}/lib
              export CC=gcc
              # libslurmfull.so lives in a subdirectory the dynamic loader does not search by default.
              export LD_LIBRARY_PATH=${slurmLib}/lib/slurm''${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
            '';
          };
        }
      );
    };
}
