# slurm-monitor

Terminal UI for watching your own Slurm jobs. Runs on your laptop and drives a small agent on the
cluster login node over one persistent SSH connection, or runs directly on the cluster.

## Install

    uv tool install .            # from a checkout; puts `slurm-monitor` on your PATH

## Configure

`~/.config/slurm-monitor/config.toml`:

    default_cluster = "greatlakes"

    [clusters.greatlakes]
    host = "you@login.example.edu"

## Prepare the cluster (once per Slurm version)

    slurm-monitor bootstrap --cluster greatlakes

Builds a Python environment with pyslurm and the filtered-query extension under
`~/.slurm-monitor` on the login node. Takes a few minutes the first time.

## Run

    slurm-monitor                      # uses default_cluster
    slurm-monitor --cluster greatlakes
    slurm-monitor --local              # on the cluster itself, no ssh

Keys: `a` Active tab, `h` History tab, `Tab` switch tab, `r` refresh, `Enter`/`Space` expand or
collapse an array row, `c` cancel the selected job (confirm with `y`, abandon with `n`/`Esc`),
`d` toggle the detail/usage panel, `?` help, `q` quit.

## Check the connection without opening the UI

    slurm-monitor check --cluster greatlakes

Prints the handshake: Slurm version the agent was built against, the version the cluster runs,
and whether the filtered job query extension loaded. A mismatch after a cluster upgrade means
`slurm-monitor bootstrap --force`.

## Development

    uv sync --group dev && uv run pytest -q     # fixture tier, runs anywhere
    nix develop                                 # toolchain; on Linux also Slurm 25.11.6 headers/libs
    mise run container:build && mise run container:shell   # Linux devShell on macOS via Apple container
    SLURM_MONITOR_CLUSTER_TESTS=1 pytest tests/cluster     # on the login node, against the live controller

Recorded fixtures live under `tests/fixtures/recorded/`; `tests/cluster/record_fixtures.py` regenerates
them on the cluster. The extension source is `src/slurm_monitor/ext/`; `patches/` carries the same
change as an upstream pyslurm patch.
