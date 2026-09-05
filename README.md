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

Keys: `a` Active tab, `h` History tab, `r` refresh, `enter`/`space` expand array, `c` cancel, `q` quit.
