#!/usr/bin/env bash
# Build and run the Linux development container (Apple `container`, aarch64 VM).
#   container/dev.sh build            build the slurm-monitor-dev image (fetches the Slurm closure once)
#   container/dev.sh shell            interactive shell in the flake devShell with the repo at /work
#   container/dev.sh run CMD [ARG..]  run one command inside the devShell, non-interactively
set -euo pipefail

repo=$(cd "$(dirname "$0")/.." && pwd)
image=slurm-monitor-dev

build() {
    # Build context holds only the Containerfile and the flake files; the repo is bind-mounted at run time.
    ctx=$(mktemp -d)
    trap 'rm -rf "$ctx"' EXIT
    cp "$repo/container/Containerfile" "$repo/flake.nix" "$repo/flake.lock" "$ctx/"
    container build -t "$image" "$ctx"
}

# Mount the repo at /work; nix develop reads the flake from the mount, so lock changes apply on the next run.
run_in_dev() {
    container run --rm "$@"
}

case "${1:-shell}" in
    build) build ;;
    shell) run_in_dev -it -v "$repo:/work" -w /work "$image" nix develop ;;
    run) shift; run_in_dev -v "$repo:/work" -w /work "$image" nix develop -c "$@" ;;
    *) echo "usage: $0 {build|shell|run CMD...}" >&2; exit 2 ;;
esac
