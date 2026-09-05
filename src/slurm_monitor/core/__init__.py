"""Agent core: every query the agent can answer, implemented with pyslurm only.

Importing this package must not require pyslurm; the pyslurm-backed implementation lives in
core.queries and is imported lazily by the agent so the client package installs anywhere.
"""
