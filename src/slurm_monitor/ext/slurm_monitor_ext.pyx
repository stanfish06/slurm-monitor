# cython: c_string_type=unicode, c_string_encoding=default
# cython: language_level=3
"""slurm_load_job_user() binding for pyslurm.

pyslurm.Jobs.load() calls slurm_load_jobs(), which has no user filter and returns the whole
cluster job table. This module mirrors Jobs.load() but calls slurm_load_job_user(), so the
controller returns only one user's jobs. Everything after the RPC (Job.from_ptr, the tmp_info
zeroing guard, the per-cluster map insertion) is the same code path as Jobs.load().

Measured on a Great Lakes login node (Slurm 25.11.5, pyslurm 25.11.2, gcc 8.5, 2026-09-05,
6,735 jobs in the controller's table, one of them ours):
    load_user_jobs()      8.6 ms first call, then 4.6-4.8 ms  (5 runs)
    pyslurm.Jobs.load()   562 ms, 416 ms                      (2 runs; ~4.2 s at 46k jobs)
Filtering: every returned job had user_id == os.getuid(); the id set matched `squeue -u $USER`
and matched Jobs.load() filtered in Python.
"""

import os

from libc.stdint cimport uint16_t, uint32_t
from libc.string cimport memset

from pyslurm cimport slurm
from pyslurm.core.job.job cimport Job, Jobs
from pyslurm.slurm cimport slurm_job_info_t, slurm_load_job_user

from pyslurm.core.error import verify_rpc
from pyslurm.utils.helpers import _getgrall_to_dict, _getpwall_to_dict


def extension_version():
    """(major, minor, micro) of the Slurm headers this module was compiled against."""
    cdef uint32_t v = slurm.SLURM_VERSION_NUMBER
    return ((v >> 16) & 0xFF, (v >> 8) & 0xFF, v & 0xFF)


def load_user_jobs(uid=None, preload_passwd_info=False, frozen=False):
    """Load one user's jobs from slurmctld via slurm_load_job_user().

    Args:
        uid: numeric user id; defaults to os.getuid().
        preload_passwd_info: as in pyslurm.Jobs.load().
        frozen: as in pyslurm.Jobs.load().

    Returns:
        pyslurm.Jobs holding only jobs whose user_id == uid.

    Raises:
        pyslurm.RPCError when the controller refuses or the RPC fails.
    """
    cdef:
        dict passwd = {}
        dict groups = {}
        Jobs jobs = Jobs(frozen=frozen)
        uint16_t flags = slurm.SHOW_ALL | slurm.SHOW_DETAIL
        uint32_t user_id = os.getuid() if uid is None else int(uid)
        Job job

    # The one line that differs from Jobs.load(): filter by user on the controller side.
    verify_rpc(slurm_load_job_user(&jobs.info, user_id, flags))

    if preload_passwd_info:
        passwd = _getpwall_to_dict()
        groups = _getgrall_to_dict()

    # Zeroed record used to replace each entry after its pointer is handed to a Job, so a
    # MemoryError mid-loop cannot lead to a double free in Jobs.__dealloc__.
    memset(&jobs.tmp_info, 0, sizeof(slurm_job_info_t))

    for cnt in range(jobs.info.record_count):
        job = Job.from_ptr(&jobs.info.job_array[cnt])
        jobs.info.job_array[cnt] = jobs.tmp_info

        if preload_passwd_info:
            job.passwd = passwd
            job.groups = groups

        cluster = job.cluster
        if cluster not in jobs.data:
            jobs.data[cluster] = {}
        jobs[cluster][job.id] = job

    # All pointers extracted; stop __dealloc__ from freeing the per-job records twice.
    jobs.info.record_count = 0
    jobs.frozen = frozen
    return jobs
