"""Tiny in-process job manager for the analyzer page.

Runs synthesis on a 2-worker thread pool so at most two optimizations hit
the CPU at once; further jobs wait in the queue. Job state lives in memory
(the site runs a single uvicorn process) and expires after an hour.
No external queue -- YAGNI.
"""

import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field

from pages.mimoshape.analyzer import AnalysisParams, UploadError, analyze_and_reconstruct

MAX_WORKERS = 2
MAX_WAITING = 8  # pending jobs beyond the two running ones
JOB_TTL = 3600.0  # seconds a finished job (and its result) is kept


@dataclass
class Job:
    id: str
    params: AnalysisParams
    fs: float
    state: str = "queued"  # queued | running | done | error
    done_blocks: int = 0
    total_blocks: int = 0
    error: str = ""
    result: object = None
    created: float = field(default_factory=time.monotonic)


class JobManager:
    def __init__(self, max_workers=MAX_WORKERS):
        self._executor = ThreadPoolExecutor(max_workers=max_workers)
        self._jobs: dict[str, Job] = {}
        self._lock = threading.Lock()

    def submit(self, record, fs: float, params: AnalysisParams) -> str:
        with self._lock:
            self._expire()
            waiting = sum(1 for j in self._jobs.values() if j.state == "queued")
            if waiting >= MAX_WAITING:
                raise UploadError(
                    "the server is busy (too many queued jobs); try again "
                    "in a few minutes"
                )
            job = Job(id=uuid.uuid4().hex, params=params, fs=fs)
            self._jobs[job.id] = job
        self._executor.submit(self._run, job, record)
        return job.id

    def get(self, job_id: str) -> Job | None:
        with self._lock:
            return self._jobs.get(job_id)

    def _run(self, job: Job, record):
        job.state = "running"

        def progress(done, total):
            job.done_blocks, job.total_blocks = done, total

        try:
            job.result = analyze_and_reconstruct(
                record, job.fs, job.params, progress=progress
            )
            job.state = "done"
        except UploadError as ex:
            job.error = str(ex)
            job.state = "error"
        except Exception as ex:  # surface unexpected failures to the user
            job.error = f"internal error: {type(ex).__name__}: {ex}"
            job.state = "error"
        job.created = time.monotonic()  # TTL counts from completion

    def _expire(self):
        cutoff = time.monotonic() - JOB_TTL
        for jid in [
            jid
            for jid, j in self._jobs.items()
            if j.created < cutoff and j.state in ("done", "error")
        ]:
            del self._jobs[jid]
