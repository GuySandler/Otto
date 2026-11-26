"""Background jobs.

Long work ("read this whole thread and summarise it", "check my week and draft
the agenda") should not hold the conversation hostage. A job runs detached,
streams status to the UI, and when it finishes the session gets to *speak* the
result — immediately if it is idle, or as soon as the current turn is over.
"""
from __future__ import annotations

import asyncio
import logging
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable

log = logging.getLogger("otto.jobs")

QUEUED = "queued"
RUNNING = "running"
DONE = "done"
FAILED = "failed"
CANCELLED = "cancelled"


@dataclass(slots=True)
class Job:
    id: str
    title: str
    kind: str = "task"
    status: str = QUEUED
    created: float = field(default_factory=time.time)
    started: float | None = None
    finished: float | None = None
    result: str = ""
    error: str | None = None
    meta: dict[str, Any] = field(default_factory=dict)
    task: asyncio.Task[Any] | None = None

    @property
    def ms(self) -> int:
        if self.started is None:
            return 0
        end = self.finished or time.time()
        return int((end - self.started) * 1000)

    def to_event(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "title": self.title,
            "kind": self.kind,
            "status": self.status,
            "ms": self.ms,
            "result": self.result[:1500],
            "error": self.error,
        }


class JobManager:
    """Per-session background work queue."""

    def __init__(
        self,
        settings,
        *,
        on_event: Callable[[Job], None] | None = None,
        on_done: Callable[[Job], None] | None = None,
        max_concurrent: int | None = None,
    ) -> None:
        cfg = settings.cfg("jobs", default={}) or {}
        self.max_concurrent = max_concurrent or int(cfg.get("max_concurrent", 3))
        self.keep = int(cfg.get("keep_results", 12))
        self.on_event = on_event
        self.on_done = on_done
        self._jobs: dict[str, Job] = {}
        self._sem = asyncio.Semaphore(self.max_concurrent)
        self._closed = False

    # ------------------------------------------------------------------ public
    def start(
        self,
        title: str,
        factory: Callable[[], Awaitable[str]],
        *,
        kind: str = "task",
        meta: dict[str, Any] | None = None,
    ) -> Job:
        job = Job(id=f"job_{uuid.uuid4().hex[:8]}", title=title[:120], kind=kind, meta=meta or {})
        self._jobs[job.id] = job
        self._prune()
        self._emit(job)
        job.task = asyncio.create_task(self._run(job, factory), name=f"otto-{job.id}")
        return job

    def get(self, job_id: str) -> Job | None:
        return self._jobs.get(job_id)

    def list(self) -> list[Job]:
        return sorted(self._jobs.values(), key=lambda job: job.created, reverse=True)

    def running(self) -> list[Job]:
        return [job for job in self._jobs.values() if job.status in (QUEUED, RUNNING)]

    def finished(self, limit: int = 5) -> list[Job]:
        done = [job for job in self._jobs.values() if job.status in (DONE, FAILED, CANCELLED)]
        return sorted(done, key=lambda job: job.finished or 0, reverse=True)[:limit]

    def cancel(self, job_id: str) -> bool:
        job = self._jobs.get(job_id)
        if job is None or job.task is None or job.task.done():
            return False
        job.status = CANCELLED
        job.finished = time.time()
        job.task.cancel()
        self._emit(job)
        return True

    async def aclose(self) -> None:
        self._closed = True
        tasks = [job.task for job in self._jobs.values() if job.task and not job.task.done()]
        for task in tasks:
            task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

    # ----------------------------------------------------------------- internal
    async def _run(self, job: Job, factory: Callable[[], Awaitable[str]]) -> None:
        async with self._sem:
            if self._closed or job.status == CANCELLED:
                return
            job.status = RUNNING
            job.started = time.time()
            self._emit(job)
            try:
                result = await factory()
                if job.status != CANCELLED:
                    job.status = DONE
                    job.result = str(result or "").strip()
            except asyncio.CancelledError:
                job.status = CANCELLED
                raise
            except Exception as exc:  # noqa: BLE001
                log.exception("job %s failed", job.id)
                job.status = FAILED
                job.error = f"{type(exc).__name__}: {exc}"
            finally:
                job.finished = time.time()
                self._emit(job)
                if job.status in (DONE, FAILED) and self.on_done is not None and not self._closed:
                    try:
                        self.on_done(job)
                    except Exception:  # noqa: BLE001
                        log.exception("job done callback failed")

    def _emit(self, job: Job) -> None:
        if self.on_event is None:
            return
        try:
            self.on_event(job)
        except Exception:  # noqa: BLE001
            log.exception("job event callback failed")

    def _prune(self) -> None:
        if len(self._jobs) <= self.keep:
            return
        ordered = sorted(self._jobs.values(), key=lambda job: job.created)
        for job in ordered:
            if len(self._jobs) <= self.keep:
                break
            if job.status in (DONE, FAILED, CANCELLED):
                self._jobs.pop(job.id, None)
