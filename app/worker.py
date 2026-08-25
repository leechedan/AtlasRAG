"""RQ worker process entrypoint.

Starts the directory watch manager's background threads (a watchdog
Observer per registered watched directory, plus the hourly reconcile
loop) and then blocks running an RQ ``SimpleWorker`` that consumes the
``"default"`` queue -- the queue name ``get_queue()`` (Task 13) uses when
no explicit name is given.

Why ``SimpleWorker`` and not ``Worker`` (Critical Finding 1 fix): ``rq.Worker``
forks a work-horse subprocess per job (confirmed by reading
``rq.worker.Worker.execute_job`` -> ``fork_work_horse`` in the installed
rq==2.1.0). This process never touches torch/models itself, so each forked
child started with an empty ``@lru_cache`` and reloaded BGE-M3 + Chinese-CLIP
+ PaddleOCR from scratch on every single job, then exited -- models were
never actually resident/warm in the worker, contradicting the intent that
the worker keep models resident across jobs. ``SimpleWorker.execute_job``
(verified by reading the installed rq==2.1.0 source) runs
``self.prepare_execution(job); self.perform_job(job, queue)`` directly in
the current process/thread -- no fork -- so the ``@lru_cache``d model
singletons (``get_text_embedder()``, ``get_clip_embedder()``) actually stay
warm across jobs. Without forking, an OS-level crash inside a job (e.g. a
segfault) could in principle take down this worker process itself, in a way
forking would have contained -- a known, accepted tradeoff here since it
also means no CUDA-after-fork hazard. Normal Python exceptions raised by a
job are unaffected: ``Worker.perform_job`` (inherited by ``SimpleWorker``,
unchanged) still catches them with a bare ``except:`` and marks the job
FAILED via ``handle_job_failure`` -- confirmed by reading
``rq.worker.Worker.perform_job`` -- so job-level exception handling and
status reporting work exactly as before.

Note on shutdown: both of ``DirectoryWatchManager``'s background threads
are daemon threads -- the watchdog ``Observer`` inherits ``daemon = True``
from ``watchdog.utils.BaseThread.__init__`` (verified by reading the
installed watchdog package directly), and the reconcile-loop thread is
created with ``daemon=True`` explicitly (see ``app/ingest/watcher.py``).
Daemon threads are killed automatically when the process exits, so there
is nothing that would hang after ``worker.work()`` returns (e.g. on
SIGINT/SIGTERM, which RQ's ``Worker.work()``/``SimpleWorker.work()`` already
handles for graceful job-level shutdown) even though ``watch_manager.stop()``
is never called here.
"""

import logging

from rq import SimpleWorker

from app.ingest.watcher import DirectoryWatchManager
from app.jobs.queue import get_redis_connection

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main() -> None:
    redis_conn = get_redis_connection()

    watch_manager = DirectoryWatchManager(redis_conn)
    watch_manager.start()
    logger.info("directory watch manager started")

    worker = SimpleWorker(["default"], connection=redis_conn)
    worker.work()


if __name__ == "__main__":
    main()
