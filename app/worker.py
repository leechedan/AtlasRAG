"""RQ worker process entrypoint.

Starts the directory watch manager's background threads (a watchdog
Observer per registered watched directory, plus the hourly reconcile
loop) and then blocks running an RQ ``Worker`` that consumes the
``"default"`` queue -- the queue name ``get_queue()`` (Task 13) uses when
no explicit name is given.

Note on shutdown: both of ``DirectoryWatchManager``'s background threads
are daemon threads -- the watchdog ``Observer`` inherits ``daemon = True``
from ``watchdog.utils.BaseThread.__init__`` (verified by reading the
installed watchdog package directly), and the reconcile-loop thread is
created with ``daemon=True`` explicitly (see ``app/ingest/watcher.py``).
Daemon threads are killed automatically when the process exits, so there
is nothing that would hang after ``worker.work()`` returns (e.g. on
SIGINT/SIGTERM, which RQ's ``Worker.work()`` already handles for
graceful job-level shutdown) even though ``watch_manager.stop()`` is
never called here.
"""

import logging

from rq import Worker

from app.ingest.watcher import DirectoryWatchManager
from app.jobs.queue import get_redis_connection

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main() -> None:
    redis_conn = get_redis_connection()

    watch_manager = DirectoryWatchManager(redis_conn)
    watch_manager.start()
    logger.info("directory watch manager started")

    worker = Worker(["default"], connection=redis_conn)
    worker.work()


if __name__ == "__main__":
    main()
