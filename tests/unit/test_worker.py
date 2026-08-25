from unittest.mock import MagicMock, patch

from app import worker


@patch("app.worker.DirectoryWatchManager")
@patch("app.worker.get_redis_connection")
@patch("app.worker.SimpleWorker")
def test_main_uses_simple_worker_not_forking_worker(mock_simple_worker_cls, mock_get_conn, mock_manager_cls):
    """Regression test for Critical Finding 1: app.worker previously used
    rq.Worker, which forks a work-horse subprocess per job (confirmed by
    reading rq.worker.Worker.execute_job -> fork_work_horse in the installed
    rq==2.1.0). The parent process never touches torch/models, so every
    forked child started with an empty @lru_cache and reloaded BGE-M3 +
    Chinese-CLIP + PaddleOCR from scratch on every single job -- models were
    never actually resident/warm in the worker. SimpleWorker executes jobs
    in-process (no fork), which is required for the @lru_cache'd model
    singletons (get_text_embedder(), get_clip_embedder()) to stay warm
    across jobs."""
    mock_conn = MagicMock()
    mock_get_conn.return_value = mock_conn
    mock_worker_instance = MagicMock()
    mock_simple_worker_cls.return_value = mock_worker_instance

    worker.main()

    mock_simple_worker_cls.assert_called_once_with(["default"], connection=mock_conn)
    mock_worker_instance.work.assert_called_once()
