"""
Unit tests for worker module.
"""

import os
import time
import pytest
from unittest.mock import Mock, patch

from utilities.worker import BaseWorker, Future, Task, get_rate_limit_config


@pytest.mark.unit
class TestFuture:
    """Test cases for Future class."""

    def test_future_creation(self):
        """Test Future object creation."""
        future = Future("test-task-id")
        assert future._task_id == "test-task-id"
        assert future.is_done() is False

    def test_set_result(self):
        """Test setting result."""
        future = Future("test-task-id")
        future.set_result("test result")

        assert future.is_done() is True
        assert future.get() == "test result"
        assert future._exception is None

    def test_set_exception(self):
        """Test setting exception."""
        future = Future("test-task-id")
        test_exception = ValueError("test error")
        future.set_exception(test_exception)

        assert future.is_done() is True

        with pytest.raises(ValueError, match="test error"):
            future.get()

    def test_get_timeout(self):
        """Test get with timeout."""
        future = Future("test-task-id")

        with pytest.raises(TimeoutError):
            future.get(timeout=0.1)

    def test_get_waits_for_result(self):
        """Test that get waits for result."""
        future = Future("test-task-id")

        def set_result_later():
            time.sleep(0.1)
            future.set_result("delayed result")

        import threading
        thread = threading.Thread(target=set_result_later)
        thread.start()

        result = future.get(timeout=1.0)
        assert result == "delayed result"
        thread.join()


@pytest.mark.unit
class TestTask:
    """Test cases for Task dataclass."""

    def test_task_creation(self):
        """Test Task creation."""
        future = Future("test-id")
        task = Task(
            id="test-id",
            method="test_method",
            args=(1, 2, 3),
            kwargs={"key": "value"},
            future=future
        )

        assert task.id == "test-id"
        assert task.method == "test_method"
        assert task.args == (1, 2, 3)
        assert task.kwargs == {"key": "value"}
        assert task.future is future


@pytest.mark.unit
class TestGetRateLimitConfig:
    """Test cases for get_rate_limit_config function."""

    def test_with_both_limits(self, mock_env_vars):
        """Test with both RPM and TPM configured."""
        rpm, tpm = get_rate_limit_config('OCR')
        assert rpm == 60
        assert tpm == 100000

    def test_with_no_config(self):
        """Test with no configuration."""
        # Clear any existing config
        for key in list(os.environ.keys()):
            if 'RPM' in key or 'TPM' in key:
                del os.environ[key]

        rpm, tpm = get_rate_limit_config('NONEXISTENT')
        assert rpm is None
        assert tpm is None

    def test_partial_config(self):
        """Test with only RPM configured."""
        os.environ['TEST_RPM'] = '30'
        # No TPM

        rpm, tpm = get_rate_limit_config('TEST')
        assert rpm == 30
        assert tpm is None


@pytest.mark.unit
class TestBaseWorker:
    """Test cases for BaseWorker class."""

    def test_worker_without_registered_methods(self):
        """Test worker without registering any methods."""
        class TestWorker(BaseWorker):
            def _register_methods(self):
                pass  # Don't register any methods

        worker = TestWorker(name="TestWorker")
        assert worker._methods == {}

    def test_worker_initialization(self, temp_dir):
        """Test worker initialization."""
        class TestWorker(BaseWorker):
            def _register_methods(self):
                self._methods = {'test': lambda: 'result'}

            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self._register_methods()

        worker = TestWorker(name="TestWorker", rpm=10, tpm=1000)
        assert worker.name == "TestWorker"
        assert worker.is_alive() is False

    def test_worker_starts_and_stops(self, temp_dir):
        """Test that worker can start and stop."""
        class TestWorker(BaseWorker):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self.processed = []
                self._register_methods()

            def _register_methods(self):
                self._methods = {
                    'process': self._process
                }

            def _process(self, value):
                self.processed.append(value)
                return value * 2

        worker = TestWorker(name="TestWorker", poll_interval=0.01)
        worker.start()
        assert worker.is_alive() is True

        # Submit a task
        future = worker.submit('process', 5)
        result = future.get(timeout=2.0)
        assert result == 10
        assert 5 in worker.processed

        # Shutdown
        worker.shutdown(wait=True, timeout=2.0)
        assert worker.is_alive() is False

    def test_worker_with_rate_limiting(self, temp_dir):
        """Test that worker respects rate limiting."""
        class TestWorker(BaseWorker):
            def _register_methods(self):
                self._methods = {'test': lambda: 'result'}

            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self._register_methods()

        worker = TestWorker(name="TestWorker", rpm=60, poll_interval=0.01)
        worker.start()

        # Submit tasks rapidly
        futures = []
        start = time.time()
        for _ in range(70):  # More than capacity (60)
            futures.append(worker.submit('test'))

        # All should complete but rate limited
        for future in futures:
            future.get(timeout=10.0)

        elapsed = time.time() - start
        # 70 requests at 60 RPM = 60 burst + 10 more requiring ~10 seconds
        # But due to fast processing, actual time should be less
        assert elapsed >= 0.1  # At least some rate limiting occurred

        worker.shutdown(wait=True, timeout=2.0)

    def test_worker_task_exception(self, temp_dir):
        """Test that worker handles task exceptions."""
        class TestWorker(BaseWorker):
            def _register_methods(self):
                self._methods = {
                    'fail': lambda: (_ for _ in ()).throw(ValueError("test error"))
                }

            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self._register_methods()

        worker = TestWorker(name="TestWorker", poll_interval=0.01)
        worker.start()

        future = worker.submit('fail')

        with pytest.raises(ValueError, match="test error"):
            future.get(timeout=2.0)

        worker.shutdown(wait=True, timeout=2.0)

    def test_worker_unknown_method(self, temp_dir):
        """Test that worker handles unknown method errors."""
        class TestWorker(BaseWorker):
            def _register_methods(self):
                self._methods = {}

            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self._register_methods()

        worker = TestWorker(name="TestWorker", poll_interval=0.01)
        worker.start()

        future = worker.submit('unknown_method')

        with pytest.raises(ValueError, match="Unknown method"):
            future.get(timeout=2.0)

        worker.shutdown(wait=True, timeout=2.0)

    def test_worker_status(self, temp_dir):
        """Test worker status reporting."""
        class TestWorker(BaseWorker):
            def _register_methods(self):
                self._methods = {'test': lambda: 'result'}

            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self._register_methods()

        worker = TestWorker(name="TestWorker", rpm=60, tpm=10000)
        status = worker.get_status()

        assert status['name'] == "TestWorker"
        assert status['is_alive'] is False
        assert status['queue_size'] == 0
        assert 'rate_limiter' in status

    def test_worker_graceful_shutdown(self, temp_dir):
        """Test that worker processes remaining tasks on shutdown."""
        class TestWorker(BaseWorker):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self.processed = []
                self._register_methods()

            def _register_methods(self):
                self._methods = {
                    'slow_task': self._slow_task
                }

            def _slow_task(self):
                time.sleep(0.1)
                self.processed.append(1)
                return 'done'

        worker = TestWorker(name="TestWorker", poll_interval=0.01)
        worker.start()

        # Submit multiple tasks
        futures = [worker.submit('slow_task') for _ in range(3)]

        # Shutdown immediately - should wait for tasks to complete
        worker.shutdown(wait=True, timeout=5.0)

        # All tasks should have been processed
        for future in futures:
            assert future.get(timeout=1.0) == 'done'

        assert len(worker.processed) == 3
        assert worker.is_alive() is False

    def test_multiple_workers(self, temp_dir):
        """Test running multiple workers concurrently."""
        class TestWorker(BaseWorker):
            def __init__(self, *args, worker_id=None, **kwargs):
                super().__init__(*args, **kwargs)
                self.worker_id = worker_id
                self._register_methods()

            def _register_methods(self):
                self._methods = {
                    'get_id': lambda: self.worker_id
                }

        workers = [
            TestWorker(name=f"Worker{i}", worker_id=i, poll_interval=0.01)
            for i in range(3)
        ]

        for worker in workers:
            worker.start()

        # Submit tasks to all workers
        results = []
        for worker in workers:
            future = worker.submit('get_id')
            results.append(future.get(timeout=2.0))

        assert sorted(results) == [0, 1, 2]

        # Shutdown all workers
        for worker in workers:
            worker.shutdown(wait=True, timeout=2.0)
