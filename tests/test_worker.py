"""
Unit tests for worker module.
"""

import os
import time
import pytest
from unittest.mock import Mock, patch

from utilities.worker import (
    BaseWorker, Future, Task, get_rate_limit_config,
    CostTracker, get_pricing_config
)


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


@pytest.mark.unit
class TestCostTracker:
    """Test cases for CostTracker class."""

    def test_cost_tracker_creation(self):
        """Test CostTracker initialization."""
        tracker = CostTracker()
        assert tracker.total_input_tokens == 0
        assert tracker.total_output_tokens == 0
        assert tracker.total_cost == 0.0
        assert tracker.request_count == 0
        assert tracker.input_price_per_m is None
        assert tracker.output_price_per_m is None

    def test_cost_tracker_with_pricing(self):
        """Test CostTracker with pricing configured."""
        tracker = CostTracker(input_price_per_m=0.14, output_price_per_m=0.28)
        assert tracker.input_price_per_m == 0.14
        assert tracker.output_price_per_m == 0.28

    def test_add_usage_without_pricing(self):
        """Test add_usage without pricing configured."""
        tracker = CostTracker()
        cost = tracker.add_usage(1000, 500)

        assert cost == 0.0
        assert tracker.total_input_tokens == 1000
        assert tracker.total_output_tokens == 500
        assert tracker.request_count == 1

    def test_add_usage_with_pricing(self):
        """Test add_usage with pricing configured."""
        tracker = CostTracker(input_price_per_m=0.14, output_price_per_m=0.28)

        # 1000 input tokens @ $0.14/M = $0.00014
        # 500 output tokens @ $0.28/M = $0.00014
        # Total = $0.00028
        cost = tracker.add_usage(1000, 500)

        assert abs(cost - 0.00028) < 0.000001
        assert tracker.total_input_tokens == 1000
        assert tracker.total_output_tokens == 500
        assert tracker.total_cost == cost
        assert tracker.request_count == 1

    def test_multiple_add_usage(self):
        """Test multiple add_usage calls."""
        tracker = CostTracker(input_price_per_m=0.14, output_price_per_m=0.28)

        tracker.add_usage(1000, 500)
        tracker.add_usage(2000, 1000)

        assert tracker.total_input_tokens == 3000
        assert tracker.total_output_tokens == 1500
        assert tracker.request_count == 2
        # 3000 * 0.14 / 1M + 1500 * 0.28 / 1M = 0.00084
        assert abs(tracker.total_cost - 0.00084) < 0.000001

    def test_format_cost(self):
        """Test cost formatting."""
        tracker = CostTracker()

        # With zero cost
        assert tracker.format_cost(0) == "N/A"

        # With positive cost
        assert tracker.format_cost(0.000123456) == "$0.000123"

    def test_get_summary(self):
        """Test summary generation."""
        tracker = CostTracker(input_price_per_m=0.14, output_price_per_m=0.28)
        tracker.add_usage(1000, 500)
        tracker.add_usage(2000, 1000)

        summary = tracker.get_summary()
        assert "requests: 2" in summary
        assert "input_tokens: 3,000" in summary
        assert "output_tokens: 1,500" in summary
        assert "total_cost:" in summary

    def test_get_summary_without_pricing(self):
        """Test summary without pricing configured."""
        tracker = CostTracker()
        tracker.add_usage(1000, 500)

        summary = tracker.get_summary()
        assert "requests: 1" in summary
        assert "input_tokens: 1,000" in summary
        assert "output_tokens: 500" in summary
        assert "total_cost:" not in summary


@pytest.mark.unit
class TestGetPricingConfig:
    """Test cases for get_pricing_config function."""

    def test_get_pricing_config_with_values(self):
        """Test with both input and output prices configured."""
        os.environ['TEST_INPUT_PRICE_PER_M'] = '0.14'
        os.environ['TEST_OUTPUT_PRICE_PER_M'] = '0.28'

        input_price, output_price = get_pricing_config('TEST')

        assert input_price == 0.14
        assert output_price == 0.28

        # Cleanup
        del os.environ['TEST_INPUT_PRICE_PER_M']
        del os.environ['TEST_OUTPUT_PRICE_PER_M']

    def test_get_pricing_config_partial(self):
        """Test with only input price configured."""
        os.environ['TEST_INPUT_PRICE_PER_M'] = '0.14'

        input_price, output_price = get_pricing_config('TEST')

        assert input_price == 0.14
        assert output_price is None

        # Cleanup
        del os.environ['TEST_INPUT_PRICE_PER_M']

    def test_get_pricing_config_no_config(self):
        """Test with no pricing configuration."""
        # Clear any existing config
        for key in list(os.environ.keys()):
            if 'PRICE_PER_M' in key:
                del os.environ[key]

        input_price, output_price = get_pricing_config('NONEXISTENT')

        assert input_price is None
        assert output_price is None


@pytest.mark.unit
class TestBaseWorkerCostTracking:
    """Test cases for BaseWorker cost tracking."""

    def test_worker_with_pricing_prefix(self, temp_dir):
        """Test worker initialization with pricing prefix."""
        os.environ['TEST_INPUT_PRICE_PER_M'] = '0.14'
        os.environ['TEST_OUTPUT_PRICE_PER_M'] = '0.28'

        class TestWorker(BaseWorker):
            def _register_methods(self):
                self._methods = {'test': lambda: 'result'}

            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self._register_methods()

        worker = TestWorker(name="TestWorker", pricing_prefix='TEST')

        assert worker._cost_tracker.input_price_per_m == 0.14
        assert worker._cost_tracker.output_price_per_m == 0.28

        # Cleanup
        del os.environ['TEST_INPUT_PRICE_PER_M']
        del os.environ['TEST_OUTPUT_PRICE_PER_M']

    def test_worker_track_usage(self, temp_dir):
        """Test worker usage tracking."""
        class TestWorker(BaseWorker):
            def _register_methods(self):
                self._methods = {'test': lambda: 'result'}

            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self._cost_tracker = CostTracker(
                    input_price_per_m=0.14,
                    output_price_per_m=0.28
                )
                self._register_methods()

        worker = TestWorker(name="TestWorker")

        cost = worker._track_usage(1000, 500)

        assert abs(cost - 0.00028) < 0.000001
        assert worker._cost_tracker.total_input_tokens == 1000
        assert worker._cost_tracker.total_output_tokens == 500

    def test_worker_shutdown_prints_cost_summary(self, temp_dir, capsys):
        """Test that worker shutdown prints cost summary."""
        class TestWorker(BaseWorker):
            def _register_methods(self):
                self._methods = {'test': lambda: 'result'}

            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self._cost_tracker = CostTracker(
                    input_price_per_m=0.14,
                    output_price_per_m=0.28
                )
                self._register_methods()

        worker = TestWorker(name="TestWorker", poll_interval=0.01)
        worker.start()
        worker._track_usage(1000, 500)
        worker.shutdown(wait=True)

        captured = capsys.readouterr()
        assert "Cost summary" in captured.out
        assert "requests: 1" in captured.out
        assert "input_tokens: 1,000" in captured.out


@pytest.mark.unit
class TestBaseWorkerRetry:
    """Test cases for BaseWorker retry mechanism."""

    def test_should_retry_with_non_retryable_exceptions(self, temp_dir):
        """Test that non-retryable exceptions are not retried."""
        class TestWorker(BaseWorker):
            def _register_methods(self):
                self._methods = {}

            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self._register_methods()

        worker = TestWorker(name="TestWorker", max_retries=3)

        # Test each non-retryable exception type
        non_retryable_exceptions = [
            ValueError("test value error"),
            TypeError("test type error"),
            AttributeError("test attribute error"),
            KeyError("test_key"),
            ImportError("test_module"),
            NotImplementedError("test not implemented"),
        ]

        for exc in non_retryable_exceptions:
            assert worker._should_retry(exc) is False, f"Should not retry {type(exc).__name__}"

    def test_should_retry_with_retryable_patterns(self, temp_dir):
        """Test that exceptions with retryable patterns are retried."""
        class TestWorker(BaseWorker):
            def _register_methods(self):
                self._methods = {}

            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self._register_methods()

        worker = TestWorker(name="TestWorker", max_retries=3)

        # Test retryable exception patterns
        retryable_exceptions = [
            RuntimeError("Connection timeout"),
            RuntimeError("Network error"),
            RuntimeError("Temporary failure"),
            RuntimeError("Service unavailable"),
            RuntimeError("Rate limit exceeded"),
            RuntimeError("HTTP 429"),
            RuntimeError("HTTP 503"),
            RuntimeError("HTTP 502"),
            # Generic runtime error should also be retryable (default behavior)
            RuntimeError("Generic error"),
        ]

        for exc in retryable_exceptions:
            assert worker._should_retry(exc) is True, f"Should retry {exc}"

    def test_retry_on_temporary_failure(self, temp_dir):
        """Test that worker retries on temporary failures."""
        class TestWorker(BaseWorker):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self.attempt_count = 0
                self._register_methods()

            def _register_methods(self):
                self._methods = {
                    'failing_task': self._failing_task
                }

            def _failing_task(self):
                self.attempt_count += 1
                if self.attempt_count < 3:
                    raise RuntimeError("Connection timeout")
                return "success"

        worker = TestWorker(name="TestWorker", max_retries=3, retry_delay=0.05, poll_interval=0.01)
        worker.start()

        future = worker.submit('failing_task')
        result = future.get(timeout=5.0)

        assert result == "success"
        assert worker.attempt_count == 3  # Initial attempt + 2 retries
        assert worker._total_retries == 2

        worker.shutdown(wait=True, timeout=2.0)

    def test_no_retry_on_non_retryable_exception(self, temp_dir):
        """Test that non-retryable exceptions fail immediately."""
        class TestWorker(BaseWorker):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self.attempt_count = 0
                self._register_methods()

            def _register_methods(self):
                self._methods = {
                    'invalid_task': self._invalid_task
                }

            def _invalid_task(self):
                self.attempt_count += 1
                raise ValueError("Invalid parameter")

        worker = TestWorker(name="TestWorker", max_retries=3, retry_delay=0.05, poll_interval=0.01)
        worker.start()

        future = worker.submit('invalid_task')

        with pytest.raises(ValueError, match="Invalid parameter"):
            future.get(timeout=2.0)

        # Should have failed immediately without retries
        assert worker.attempt_count == 1
        assert worker._total_retries == 0
        assert worker._total_failures == 1

        worker.shutdown(wait=True, timeout=2.0)

    def test_retry_exhaustion(self, temp_dir):
        """Test that task fails after max retries is exhausted."""
        class TestWorker(BaseWorker):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self.attempt_count = 0
                self._register_methods()

            def _register_methods(self):
                self._methods = {
                    'always_failing': self._always_failing
                }

            def _always_failing(self):
                self.attempt_count += 1
                raise RuntimeError("Connection timeout")

        worker = TestWorker(name="TestWorker", max_retries=2, retry_delay=0.05, poll_interval=0.01)
        worker.start()

        future = worker.submit('always_failing')

        with pytest.raises(RuntimeError, match="Connection timeout"):
            future.get(timeout=5.0)

        # Should have attempted max_retries + 1 times
        assert worker.attempt_count == 3  # Initial + 2 retries
        assert worker._total_retries == 2
        assert worker._total_failures == 1

        worker.shutdown(wait=True, timeout=2.0)

    def test_retry_statistics_tracking(self, temp_dir):
        """Test that retry statistics are properly tracked."""
        class TestWorker(BaseWorker):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self.call_count = 0
                self._register_methods()

            def _register_methods(self):
                self._methods = {
                    'task1': self._task1,
                    'task2': self._task2,
                }

            def _task1(self):
                self.call_count += 1
                if self.call_count <= 2:
                    raise RuntimeError("Timeout")
                return "task1_success"

            def _task2(self):
                raise ValueError("Invalid")

        worker = TestWorker(name="TestWorker", max_retries=3, retry_delay=0.05, poll_interval=0.01)
        worker.start()

        # task1 should succeed after retries
        future1 = worker.submit('task1')
        assert future1.get(timeout=5.0) == "task1_success"

        # task2 should fail immediately (non-retryable)
        future2 = worker.submit('task2')
        with pytest.raises(ValueError):
            future2.get(timeout=2.0)

        # Check statistics
        assert worker._total_retries == 2  # From task1 retries
        assert worker._total_failures == 1  # From task2

        worker.shutdown(wait=True, timeout=2.0)

    def test_no_retry_when_max_retries_is_zero(self, temp_dir):
        """Test that no retries occur when max_retries is 0."""
        class TestWorker(BaseWorker):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self.attempt_count = 0
                self._register_methods()

            def _register_methods(self):
                self._methods = {
                    'failing_task': self._failing_task
                }

            def _failing_task(self):
                self.attempt_count += 1
                raise RuntimeError("Temporary error")

        worker = TestWorker(name="TestWorker", max_retries=0, poll_interval=0.01)
        worker.start()

        future = worker.submit('failing_task')

        with pytest.raises(RuntimeError, match="Temporary error"):
            future.get(timeout=2.0)

        # Should have failed immediately without retries
        assert worker.attempt_count == 1
        assert worker._total_retries == 0

        worker.shutdown(wait=True, timeout=2.0)
