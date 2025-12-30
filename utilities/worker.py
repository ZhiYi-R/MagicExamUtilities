"""
Base Worker class for async task processing with rate limiting.
"""

import os
import threading
import time
import uuid
from queue import Queue, Empty
from typing import Any, Callable, Optional, Dict, Tuple, Type
from dataclasses import dataclass, field
from functools import wraps
import concurrent.futures

from .rate_limiter import RateLimiter


@dataclass
class CostTracker:
    """Track API costs for a worker."""
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    total_cost: float = 0.0
    request_count: int = 0

    # Pricing in USD per 1M tokens (None = not configured)
    input_price_per_m: Optional[float] = None
    output_price_per_m: Optional[float] = None

    def add_usage(self, input_tokens: int, output_tokens: int) -> float:
        """
        Add token usage and return the cost for this request.

        Args:
            input_tokens: Number of input tokens
            output_tokens: Number of output tokens

        Returns:
            Cost in USD for this request (0 if pricing not configured)
        """
        self.total_input_tokens += input_tokens
        self.total_output_tokens += output_tokens
        self.request_count += 1

        # Calculate cost if pricing is configured
        if self.input_price_per_m is not None and self.output_price_per_m is not None:
            input_cost = (input_tokens / 1_000_000) * self.input_price_per_m
            output_cost = (output_tokens / 1_000_000) * self.output_price_per_m
            request_cost = input_cost + output_cost
            self.total_cost += request_cost
            return request_cost

        return 0.0

    def format_cost(self, cost: float) -> str:
        """Format cost for display."""
        return f"${cost:.6f}" if cost > 0 else "N/A"

    def get_summary(self) -> str:
        """Get a summary of usage and costs."""
        parts = [
            f"requests: {self.request_count}",
            f"input_tokens: {self.total_input_tokens:,}",
            f"output_tokens: {self.total_output_tokens:,}",
        ]
        if self.total_cost > 0:
            parts.append(f"total_cost: {self.format_cost(self.total_cost)}")
        return ", ".join(parts)


def get_pricing_config(prefix: str) -> Tuple[Optional[float], Optional[float]]:
    """
    Get pricing configuration from environment variables.

    Args:
        prefix: Environment variable prefix (e.g., 'OCR', 'ASR', 'SUMMARIZATION')

    Returns:
        Tuple of (input_price_per_m, output_price_per_m) in USD
    """
    input_price = os.environ.get(f'{prefix}_INPUT_PRICE_PER_M')
    output_price = os.environ.get(f'{prefix}_OUTPUT_PRICE_PER_M')

    input_price_per_m = float(input_price) if input_price else None
    output_price_per_m = float(output_price) if output_price else None

    return input_price_per_m, output_price_per_m


@dataclass
class Task:
    """A task to be processed by a worker."""
    id: str
    method: str
    args: tuple
    kwargs: dict
    future: 'Future'
    timeout: Optional[float] = None  # Timeout for this specific task (None = use worker default)


class Future:
    """Future object for getting task results."""

    def __init__(self, task_id: str):
        self._task_id = task_id
        self._event = threading.Event()
        self._result: Any = None
        self._exception: Optional[Exception] = None

    def set_result(self, result: Any) -> None:
        """Set the result of the task."""
        self._result = result
        self._event.set()

    def set_exception(self, exception: Exception) -> None:
        """Set an exception for the task."""
        self._exception = exception
        self._event.set()

    def get(self, timeout: Optional[float] = None) -> Any:
        """
        Get the result of the task.

        Args:
            timeout: Maximum time to wait in seconds (None = wait forever)

        Returns:
            The result of the task

        Raises:
            TimeoutError: If timeout occurs
            Exception: The exception raised during task execution
        """
        if not self._event.wait(timeout=timeout):
            raise TimeoutError(f"Task {self._task_id} timed out")

        if self._exception is not None:
            raise self._exception

        return self._result

    def is_done(self) -> bool:
        """Check if the task is completed."""
        return self._event.is_set()


class WorkerTimeoutError(TimeoutError):
    """Timeout error raised by worker task methods."""
    pass


def with_timeout(timeout_seconds: Optional[float]):
    """
    Decorator to add timeout handling to worker task methods.

    This uses a thread pool to execute the function with a timeout.
    If the function times out, a WorkerTimeoutError is raised which
    can be caught and retried by the worker's retry mechanism.

    Args:
        timeout_seconds: Timeout in seconds (None = no timeout)

    Returns:
        Decorated function
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            if timeout_seconds is None:
                # No timeout, execute directly
                return func(*args, **kwargs)

            # Use ThreadPoolExecutor to enforce timeout
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(func, *args, **kwargs)
                try:
                    return future.result(timeout=timeout_seconds)
                except concurrent.futures.TimeoutError:
                    # Cancel the future if still running
                    future.cancel()
                    raise WorkerTimeoutError(
                        f"Task {func.__name__} timed out after {timeout_seconds} seconds"
                    )

        return wrapper
    return decorator


class BaseWorker(threading.Thread):
    """
    Base worker class for async task processing.

    Workers run as daemon threads that process tasks from a queue.
    They support rate limiting, graceful shutdown, and automatic retry.
    """

    # Exceptions that should not be retried
    NON_RETRYABLE_EXCEPTIONS: Tuple[Type[Exception], ...] = (
        ValueError,
        TypeError,
        AttributeError,
        KeyError,
        ImportError,
        NotImplementedError,
    )

    def __init__(self,
                 name: str,
                 rpm: Optional[int] = None,
                 tpm: Optional[int] = None,
                 poll_interval: float = 0.1,
                 pricing_prefix: Optional[str] = None,
                 max_retries: int = 0,
                 retry_delay: float = 1.0,
                 task_timeout: Optional[float] = None):
        """
        Initialize the worker.

        Args:
            name: Worker name for logging
            rpm: Requests per minute limit (None = no limit)
            tpm: Tokens per minute limit (None = no limit)
            poll_interval: Time to sleep when queue is empty (seconds)
            pricing_prefix: Prefix for pricing env vars (e.g., 'OCR', 'SUMMARIZATION')
            max_retries: Maximum number of retries for failed tasks (0 = no retry)
            retry_delay: Delay between retries in seconds
            task_timeout: Timeout for individual task execution in seconds (None = no timeout)
        """
        super().__init__(daemon=True, name=name)
        self._queue: Queue[Task] = Queue()
        self._rate_limiter = RateLimiter(rpm=rpm, tpm=tpm)
        self._poll_interval = poll_interval
        self._shutdown_event = threading.Event()
        self._methods: Dict[str, Callable] = {}
        self._max_retries = max_retries
        self._retry_delay = retry_delay
        self._task_timeout = task_timeout

        # Statistics for retries
        self._total_retries: int = 0
        self._total_failures: int = 0

        # Initialize cost tracker
        self._cost_tracker = CostTracker()
        if pricing_prefix:
            input_price, output_price = get_pricing_config(pricing_prefix)
            self._cost_tracker.input_price_per_m = input_price
            self._cost_tracker.output_price_per_m = output_price
            if input_price is not None or output_price is not None:
                print(f"[{name}] Pricing configured: input=${input_price}/M, output=${output_price}/M")

        # Log retry configuration
        if max_retries > 0:
            print(f"[{name}] Retry configured: max_retries={max_retries}, retry_delay={retry_delay}s")

        # Note: _register_methods should be called by subclasses after their initialization

    def _register_methods(self) -> None:
        """Register methods that can be called by tasks."""
        # Subclasses should override this to register their methods
        # Default implementation does nothing (not raise error)
        pass

    def _should_retry(self, exception: Exception) -> bool:
        """
        Determine if an exception should trigger a retry.

        Args:
            exception: The exception that was raised

        Returns:
            True if the exception is retryable, False otherwise
        """
        # Check if it's a non-retryable exception type
        for exc_type in self.NON_RETRYABLE_EXCEPTIONS:
            if isinstance(exception, exc_type):
                return False

        # Check for specific exception patterns
        exc_str = str(exception).lower()

        # Network/timeout errors are retryable
        retryable_patterns = [
            'timeout',
            'connection',
            'network',
            'temporary',
            'unavailable',
            'rate limit',
            '429',  # HTTP 429 Too Many Requests
            '503',  # HTTP 503 Service Unavailable
            '502',  # HTTP 502 Bad Gateway
        ]

        for pattern in retryable_patterns:
            if pattern in exc_str:
                return True

        # Default to retry for unknown exceptions (can be overridden)
        return True

    def submit(self, method: str, *args, **kwargs) -> Future:
        """
        Submit a task to the worker.

        Args:
            method: Name of the method to call
            *args: Positional arguments for the method
            **kwargs: Keyword arguments for the method

        Returns:
            A Future object for getting the result
        """
        task_id = str(uuid.uuid4())
        future = Future(task_id)
        task = Task(
            id=task_id,
            method=method,
            args=args,
            kwargs=kwargs,
            future=future,
            timeout=kwargs.pop('_task_timeout', None)  # Extract timeout from kwargs
        )
        self._queue.put(task)
        return future

    def _execute_task_with_timeout(self, task: Task, timeout: Optional[float]) -> Any:
        """
        Execute a task with optional timeout.

        Args:
            task: The task to execute
            timeout: Optional timeout in seconds (None = use worker default)

        Returns:
            The result of the task execution

        Raises:
            WorkerTimeoutError: If the task times out
        """
        if task.method not in self._methods:
            raise ValueError(f"Unknown method: {task.method}")

        method = self._methods[task.method]

        # Use provided timeout or fall back to worker default
        effective_timeout = timeout if timeout is not None else self._task_timeout

        if effective_timeout is not None:
            return self._execute_with_timeout(method, task.args, task.kwargs, effective_timeout)
        else:
            return method(*task.args, **task.kwargs)

    def _execute_task_with_retry(self, task: Task) -> None:
        """
        Execute a task with retry logic.

        Args:
            task: The task to execute
        """
        last_exception = None

        for attempt in range(self._max_retries + 1):
            try:
                # Execute with timeout (uses task.timeout or worker default)
                result = self._execute_task_with_timeout(task, task.timeout)

                # Success!
                if attempt > 0:
                    print(f"[{self.name}] Task {task.id[:8]} succeeded after {attempt} retries")
                task.future.set_result(result)
                return

            except Exception as e:
                last_exception = e
                should_retry = self._should_retry(e) and attempt < self._max_retries

                if should_retry:
                    self._total_retries += 1
                    print(f"[{self.name}] Task {task.id[:8]} failed (attempt {attempt + 1}/{self._max_retries + 1}): {e}, retrying in {self._retry_delay}s...")
                    time.sleep(self._retry_delay)
                else:
                    # No more retries or non-retryable exception
                    if attempt > 0:
                        print(f"[{self.name}] Task {task.id[:8]} failed after {attempt + 1} attempts: {e}")
                    self._total_failures += 1
                    task.future.set_exception(e)
                    return

    def _execute_with_timeout(self, method: Callable, args: tuple, kwargs: dict, timeout: float) -> Any:
        """
        Execute a method with timeout.

        Note: The timeout is passed to the method via _timeout kwarg.
        Individual methods are responsible for enforcing the timeout
        (e.g., by passing it to underlying API calls).

        Args:
            method: The method to execute
            args: Positional arguments
            kwargs: Keyword arguments
            timeout: Timeout in seconds

        Returns:
            The result of the method call

        Raises:
            Exception: Any exception raised by the method (including WorkerTimeoutError)
        """
        # Pass timeout to the method via kwargs
        # The method is responsible for enforcing the timeout
        kwargs_with_timeout = {**kwargs, '_timeout': timeout}
        return method(*args, **kwargs_with_timeout)

    def run(self) -> None:
        """Main worker loop."""
        print(f"[{self.name}] Worker started")

        while not self._shutdown_event.is_set():
            try:
                task = self._queue.get(timeout=self._poll_interval)
            except Empty:
                continue

            # Apply rate limiting
            tokens = self._estimate_tokens(task)
            self._rate_limiter.acquire(tokens=tokens, block=True)

            # Process the task with retry
            self._execute_task_with_retry(task)

        # Process remaining tasks in queue before shutdown (graceful shutdown)
        while True:
            try:
                task = self._queue.get_nowait()
            except Empty:
                break

            # Apply rate limiting
            tokens = self._estimate_tokens(task)
            self._rate_limiter.acquire(tokens=tokens, block=True)

            # Process the task with retry
            self._execute_task_with_retry(task)

        print(f"[{self.name}] Worker stopped")

    def _estimate_tokens(self, task: Task) -> int:
        """
        Estimate the number of tokens a task will consume.

        Subclasses can override this for more accurate estimation.
        """
        return 1

    def _track_usage(self, input_tokens: int, output_tokens: int) -> float:
        """
        Track token usage and return the cost for this request.

        Args:
            input_tokens: Number of input tokens consumed
            output_tokens: Number of output tokens consumed

        Returns:
            Cost in USD for this request (0 if pricing not configured)
        """
        return self._cost_tracker.add_usage(input_tokens, output_tokens)

    def shutdown(self, wait: bool = True, timeout: Optional[float] = None) -> None:
        """
        Shutdown the worker gracefully.

        Args:
            wait: If True, wait for the worker to finish
            timeout: Maximum time to wait in seconds
        """
        self._shutdown_event.set()

        if wait:
            self.join(timeout=timeout)

        # Print cost summary on shutdown
        if self._cost_tracker.request_count > 0:
            print(f"[{self.name}] Cost summary: {self._cost_tracker.get_summary()}")

        # Print retry statistics
        if self._max_retries > 0 and (self._total_retries > 0 or self._total_failures > 0):
            print(f"[{self.name}] Retry statistics: total_retries={self._total_retries}, total_failures={self._total_failures}")

    def get_status(self) -> dict:
        """Get the current status of the worker."""
        return {
            'name': self.name,
            'is_alive': self.is_alive(),
            'queue_size': self._queue.qsize(),
            'rate_limiter': self._rate_limiter.get_status(),
        }

    def get_cost_tracker(self) -> CostTracker:
        """Get the current cost tracker state."""
        return self._cost_tracker

    def get_cost_summary(self) -> str:
        """Get a formatted summary of current costs."""
        return self._cost_tracker.get_summary()

    def get_model_name(self) -> str:
        """
        Get the model name used by this worker.

        Subclasses should override this to return their specific model name.

        Returns:
            Model name or empty string if not configured
        """
        return ""


def get_rate_limit_config(prefix: str) -> tuple:
    """
    Get rate limit configuration from environment variables.

    Args:
        prefix: Environment variable prefix (e.g., 'OCR', 'ASR', 'SUMMARIZATION')

    Returns:
        Tuple of (rpm, tpm) - (None, None) if not configured
    """
    rpm = os.environ.get(f'{prefix}_RPM')
    tpm = os.environ.get(f'{prefix}_TPM')

    rpm = int(rpm) if rpm else None
    tpm = int(tpm) if tpm else None

    return rpm, tpm


def get_retry_config(prefix: str) -> Tuple[int, float]:
    """
    Get retry configuration from environment variables.

    Args:
        prefix: Environment variable prefix (e.g., 'OCR', 'ASR', 'SUMMARIZATION')

    Returns:
        Tuple of (max_retries, retry_delay) - (0, 1.0) if not configured
    """
    max_retries = os.environ.get(f'{prefix}_MAX_RETRIES')
    retry_delay = os.environ.get(f'{prefix}_RETRY_DELAY')

    max_retries = int(max_retries) if max_retries else 0
    retry_delay = float(retry_delay) if retry_delay else 1.0

    return max_retries, retry_delay
