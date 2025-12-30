"""
Base Worker class for async task processing with rate limiting.
"""

import os
import threading
import time
import uuid
from queue import Queue, Empty
from typing import Any, Callable, Optional, Dict
from dataclasses import dataclass

from .rate_limiter import RateLimiter


@dataclass
class Task:
    """A task to be processed by a worker."""
    id: str
    method: str
    args: tuple
    kwargs: dict
    future: 'Future'


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


class BaseWorker(threading.Thread):
    """
    Base worker class for async task processing.

    Workers run as daemon threads that process tasks from a queue.
    They support rate limiting and graceful shutdown.
    """

    def __init__(self,
                 name: str,
                 rpm: Optional[int] = None,
                 tpm: Optional[int] = None,
                 poll_interval: float = 0.1):
        """
        Initialize the worker.

        Args:
            name: Worker name for logging
            rpm: Requests per minute limit (None = no limit)
            tpm: Tokens per minute limit (None = no limit)
            poll_interval: Time to sleep when queue is empty (seconds)
        """
        super().__init__(daemon=True, name=name)
        self._queue: Queue[Task] = Queue()
        self._rate_limiter = RateLimiter(rpm=rpm, tpm=tpm)
        self._poll_interval = poll_interval
        self._shutdown_event = threading.Event()
        self._methods: Dict[str, Callable] = {}
        # Note: _register_methods should be called by subclasses after their initialization

    def _register_methods(self) -> None:
        """Register methods that can be called by tasks."""
        # Subclasses should override this to register their methods
        # Default implementation does nothing (not raise error)
        pass

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
            future=future
        )
        self._queue.put(task)
        return future

    def run(self) -> None:
        """Main worker loop."""
        print(f"[{self.name}] Worker started")

        while not self._shutdown_event.is_set():
            try:
                task = self._queue.get(timeout=self._poll_interval)
            except Empty:
                continue

            # Apply rate limiting
            # Estimate token count based on method (can be overridden)
            tokens = self._estimate_tokens(task)
            self._rate_limiter.acquire(tokens=tokens, block=True)

            # Process the task
            try:
                if task.method not in self._methods:
                    raise ValueError(f"Unknown method: {task.method}")

                method = self._methods[task.method]
                result = method(*task.args, **task.kwargs)
                task.future.set_result(result)

            except Exception as e:
                task.future.set_exception(e)

        # Process remaining tasks in queue before shutdown (graceful shutdown)
        while True:
            try:
                task = self._queue.get_nowait()
            except Empty:
                break

            # Apply rate limiting
            tokens = self._estimate_tokens(task)
            self._rate_limiter.acquire(tokens=tokens, block=True)

            # Process the task
            try:
                if task.method not in self._methods:
                    raise ValueError(f"Unknown method: {task.method}")

                method = self._methods[task.method]
                result = method(*task.args, **task.kwargs)
                task.future.set_result(result)

            except Exception as e:
                task.future.set_exception(e)

        print(f"[{self.name}] Worker stopped")

    def _estimate_tokens(self, task: Task) -> int:
        """
        Estimate the number of tokens a task will consume.

        Subclasses can override this for more accurate estimation.
        """
        return 1

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

    def get_status(self) -> dict:
        """Get the current status of the worker."""
        return {
            'name': self.name,
            'is_alive': self.is_alive(),
            'queue_size': self._queue.qsize(),
            'rate_limiter': self._rate_limiter.get_status(),
        }


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
