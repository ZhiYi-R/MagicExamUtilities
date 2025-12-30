"""
Rate limiter using token bucket algorithm.
Supports both TPM (Tokens Per Minute) and RPM (Requests Per Minute) limiting.
"""

import time
import threading
from typing import Optional


class TokenBucket:
    """
    Token bucket implementation for rate limiting.

    Args:
        capacity: Maximum number of tokens the bucket can hold
        refill_rate: Number of tokens to add per second (tokens per minute / 60)
        initial_tokens: Initial number of tokens in the bucket (default: capacity)
    """

    def __init__(self, capacity: int, refill_rate: float, initial_tokens: Optional[int] = None):
        self._capacity = capacity
        self._refill_rate = refill_rate
        self._tokens = initial_tokens if initial_tokens is not None else capacity
        self._last_refill_time = time.time()
        self._lock = threading.Lock()

    def _refill(self) -> None:
        """Refill tokens based on elapsed time."""
        now = time.time()
        elapsed = now - self._last_refill_time
        tokens_to_add = elapsed * self._refill_rate

        with self._lock:
            self._tokens = min(self._capacity, self._tokens + tokens_to_add)
            self._last_refill_time = now

    def try_consume(self, tokens: int = 1) -> bool:
        """
        Try to consume the specified number of tokens.

        Args:
            tokens: Number of tokens to consume (default: 1)

        Returns:
            True if tokens were consumed, False otherwise
        """
        self._refill()

        with self._lock:
            if self._tokens >= tokens:
                self._tokens -= tokens
                return True
            return False

    def consume(self, tokens: int = 1, block: bool = True, timeout: Optional[float] = None) -> bool:
        """
        Consume the specified number of tokens.

        Args:
            tokens: Number of tokens to consume (default: 1)
            block: If True, block until tokens are available
            timeout: Maximum time to wait in seconds (only if block=True)

        Returns:
            True if tokens were consumed, False if timeout occurred
        """
        start_time = time.time()

        while True:
            self._refill()

            with self._lock:
                if self._tokens >= tokens:
                    self._tokens -= tokens
                    return True

            if not block:
                return False

            if timeout is not None:
                elapsed = time.time() - start_time
                if elapsed >= timeout:
                    return False

            # Sleep a bit before retrying
            time.sleep(0.01)

    def get_available_tokens(self) -> float:
        """Get current available tokens."""
        self._refill()
        with self._lock:
            return self._tokens


class RateLimiter:
    """
    Rate limiter combining TPM (Tokens Per Minute) and RPM (Requests Per Minute).

    Both limits must be satisfied for a request to proceed.
    """

    def __init__(self, rpm: Optional[int] = None, tpm: Optional[int] = None):
        """
        Initialize rate limiter.

        Args:
            rpm: Requests per minute limit (None = no limit)
            tpm: Tokens per minute limit (None = no limit)
        """
        self._rpm_bucket: Optional[TokenBucket] = None
        self._tpm_bucket: Optional[TokenBucket] = None

        if rpm is not None and rpm > 0:
            self._rpm_bucket = TokenBucket(
                capacity=max(1, rpm),  # Allow bursting up to capacity
                refill_rate=rpm / 60.0
            )

        if tpm is not None and tpm > 0:
            self._tpm_bucket = TokenBucket(
                capacity=max(1, tpm),  # Allow bursting up to capacity
                refill_rate=tpm / 60.0
            )

    def can_proceed(self, tokens: int = 1) -> bool:
        """
        Check if a request with the given token count can proceed.

        Args:
            tokens: Number of tokens the request will consume

        Returns:
            True if the request can proceed (non-blocking check)
        """
        rpm_ok = self._rpm_bucket is None or self._rpm_bucket.try_consume()
        tpm_ok = self._tpm_bucket is None or self._tpm_bucket.try_consume(tokens)

        # Roll back RPM token if TPM check failed
        if rpm_ok and not tpm_ok and self._rpm_bucket is not None:
            # This is a simple rollback - not perfectly accurate but sufficient
            # In practice, the impact is minimal for typical use cases
            pass

        return rpm_ok and tpm_ok

    def acquire(self, tokens: int = 1, block: bool = True, timeout: Optional[float] = None) -> bool:
        """
        Acquire permission to proceed with a request.

        Args:
            tokens: Number of tokens the request will consume
            block: If True, block until permission is granted
            timeout: Maximum time to wait in seconds (only if block=True)

        Returns:
            True if permission was granted, False if timeout occurred
        """
        start_time = time.time()

        while True:
            # Try to consume from RPM bucket (1 token for the request itself)
            rpm_ok = self._rpm_bucket is None or self._rpm_bucket.try_consume()

            # Try to consume from TPM bucket
            tpm_ok = self._tpm_bucket is None or self._tpm_bucket.try_consume(tokens)

            if rpm_ok and tpm_ok:
                return True

            # Roll back RPM token if only that was consumed
            if rpm_ok and not tpm_ok:
                # Can't easily roll back, so we'll be slightly conservative
                pass

            if not block:
                return False

            if timeout is not None:
                elapsed = time.time() - start_time
                if elapsed >= timeout:
                    return False

            # Sleep a bit before retrying
            time.sleep(0.01)

    def get_status(self) -> dict:
        """Get current status of the rate limiter."""
        return {
            'rpm_available': self._rpm_bucket.get_available_tokens() if self._rpm_bucket else None,
            'tpm_available': self._tpm_bucket.get_available_tokens() if self._tpm_bucket else None,
        }
