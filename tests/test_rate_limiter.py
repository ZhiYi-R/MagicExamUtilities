"""
Unit tests for rate_limiter module.
"""

import time
import pytest

from utilities.rate_limiter import TokenBucket, RateLimiter


@pytest.mark.unit
class TestTokenBucket:
    """Test cases for TokenBucket class."""

    def test_initialization(self):
        """Test TokenBucket initialization."""
        bucket = TokenBucket(capacity=100, refill_rate=10)
        assert bucket.get_available_tokens() == 100

    def test_initialization_with_custom_tokens(self):
        """Test TokenBucket initialization with custom initial tokens."""
        bucket = TokenBucket(capacity=100, refill_rate=10, initial_tokens=50)
        assert bucket.get_available_tokens() == 50

    def test_consume_tokens(self):
        """Test consuming tokens from bucket."""
        bucket = TokenBucket(capacity=100, refill_rate=10)
        assert bucket.try_consume(10) is True
        assert bucket.get_available_tokens() == 90

    def test_consume_more_than_available(self):
        """Test consuming more tokens than available."""
        bucket = TokenBucket(capacity=100, refill_rate=10)
        assert bucket.try_consume(150) is False
        # Tokens should not be consumed on failure
        assert bucket.get_available_tokens() == 100

    def test_refill_over_time(self):
        """Test that tokens refill over time."""
        bucket = TokenBucket(capacity=10, refill_rate=60)  # 60 tokens per second = 1 per second
        bucket.try_consume(10)
        assert bucket.get_available_tokens() == 0

        time.sleep(0.1)  # Wait 100ms
        available = bucket.get_available_tokens()
        assert available > 0
        assert available < 10  # Should not be full yet

    def test_consume_blocking(self):
        """Test blocking consume."""
        bucket = TokenBucket(capacity=10, refill_rate=60)  # 60 tokens per second = 1 per ~17ms
        bucket.try_consume(10)
        assert bucket.get_available_tokens() == 0

        start = time.time()
        result = bucket.consume(1, block=True, timeout=0.15)
        elapsed = time.time() - start

        assert result is True
        # Should wait for at least one token refill period (~17ms)
        # But less than timeout
        assert 0.01 <= elapsed < 0.15

    def test_consume_blocking_timeout(self):
        """Test blocking consume with timeout."""
        bucket = TokenBucket(capacity=10, refill_rate=6)  # 0.1 token per second
        bucket.try_consume(10)
        assert bucket.get_available_tokens() == 0

        start = time.time()
        result = bucket.consume(1, block=True, timeout=0.1)
        elapsed = time.time() - start

        assert result is False
        assert elapsed >= 0.1  # Should have waited for timeout

    def test_consume_non_blocking(self):
        """Test non-blocking consume."""
        bucket = TokenBucket(capacity=10, refill_rate=60)
        bucket.try_consume(10)

        result = bucket.consume(1, block=False)
        assert result is False

    def test_capacity_limit(self):
        """Test that tokens never exceed capacity."""
        bucket = TokenBucket(capacity=100, refill_rate=60)
        bucket.try_consume(50)

        time.sleep(2)  # Wait for refill
        available = bucket.get_available_tokens()
        assert available <= 100

    def test_multiple_consumptions(self):
        """Test multiple token consumptions."""
        bucket = TokenBucket(capacity=100, refill_rate=60)

        for _ in range(10):
            assert bucket.try_consume(5) is True

        assert bucket.get_available_tokens() == 50


@pytest.mark.unit
class TestRateLimiter:
    """Test cases for RateLimiter class."""

    def test_initialization_with_no_limits(self):
        """Test RateLimiter with no limits."""
        limiter = RateLimiter(rpm=None, tpm=None)
        status = limiter.get_status()
        assert status['rpm_available'] is None
        assert status['tpm_available'] is None

    def test_initialization_with_rpm_only(self):
        """Test RateLimiter with only RPM limit."""
        limiter = RateLimiter(rpm=60, tpm=None)
        status = limiter.get_status()
        assert status['rpm_available'] == 60
        assert status['tpm_available'] is None

    def test_initialization_with_tpm_only(self):
        """Test RateLimiter with only TPM limit."""
        limiter = RateLimiter(rpm=None, tpm=10000)
        status = limiter.get_status()
        assert status['rpm_available'] is None
        assert status['tpm_available'] == 10000

    def test_initialization_with_both_limits(self):
        """Test RateLimiter with both RPM and TPM limits."""
        limiter = RateLimiter(rpm=60, tpm=10000)
        status = limiter.get_status()
        assert status['rpm_available'] == 60
        assert status['tpm_available'] == 10000

    def test_can_proceed_no_limits(self):
        """Test can_proceed with no limits."""
        limiter = RateLimiter(rpm=None, tpm=None)
        assert limiter.can_proceed(tokens=100) is True

    def test_can_proceed_with_rpm(self):
        """Test can_proceed with RPM limit."""
        limiter = RateLimiter(rpm=60, tpm=None)

        for _ in range(60):
            assert limiter.can_proceed() is True

        assert limiter.can_proceed() is False

    def test_can_proceed_with_tpm(self):
        """Test can_proceed with TPM limit."""
        limiter = RateLimiter(rpm=None, tpm=1000)

        assert limiter.can_proceed(tokens=500) is True
        assert limiter.can_proceed(tokens=500) is True
        assert limiter.can_proceed(tokens=1) is False

    def test_can_proceed_with_both_limits(self):
        """Test can_proceed with both RPM and TPM limits."""
        limiter = RateLimiter(rpm=10, tpm=1000)

        # Should be limited by RPM first
        for _ in range(10):
            assert limiter.can_proceed(tokens=1) is True

        # RPM exhausted
        assert limiter.can_proceed(tokens=1) is False

    def test_acquire_no_limits(self):
        """Test acquire with no limits."""
        limiter = RateLimiter(rpm=None, tpm=None)
        assert limiter.acquire(tokens=1000, block=False) is True

    def test_acquire_blocking(self):
        """Test blocking acquire."""
        limiter = RateLimiter(rpm=60, tpm=None)  # 1 RPS (60 requests per minute)

        # Consume all tokens
        for _ in range(60):
            limiter.acquire(block=False)

        start = time.time()
        result = limiter.acquire(block=True, timeout=2.0)
        elapsed = time.time() - start

        assert result is True
        # Should wait for at least ~1 second to get 1 token
        assert 0.8 <= elapsed < 2.0

    def test_acquire_timeout(self):
        """Test acquire with timeout."""
        limiter = RateLimiter(rpm=6, tpm=None)  # 0.1 RPS

        # Consume all tokens
        for _ in range(6):
            limiter.acquire(block=False)

        start = time.time()
        result = limiter.acquire(block=True, timeout=0.5)
        elapsed = time.time() - start

        assert result is False
        assert elapsed >= 0.5

    def test_rpm_and_tpm_both_consumed(self):
        """Test that both RPM and TPM are consumed."""
        limiter = RateLimiter(rpm=60, tpm=6000)

        # Consume 1 request with 100 tokens
        limiter.acquire(tokens=100, block=False)

        status = limiter.get_status()
        assert status['rpm_available'] == 59
        assert status['tpm_available'] == 5900

    def test_tpm_limit_blocks_requests(self):
        """Test that TPM limit blocks requests even with RPM available."""
        limiter = RateLimiter(rpm=1000, tpm=100)

        # One large request should exhaust TPM but not RPM
        assert limiter.acquire(tokens=100, block=False) is True

        status = limiter.get_status()
        assert status['rpm_available'] == 999
        assert status['tpm_available'] == 0

        # Next request should be blocked by TPM
        assert limiter.can_proceed(tokens=1) is False

    def test_rpm_limit_blocks_requests(self):
        """Test that RPM limit blocks requests even with TPM available."""
        limiter = RateLimiter(rpm=2, tpm=100000)

        # Consume all RPM
        assert limiter.acquire(block=False) is True
        assert limiter.acquire(block=False) is True

        status = limiter.get_status()
        assert status['rpm_available'] == 0
        assert status['tpm_available'] > 0

        # Next request should be blocked by RPM
        assert limiter.can_proceed() is False

    def test_zero_limits_disables_limiting(self):
        """Test that setting 0 disables the limit."""
        limiter = RateLimiter(rpm=0, tpm=0)
        assert limiter.can_proceed(tokens=1000000) is True
