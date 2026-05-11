"""
data/http_resilient.py — Resilient HTTP layer for Bybit V5 API calls.

Provides:
  - request_with_retry()  : drop-in replacement for requests.get with
                            exponential-backoff retry on transient errors.
  - CircuitBreaker        : globally-shared breaker that pauses requests
                            after N consecutive failures.
  - DEFAULT_RETRY         : sensible defaults (3 retries, 1/2/4 s backoff).
  - BYBIT_BREAKER         : singleton breaker shared by all callers.

Retry-eligible conditions:
  - HTTP 429 (rate limited)
  - HTTP 5xx (server error)
  - requests.exceptions.ConnectionError
  - requests.exceptions.Timeout
  - requests.exceptions.ChunkedEncodingError
  - Any other requests.RequestException (treated as transient)

Non-retryable (returned immediately):
  - HTTP 4xx other than 429 (caller's request is wrong)
  - Successful response (2xx)

Returns None when:
  - All retries exhausted
  - Circuit breaker is OPEN (paused window)

Design notes:
  - Single-threaded: no locks. Safe inside the paper-trader's single poll thread.
  - Logging: every retry, every breaker state change, and every final failure
    emits a WARNING/ERROR log so 24/7 operators can grep for SLA events.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Optional

import requests

log = logging.getLogger(__name__)


# ── Retry config ──────────────────────────────────────────────────────────────

@dataclass
class RetryConfig:
    """
    Tunable retry parameters.

    max_retries  : number of additional attempts after the first failure.
                   Total HTTP calls = 1 + max_retries.  Default 3 = 4 calls.
    base_delay_s : delay before the first retry (seconds).
    max_delay_s  : ceiling for backoff growth (seconds).
    backoff_mult : exponential multiplier between retries.
                   With base=1.0 and mult=2.0 the sequence is 1s, 2s, 4s, 8s ...
    """
    max_retries:  int   = 3
    base_delay_s: float = 1.0
    max_delay_s:  float = 8.0
    backoff_mult: float = 2.0


DEFAULT_RETRY = RetryConfig()

# HTTP status codes that should trigger a retry rather than abort.
RETRYABLE_STATUSES: frozenset[int] = frozenset({408, 425, 429, 500, 502, 503, 504})


# ── Circuit breaker ───────────────────────────────────────────────────────────

class CircuitBreaker:
    """
    Simple three-state breaker.

    States:
      CLOSED     : requests pass through (normal operation).
      OPEN       : requests are blocked for cooldown_s seconds after N consecutive
                   failures.  allow_request() returns False during this window.
      HALF_OPEN  : after cooldown, a single probe request is allowed.  Success
                   closes the breaker; failure re-opens for another cooldown.

    Thresholds default to 10 consecutive failures with 60 s cooldown.
    """

    def __init__(
        self,
        name: str = "bybit",
        failure_threshold: int = 10,
        cooldown_s: float = 60.0,
    ) -> None:
        self.name              = name
        self.failure_threshold = failure_threshold
        self.cooldown_s        = cooldown_s
        self._state            = "CLOSED"
        self._consecutive_fail = 0
        self._opened_at        = 0.0

    @property
    def state(self) -> str:
        return self._state

    def allow_request(self) -> bool:
        """
        Returns True if a request should be attempted right now.

        Transitions OPEN -> HALF_OPEN automatically when cooldown has elapsed.
        """
        if self._state == "OPEN":
            if (time.monotonic() - self._opened_at) >= self.cooldown_s:
                self._state = "HALF_OPEN"
                log.warning(
                    "[circuit:%s] cooldown elapsed -> HALF_OPEN (probe next request)",
                    self.name,
                )
                return True
            return False
        return True   # CLOSED or HALF_OPEN

    def record_success(self) -> None:
        """Reset the breaker on any successful response."""
        if self._state != "CLOSED":
            log.info(
                "[circuit:%s] success on %s -> CLOSED",
                self.name, self._state,
            )
        self._state            = "CLOSED"
        self._consecutive_fail = 0

    def record_failure(self) -> None:
        """Count a failed request.  Opens the breaker once threshold hit."""
        self._consecutive_fail += 1

        if self._state == "HALF_OPEN":
            # Probe failed -> back to OPEN with fresh cooldown
            self._state     = "OPEN"
            self._opened_at = time.monotonic()
            log.error(
                "[circuit:%s] probe failed (consecutive=%d) -> OPEN for %.0fs",
                self.name, self._consecutive_fail, self.cooldown_s,
            )
            return

        if self._consecutive_fail >= self.failure_threshold and self._state == "CLOSED":
            self._state     = "OPEN"
            self._opened_at = time.monotonic()
            log.error(
                "[circuit:%s] %d consecutive failures -> OPEN for %.0fs",
                self.name, self._consecutive_fail, self.cooldown_s,
            )


# Module-level singleton shared by all Bybit callers.
BYBIT_BREAKER = CircuitBreaker(name="bybit")


# ── Resilient request ─────────────────────────────────────────────────────────

class CircuitOpenError(RuntimeError):
    """Raised when the breaker is OPEN and a request was blocked."""


def request_with_retry(
    url: str,
    *,
    method:   str           = "GET",
    params:   Optional[dict] = None,
    headers:  Optional[dict] = None,
    timeout:  float         = 20.0,
    retry:    RetryConfig   = DEFAULT_RETRY,
    breaker:  CircuitBreaker = BYBIT_BREAKER,
) -> Optional[requests.Response]:
    """
    Issue an HTTP request with retry + exponential backoff + circuit breaker.

    Returns
    -------
    requests.Response on success (status_code 2xx-4xx-except-retryable).
    None if breaker is OPEN, or all retries exhausted.

    A 4xx response (other than retryable codes) is returned to the caller so it
    can read .status_code / .text; it does NOT trigger a retry.  Only transport
    errors and retryable status codes (429, 5xx, etc.) consume retry budget.
    """
    if not breaker.allow_request():
        log.warning(
            "[circuit:%s] OPEN — skipping request to %s",
            breaker.name, url,
        )
        return None

    delay = retry.base_delay_s
    last_err: Optional[str] = None

    for attempt in range(retry.max_retries + 1):
        try:
            resp = requests.request(
                method,
                url,
                params=params,
                headers=headers,
                timeout=timeout,
            )

            # Retryable HTTP status: count as failure, back off, retry
            if resp.status_code in RETRYABLE_STATUSES:
                last_err = f"HTTP {resp.status_code}"
                breaker.record_failure()
                _log_retry(attempt, retry, url, last_err)
                if attempt < retry.max_retries:
                    time.sleep(delay)
                    delay = min(delay * retry.backoff_mult, retry.max_delay_s)
                    continue
                log.error(
                    "Request to %s exhausted %d retries (last error: %s)",
                    url, retry.max_retries, last_err,
                )
                return None

            # Any other response (2xx-4xx not in retryable set): success path
            breaker.record_success()
            return resp

        except (requests.exceptions.ConnectionError,
                requests.exceptions.Timeout,
                requests.exceptions.ChunkedEncodingError) as exc:
            # Transport-level errors are retryable
            last_err = type(exc).__name__
            breaker.record_failure()
            _log_retry(attempt, retry, url, last_err)
            if attempt < retry.max_retries:
                time.sleep(delay)
                delay = min(delay * retry.backoff_mult, retry.max_delay_s)
                continue
            log.error(
                "Request to %s exhausted %d retries (last error: %s)",
                url, retry.max_retries, last_err,
            )
            return None

        except requests.exceptions.RequestException as exc:
            # Unknown but transport-related: retry once more, then give up
            last_err = f"{type(exc).__name__}: {exc}"
            breaker.record_failure()
            _log_retry(attempt, retry, url, last_err)
            if attempt < retry.max_retries:
                time.sleep(delay)
                delay = min(delay * retry.backoff_mult, retry.max_delay_s)
                continue
            log.error(
                "Request to %s exhausted %d retries (last error: %s)",
                url, retry.max_retries, last_err,
            )
            return None

    # Should be unreachable, but be defensive
    return None


def _log_retry(attempt: int, retry: RetryConfig, url: str, err: str) -> None:
    """One-line retry log line; placed at WARNING level for grep-ability."""
    if attempt < retry.max_retries:
        log.warning(
            "Retry %d/%d for %s after %s",
            attempt + 1, retry.max_retries, url, err,
        )
