"""Coordinate on-demand work that must eventually cover a dense universe.

The coordinator owns only index claims, waiter synchronization, completion
coverage, and sticky failure propagation. It deliberately does not create
threads, call providers, or retain completed values unless an active waiter
needs the value produced by the current owner.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from threading import Condition
from typing import Any, Literal

from .._validation import non_negative_int

_MISSING = object()
_AttemptStatus = Literal["running", "completed", "failed"]


class CoverageAbortedError(RuntimeError):
    """Raised when coverage coordination is explicitly aborted."""


@dataclass
class _Attempt:
    index: int
    status: _AttemptStatus = "running"
    value: Any = _MISSING
    error: BaseException | None = None
    value_waiters: int = 0


class CoverageLease:
    """A single index claim returned by :class:`CoverageCoordinator`."""

    def __init__(
        self,
        coordinator: CoverageCoordinator,
        attempt: _Attempt,
        *,
        owner: bool,
        require_value: bool,
    ) -> None:
        self.index = attempt.index
        self.owner = owner
        self._coordinator = coordinator
        self._attempt = attempt
        self._require_value = require_value
        self._local_value: Any = _MISSING

    def complete(self, value: Any) -> None:
        """Complete an owned attempt and wake all waiters for its index."""

        if not self.owner:
            raise RuntimeError("only the owner can complete a coverage lease.")
        self._coordinator._complete(self, value)

    def fail(self, error: BaseException) -> None:
        """Fail an owned attempt and make the error globally sticky."""

        if not self.owner:
            raise RuntimeError("only the owner can fail a coverage lease.")
        if not isinstance(error, BaseException):
            raise TypeError("error must be an exception.")
        self._coordinator._fail(self, error)

    def wait(self) -> Any:
        """Wait for the owner and return its value when one was requested."""

        return self._coordinator._wait(self)


class CoverageBatchLease:
    """Ordered collection of leases acquired by one batch claim."""

    def __init__(self, leases: tuple[CoverageLease, ...]) -> None:
        self.leases = leases

    @property
    def owners(self) -> tuple[CoverageLease, ...]:
        return tuple(lease for lease in self.leases if lease.owner)

    def wait(self) -> tuple[Any, ...]:
        return tuple(lease.wait() for lease in self.leases)

    def __iter__(self) -> Iterator[CoverageLease]:
        return iter(self.leases)

    def __len__(self) -> int:
        return len(self.leases)

    def __getitem__(self, index: int) -> CoverageLease:
        return self.leases[index]


class CoverageCoordinator:
    """Synchronize demand-driven owners while tracking dense completion.

    ``completed`` records durable coverage known before this process starts.
    It does not provide retained values. A value-requiring claim for such an
    index therefore becomes a new owner, while ordinary claims remain covered.
    """

    def __init__(
        self,
        expected: int,
        completed: Iterable[int] = (),
    ) -> None:
        self.expected = non_negative_int("expected", expected)
        completed_indexes = set(completed)
        for index in completed_indexes:
            self._validate_index(index)
        self._completed = completed_indexes
        self._condition = Condition()
        self._inflight: dict[int, _Attempt] = {}
        self._available: dict[int, _Attempt] = {}
        self._active_owners = 0
        self._error: BaseException | None = None
        self._closed = False

    @property
    def completed_count(self) -> int:
        with self._condition:
            return len(self._completed)

    @property
    def complete(self) -> bool:
        with self._condition:
            return len(self._completed) == self.expected

    def claim(
        self,
        index: int,
        *,
        require_value: bool = False,
    ) -> CoverageLease:
        """Claim one index, becoming its owner only when work is required."""

        self._validate_require_value(require_value)
        self._validate_index(index)
        with self._condition:
            self._raise_unavailable_locked()
            return self._claim_locked(index, require_value=require_value)

    def claim_batch(
        self,
        indexes: Iterable[int],
        *,
        require_value: bool = False,
    ) -> CoverageBatchLease:
        """Atomically claim an ordered group of indexes."""

        self._validate_require_value(require_value)
        claimed_indexes = tuple(indexes)
        for index in claimed_indexes:
            self._validate_index(index)
        with self._condition:
            self._raise_unavailable_locked()
            return CoverageBatchLease(
                tuple(
                    self._claim_locked(index, require_value=require_value)
                    for index in claimed_indexes
                )
            )

    def full_sweep(self) -> Iterator[CoverageLease]:
        """Yield owner leases for indexes not completed or already in flight."""

        for index in range(self.expected):
            lease = self.claim(index)
            if lease.owner:
                yield lease

    def wait_complete(self) -> None:
        """Wait for dense coverage without closing the coordinator."""

        with self._condition:
            while len(self._completed) != self.expected:
                self._raise_error_locked()
                if self._closed and self._active_owners == 0:
                    raise RuntimeError(
                        "coverage coordinator closed before coverage completed."
                    )
                self._condition.wait()
            self._raise_error_locked()

    def close(self) -> None:
        """Block new claims and wait for all current owners to finish."""

        with self._condition:
            self._closed = True
            self._condition.notify_all()
            while self._active_owners != 0:
                self._condition.wait()
            self._raise_error_locked()

    def abort(self, error: BaseException | None = None) -> None:
        """Abort current work and wake every waiter immediately."""

        if error is not None and not isinstance(error, BaseException):
            raise TypeError("error must be an exception or None.")
        with self._condition:
            if self._error is None:
                if error is None:
                    error = CoverageAbortedError(
                        "coverage coordination was aborted."
                    )
                self._error = error
            self._closed = True
            for attempt in self._inflight.values():
                attempt.status = "failed"
                attempt.error = self._error
                attempt.value = _MISSING
            self._inflight.clear()
            self._active_owners = 0
            self._condition.notify_all()

    def _claim_locked(
        self,
        index: int,
        *,
        require_value: bool,
    ) -> CoverageLease:
        attempt = self._inflight.get(index)
        if attempt is not None:
            if require_value:
                attempt.value_waiters += 1
            return CoverageLease(
                self,
                attempt,
                owner=False,
                require_value=require_value,
            )

        if index in self._completed:
            if not require_value:
                return CoverageLease(
                    self,
                    _Attempt(index=index, status="completed"),
                    owner=False,
                    require_value=False,
                )
            available = self._available.get(index)
            if available is not None:
                available.value_waiters += 1
                return CoverageLease(
                    self,
                    available,
                    owner=False,
                    require_value=True,
                )

        attempt = _Attempt(index=index)
        self._inflight[index] = attempt
        self._active_owners += 1
        return CoverageLease(
            self,
            attempt,
            owner=True,
            require_value=require_value,
        )

    def _complete(self, lease: CoverageLease, value: Any) -> None:
        with self._condition:
            attempt = self._owned_attempt_locked(lease)
            if lease._require_value:
                lease._local_value = value
            attempt.status = "completed"
            self._inflight.pop(lease.index)
            self._active_owners -= 1
            if self._error is None:
                self._completed.add(lease.index)
                if attempt.value_waiters != 0:
                    attempt.value = value
                    self._available[lease.index] = attempt
            self._condition.notify_all()

    def _fail(self, lease: CoverageLease, error: BaseException) -> None:
        with self._condition:
            attempt = self._owned_attempt_locked(lease)
            attempt.status = "failed"
            attempt.error = error
            self._inflight.pop(lease.index)
            self._active_owners -= 1
            if self._error is None:
                self._error = error
            self._condition.notify_all()

    def _wait(self, lease: CoverageLease) -> Any:
        with self._condition:
            attempt = lease._attempt
            while attempt.status == "running" and self._error is None:
                self._condition.wait()
            if attempt.status == "failed":
                if attempt.error is None:
                    raise RuntimeError("failed coverage attempt has no error.")
                raise attempt.error
            if attempt.status != "completed":
                self._raise_error_locked()
            if not lease._require_value:
                return None
            if lease._local_value is not _MISSING:
                return lease._local_value
            if attempt.value is _MISSING:
                raise RuntimeError(
                    f"coverage value for index {lease.index} is no longer available."
                )
            value = attempt.value
            lease._local_value = value
            attempt.value_waiters -= 1
            if attempt.value_waiters == 0:
                attempt.value = _MISSING
                if self._available.get(lease.index) is attempt:
                    self._available.pop(lease.index)
            return value

    def _owned_attempt_locked(self, lease: CoverageLease) -> _Attempt:
        attempt = lease._attempt
        if attempt.status != "running":
            raise RuntimeError("coverage lease is no longer active.")
        if self._inflight.get(lease.index) is not attempt:
            raise RuntimeError("coverage lease does not own the current attempt.")
        return attempt

    def _discard_available_locked(self) -> None:
        for attempt in self._available.values():
            attempt.value = _MISSING
        self._available.clear()

    def _raise_unavailable_locked(self) -> None:
        self._raise_error_locked()
        if self._closed:
            raise RuntimeError("coverage coordinator is closed.")

    def _raise_error_locked(self) -> None:
        if self._error is not None:
            raise self._error

    def _validate_index(self, index: int) -> None:
        if not isinstance(index, int) or isinstance(index, bool):
            raise TypeError("coverage index must be an integer.")
        if index < 0 or index >= self.expected:
            raise IndexError(
                f"coverage index {index} is outside [0, {self.expected})."
            )

    @staticmethod
    def _validate_require_value(require_value: bool) -> None:
        if not isinstance(require_value, bool):
            raise TypeError("require_value must be a bool.")
