"""Lightweight online reject-and-replace for map-style datasets.

This is a low reject-rate CPU safety net, not a substitute for
``FilteredDataset`` cached partitions. Heavy or GPU predicates belong on the
offline ``FilterRule.apply`` path.
"""

from __future__ import annotations

import logging
import random
import time
from collections import deque
from collections.abc import Callable, Iterator, Sequence
from typing import cast

from ..dataset.abc import MapStyleABC
from ..types.item import Sample
from .rules import label
from .types import FilterDecision, FilterOutput, FilterPredicate

_LOGGER = logging.getLogger(__name__)

_DEFAULT_BUFFER_SIZE = 64
_DEFAULT_MIN_BUFFER = 8
_DEFAULT_UPDATE_PROB = 0.25
_DEFAULT_MAX_PROBE = 32
_DEFAULT_WARN_REJECT_RATIO = 0.05
_DEFAULT_MAX_REJECT_RATIO = 0.20
_DEFAULT_STATS_WINDOW = 200
_DEFAULT_WARN_COOLDOWN_S = 30.0
_ACCEPT = "accept"


class RejectReplaceDataset(MapStyleABC):
    """Replace rejected samples with sequential look-ahead or an accept buffer.

    ``__len__`` stays equal to the wrapped dataset. Each ``__getitem__`` keeps
    the requested index when the predicate accepts it; otherwise it probes
    nearby indices, then falls back to a worker-local accept buffer.
    """

    __slots__ = (
        "_buffer",
        "_buffer_size",
        "_dataset",
        "_max_probe",
        "_max_reject_ratio",
        "_min_buffer",
        "_name",
        "_predicate",
        "_rng",
        "_stats",
        "_stats_window",
        "_update_prob",
        "_warn_cooldown_s",
        "_warn_reject_ratio",
        "_warned_at",
    )

    def __init__(
        self,
        dataset: MapStyleABC,
        predicate: FilterPredicate,
        *,
        name: str = "online_filter",
        buffer_size: int = _DEFAULT_BUFFER_SIZE,
        min_buffer: int = _DEFAULT_MIN_BUFFER,
        update_prob: float = _DEFAULT_UPDATE_PROB,
        max_probe: int = _DEFAULT_MAX_PROBE,
        warn_reject_ratio: float = _DEFAULT_WARN_REJECT_RATIO,
        max_reject_ratio: float = _DEFAULT_MAX_REJECT_RATIO,
        stats_window: int = _DEFAULT_STATS_WINDOW,
        warn_cooldown_s: float = _DEFAULT_WARN_COOLDOWN_S,
        seed: int | None = None,
    ) -> None:
        if not isinstance(dataset, MapStyleABC):
            raise TypeError("dataset must be a MapStyleABC.")
        if not callable(predicate):
            raise TypeError("predicate must be callable.")
        if not isinstance(name, str) or name == "":
            raise ValueError("name must be a non-empty string.")
        if type(buffer_size) is not int or buffer_size < 1:
            raise ValueError("buffer_size must be an integer >= 1.")
        if type(min_buffer) is not int or min_buffer < 1:
            raise ValueError("min_buffer must be an integer >= 1.")
        if min_buffer > buffer_size:
            raise ValueError("min_buffer must be <= buffer_size.")
        if not isinstance(update_prob, (int, float)) or isinstance(update_prob, bool):
            raise TypeError("update_prob must be a float.")
        if not 0.0 <= float(update_prob) <= 1.0:
            raise ValueError("update_prob must be in [0, 1].")
        if type(max_probe) is not int or max_probe < 1:
            raise ValueError("max_probe must be an integer >= 1.")
        if not isinstance(warn_reject_ratio, (int, float)) or isinstance(
            warn_reject_ratio, bool
        ):
            raise TypeError("warn_reject_ratio must be a float.")
        if not isinstance(max_reject_ratio, (int, float)) or isinstance(
            max_reject_ratio, bool
        ):
            raise TypeError("max_reject_ratio must be a float.")
        warn_ratio = float(warn_reject_ratio)
        max_ratio = float(max_reject_ratio)
        if not 0.0 <= warn_ratio <= max_ratio <= 1.0:
            raise ValueError(
                "require 0 <= warn_reject_ratio <= max_reject_ratio <= 1."
            )
        if type(stats_window) is not int or stats_window < 1:
            raise ValueError("stats_window must be an integer >= 1.")
        if not isinstance(warn_cooldown_s, (int, float)) or isinstance(
            warn_cooldown_s, bool
        ):
            raise TypeError("warn_cooldown_s must be a float.")
        if float(warn_cooldown_s) < 0.0:
            raise ValueError("warn_cooldown_s must be >= 0.")
        if seed is not None and type(seed) is not int:
            raise TypeError("seed must be an integer or None.")

        self._dataset = dataset
        self._predicate = predicate
        self._name = name
        self._buffer_size = buffer_size
        self._min_buffer = min_buffer
        self._update_prob = float(update_prob)
        self._max_probe = max_probe
        self._warn_reject_ratio = warn_ratio
        self._max_reject_ratio = max_ratio
        self._stats_window = stats_window
        self._warn_cooldown_s = float(warn_cooldown_s)
        self._buffer: list[Sample] = []
        self._stats: deque[bool] = deque(maxlen=stats_window)
        self._warned_at = float("-inf")
        self._rng = random.Random(seed)

    @property
    def dataset(self) -> MapStyleABC:
        return self._dataset

    @property
    def name(self) -> str:
        return self._name

    @property
    def predicate(self) -> FilterPredicate:
        return self._predicate

    @property
    def buffer_size(self) -> int:
        return self._buffer_size

    @property
    def min_buffer(self) -> int:
        return self._min_buffer

    @property
    def buffer_filled(self) -> int:
        return len(self._buffer)

    @property
    def reject_ratio(self) -> float | None:
        if not self._stats:
            return None
        rejects = sum(1 for accepted in self._stats if not accepted)
        return rejects / len(self._stats)

    def __len__(self) -> int:
        return len(self._dataset)

    def __getitem__(self, index: int) -> Sample:
        if type(index) is not int:
            raise TypeError("index must be an integer.")
        length = len(self._dataset)
        if index < 0 or index >= length:
            raise IndexError(f"index {index} out of range for length {length}.")

        sample = self._dataset[index]
        if self._accepted(sample):
            self._record(accepted=True)
            self._store(sample)
            return sample

        self._record(accepted=False)
        self._check_rates(primary_index=index)

        for offset in range(1, self._max_probe + 1):
            candidate_index = (index + offset) % length
            if candidate_index == index:
                continue
            candidate = self._dataset[candidate_index]
            if self._accepted(candidate):
                self._store(candidate)
                return candidate

        if len(self._buffer) >= self._min_buffer:
            return self._buffer[self._rng.randrange(len(self._buffer))]

        raise RuntimeError(
            f"online filter {self._name!r} could not find an accept sample "
            f"for index {index}: probed {self._max_probe} neighbors and "
            f"buffer has {len(self._buffer)}/{self._min_buffer} accepts. "
            "Use FilteredDataset for high reject rates."
        )

    def global_index(self, index: int) -> int:
        method = getattr(self._dataset, "global_index", None)
        if not callable(method):
            return index
        return cast(Callable[[int], int], method)(index)

    def _shuffle(
        self,
        *,
        shuffle: bool,
        seed: int,
        epoch: int,
        num_replicas: int,
        rank: int,
    ) -> Iterator[Sequence[int]]:
        yield from self._dataset._shuffle(
            shuffle=shuffle,
            seed=seed,
            epoch=epoch,
            num_replicas=num_replicas,
            rank=rank,
        )

    def _accepted(self, sample: Sample) -> bool:
        return _decision_label(self._predicate(sample)) == _ACCEPT

    def _store(self, sample: Sample) -> None:
        if len(self._buffer) < self._buffer_size:
            self._buffer.append(sample)
            return
        if self._rng.random() < self._update_prob:
            self._buffer[self._rng.randrange(self._buffer_size)] = sample

    def _record(self, *, accepted: bool) -> None:
        self._stats.append(accepted)

    def _check_rates(self, *, primary_index: int) -> None:
        ratio = self.reject_ratio
        if ratio is None or len(self._stats) < self._stats_window:
            return
        rejects = sum(1 for accepted in self._stats if not accepted)
        accepts = len(self._stats) - rejects
        if ratio >= self._max_reject_ratio:
            raise RuntimeError(
                f"online filter {self._name!r} reject ratio "
                f"{ratio:.1%} over the last {len(self._stats)} primary reads "
                f"(reject={rejects}, accept={accepts}) exceeds "
                f"max_reject_ratio={self._max_reject_ratio:.1%} at index "
                f"{primary_index}. Use FilteredDataset for high reject rates."
            )
        if ratio < self._warn_reject_ratio:
            return
        now = time.monotonic()
        if now - self._warned_at < self._warn_cooldown_s:
            return
        self._warned_at = now
        _LOGGER.warning(
            "online filter %r reject ratio %.1f%% over the last %d primary "
            "reads (reject=%d, accept=%d) at index %d; prefer FilteredDataset "
            "when rejects are common",
            self._name,
            ratio * 100.0,
            len(self._stats),
            rejects,
            accepts,
            primary_index,
        )


def _decision_label(value: FilterOutput) -> str:
    if isinstance(value, FilterDecision):
        return label(value.label)
    return label(value)


__all__ = ["RejectReplaceDataset"]
