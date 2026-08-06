"""Online filter runs over complete dataset universes.

``FilterRun`` separates durable, full-universe decision materialization from
the selection returned to callers. Existing selections are intersected only
at the returned ``SelectionView`` boundary; they never enter filter identity
or predicate scan coverage.
"""

from __future__ import annotations

import hashlib
import operator
import os
import uuid
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from threading import Condition, Lock, Thread
from typing import Any, cast

from .._io.parquet import pyarrow
from .._runtime.resume import resume_dir
from ..dataset.abc import MapStyleABC
from ..dataset.universe import DatasetUniverse, SampleIdentity
from ..dataset.view import (
    DecisionSet,
    rebase_selection,
    Selection,
    SelectionView,
    StaticSelection,
    UnknownDecisionError,
)
from .cache.identity import (
    FilterBase,
    filter_base,
    filter_identity,
    filter_identity_key,
    filter_path,
    filter_universe,
    metadata,
)
from .cache.ready import ready_filter_generation
from .cache.storage import read_partitions
from .runtime.apply import apply_filter
from .types import (
    DatasetFactory,
    FilterRunStatus,
    _FilterChunk,
)


@dataclass(frozen=True)
class _IndexSampleIdentity(SampleIdentity):
    dataset: MapStyleABC
    namespace: str

    def sample_id(self, index: int) -> str:
        global_index = getattr(self.dataset, "global_index", None)
        value = global_index(index) if callable(global_index) else index
        if type(value) is not int:
            raise TypeError("global_index() must return an integer.")
        return f"{self.namespace}:{value}"


@dataclass(frozen=True)
class _OpenedDatasetFactory:
    dataset: FilterBase

    def __call__(self) -> FilterBase:
        return self.dataset


class _LiveSelection:
    """Mutable decision coverage behind the immutable selection contract."""

    def __init__(
        self,
        universe: DatasetUniverse,
        labels: frozenset[str] | None,
    ) -> None:
        self._universe = universe
        self._labels = labels
        self._decisions: list[str | None] = [None] * len(universe)
        self._condition = Condition()
        self._completed = 0
        self._complete = False
        self._error: BaseException | None = None
        self._positions = _global_positions(universe)

    @property
    def universe(self) -> DatasetUniverse:
        return self._universe

    def contains(self, universe_index: int) -> bool | None:
        position = _position(universe_index, len(self.universe))
        with self._condition:
            self._raise_error_locked()
            decision = self._decisions[position]
            if decision is None:
                return None
            return self._selected(decision)

    def selected_index(self, position: int) -> int:
        return _LiveSelectionView(self.universe, (self,)).selected_index(position)

    def __len__(self) -> int:
        self.wait_complete()
        with self._condition:
            return sum(
                self._selected(cast(str, decision)) for decision in self._decisions
            )

    def rebase(self, universe: DatasetUniverse) -> Selection:
        return rebase_selection(self, universe)

    def publish(self, chunk: _FilterChunk) -> None:
        decisions = tuple(
            (int(global_index), partition_label)
            for partition_label, indexes in chunk.partitions.items()
            for global_index in indexes
        )
        with self._condition:
            self._raise_error_locked()
            for global_index, partition_label in decisions:
                try:
                    position = self._positions[global_index]
                except KeyError as exc:
                    raise ValueError(
                        "filter decision index is outside the dataset universe: "
                        f"{global_index}"
                    ) from exc
                current = self._decisions[position]
                if current is not None:
                    if current != partition_label:
                        raise ValueError(
                            "filter decision changed while resuming universe index "
                            f"{position}."
                        )
                    continue
                self._decisions[position] = partition_label
                self._completed += 1
            self._condition.notify_all()

    def wait_decision(self, universe_index: int) -> bool:
        position = _position(universe_index, len(self.universe))
        with self._condition:
            while self._decisions[position] is None and not self._complete:
                self._raise_error_locked()
                self._condition.wait()
            self._raise_error_locked()
            decision = self._decisions[position]
            if decision is None:
                raise RuntimeError(
                    f"filter completed without a decision for universe index {position}."
                )
            return self._selected(decision)

    def finish(self) -> None:
        with self._condition:
            self._raise_error_locked()
            if self._completed != len(self._decisions):
                error = RuntimeError(
                    "filter decision coverage is incomplete: "
                    f"expected={len(self._decisions)} completed={self._completed}."
                )
                self._error = error
                self._condition.notify_all()
                raise error
            self._complete = True
            self._condition.notify_all()

    def fail(self, error: BaseException) -> None:
        with self._condition:
            if self._error is None:
                self._error = error
            self._condition.notify_all()

    def wait_complete(self) -> None:
        with self._condition:
            while not self._complete:
                self._raise_error_locked()
                self._condition.wait()
            self._raise_error_locked()

    def _selected(self, decision: str) -> bool:
        return self._labels is None or decision in self._labels

    def _raise_error_locked(self) -> None:
        if self._error is not None:
            raise self._error


@dataclass(frozen=True)
class _OwnedSelection:
    """Keep a filter run alive for every rebased view that uses its decisions."""

    selection: Selection
    owner: FilterRun

    @property
    def universe(self) -> DatasetUniverse:
        return self.selection.universe

    def contains(self, universe_index: int) -> bool | None:
        return self.selection.contains(universe_index)

    def selected_index(self, position: int) -> int:
        return self.selection.selected_index(position)

    def __len__(self) -> int:
        return len(self.selection)

    def rebase(self, universe: DatasetUniverse) -> Selection:
        return _OwnedSelection(self.selection.rebase(universe), self.owner)

    def wait_decision(self, universe_index: int) -> bool:
        wait = getattr(self.selection, "wait_decision", None)
        if not callable(wait):
            state = self.selection.contains(universe_index)
            if state is None:
                raise UnknownDecisionError(universe_index)
            return state
        return bool(wait(universe_index))

    def wait_complete(self) -> None:
        wait = getattr(self.selection, "wait_complete", None)
        if callable(wait):
            wait()

    def close(self) -> None:
        self.owner.close()


class _LiveSelectionView(SelectionView):
    """SelectionView whose unresolved live decisions block instead of reject."""

    def selected_index(self, position: int) -> int:
        logical_position = operator.index(position)
        if logical_position < 0:
            logical_position += len(self)
        if logical_position < 0:
            raise IndexError("selection index out of range")
        selected_position = 0
        for universe_index in range(len(self.universe)):
            if self._blocking_contains(universe_index):
                if selected_position == logical_position:
                    return universe_index
                selected_position += 1
        raise IndexError("selection index out of range")

    @property
    def indices(self) -> tuple[int, ...]:
        self._wait_complete()
        return super().indices

    def __len__(self) -> int:
        self._wait_complete()
        return super().__len__()

    def _blocking_contains(self, universe_index: int) -> bool:
        while True:
            unknown: list[Any] = []
            for selection in self.selections:
                state = selection.contains(universe_index)
                if state is False:
                    return False
                if state is None:
                    unknown.append(selection)
            if not unknown:
                return True
            wait = getattr(unknown[0], "wait_decision", None)
            if not callable(wait):
                raise UnknownDecisionError(universe_index)
            wait(universe_index)

    def _wait_complete(self) -> None:
        for selection in self.selections:
            wait = getattr(selection, "wait_complete", None)
            if callable(wait):
                wait()


class _ArrowDecisionFragments:
    """Immutable Arrow IPC snapshots used only by an in-progress filter run."""

    def __init__(self, cache_path: Path) -> None:
        self.path = resume_dir(cache_path, "filter") / "decisions"
        self._lock = Lock()

    def write(self, chunk: _FilterChunk) -> None:
        rows = sorted(
            (int(index), partition_label)
            for partition_label, indexes in chunk.partitions.items()
            for index in indexes
        )
        if not rows:
            return
        digest = hashlib.sha256()
        for index, partition_label in rows:
            digest.update(str(index).encode("ascii"))
            digest.update(b"\0")
            digest.update(partition_label.encode("utf-8"))
            digest.update(b"\n")
        target = self.path / f"{digest.hexdigest()[:24]}.arrow"
        with self._lock:
            if target.is_file():
                return
            self.path.mkdir(parents=True, exist_ok=True)
            tmp = target.with_name(f".{target.name}.{uuid.uuid4().hex}.tmp")
            pa, _ = pyarrow()
            schema = pa.schema(
                (("index", pa.int64()), ("label", pa.string())),
                metadata={b"schema_version": b"1"},
            )
            table = pa.Table.from_arrays(
                (
                    pa.array((index for index, _ in rows), type=pa.int64()),
                    pa.array((value for _, value in rows), type=pa.string()),
                ),
                schema=schema,
            )
            try:
                with pa.OSFile(str(tmp), "wb") as sink:
                    with pa.ipc.new_file(sink, schema) as writer:
                        writer.write_table(table)
                os.replace(tmp, target)
            finally:
                try:
                    tmp.unlink()
                except FileNotFoundError:
                    pass


class FilterRun:
    """A background full-universe filter run and its live selection view."""

    def __init__(
        self,
        dataset: SelectionView,
        *,
        selection: _LiveSelection | None,
        generation_lease: Any = None,
    ) -> None:
        self.dataset = dataset
        self._selection = selection
        self._generation_lease = generation_lease
        self._condition = Condition()
        self._status = (
            FilterRunStatus.COMPLETE if selection is None else FilterRunStatus.RUNNING
        )
        self._error: BaseException | None = None
        self._thread: Thread | None = None
        self._closed = False

    @property
    def status(self) -> FilterRunStatus:
        with self._condition:
            return self._status

    def wait(self) -> SelectionView:
        thread = self._thread
        if thread is not None:
            thread.join()
        with self._condition:
            if self._error is not None:
                raise self._error
        return self.dataset

    def close(self) -> None:
        with self._condition:
            if self._closed:
                return
            self._closed = True
        error: BaseException | None = None
        try:
            self.wait()
        except BaseException as exc:
            error = exc
        lease = self._generation_lease
        self._generation_lease = None
        if lease is not None:
            try:
                lease.close()
            except BaseException as exc:
                if error is None:
                    error = exc
        try:
            self.dataset.close()
        except BaseException as exc:
            if error is None:
                error = exc
        if error is not None:
            raise error

    def __enter__(self) -> FilterRun:
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        self.close()

    def _start(self, target: Any) -> None:
        self._thread = Thread(
            target=target,
            name="anydataset-filter-run",
            daemon=True,
        )
        self._thread.start()

    def _complete(self) -> None:
        with self._condition:
            self._status = FilterRunStatus.COMPLETE
            self._condition.notify_all()

    def _fail(self, error: BaseException) -> None:
        if self._selection is not None:
            self._selection.fail(error)
        with self._condition:
            if self._error is None:
                self._error = error
            self._status = FilterRunStatus.FAILED
            self._condition.notify_all()


def open_filter(
    rule: Any,
    *,
    dataset_factory: DatasetFactory,
    labels: Any,
    options: Mapping[str, Any],
) -> FilterRun:
    source = dataset_factory()
    raw, upstream, source_universe = _source_view(
        source,
        input_id=options["input_id"],
    )
    identity = filter_identity(raw, input_id=options["input_id"])
    universe = _universe(raw, identity) if source_universe is None else source_universe
    upstream = _rebase_upstream(upstream, universe)
    selected_labels = _labels(labels)
    expected = metadata(identity, len(universe), rule)
    cache_path = filter_path(rule, identity)

    if not options["rebuild"]:
        generation, _ = ready_filter_generation(
            cache_path,
            expected,
            metrics=options["metrics"],
        )
        if generation is not None:
            try:
                selection = _ready_selection(
                    universe,
                    read_partitions(generation.path),
                    selected_labels,
                )
                dataset = SelectionView(
                    universe,
                    upstream + (selection,),
                )
                run = FilterRun(
                    dataset,
                    selection=None,
                    generation_lease=generation.lease,
                )
                run.dataset = SelectionView(
                    universe,
                    upstream + (_OwnedSelection(selection, run),),
                )
                return run
            except Exception:
                generation.lease.close()
                raise

    selection = _LiveSelection(universe, selected_labels)
    dataset = _LiveSelectionView(universe, upstream + (selection,))
    run = FilterRun(dataset, selection=selection)
    run.dataset = _LiveSelectionView(
        universe,
        upstream + (_OwnedSelection(selection, run),),
    )
    fragments = _ArrowDecisionFragments(cache_path)

    def observer(chunk: _FilterChunk) -> None:
        _observe_chunk(fragments, selection, chunk)

    def build() -> None:
        applied = None
        try:
            applied = apply_filter(
                rule,
                input_id=options["input_id"],
                metrics=options["metrics"],
                device=options["device"],
                batch_size=options["batch_size"],
                num_workers=options["num_workers"],
                prefetch_factor=options["prefetch_factor"],
                commit_samples=options["commit_samples"],
                max_shard_samples=options["max_shard_samples"],
                write_workers=options["write_workers"],
                write_prefetch=options["write_prefetch"],
                worker_timeout=options["worker_timeout"],
                runtime=options["runtime"],
                rebuild=options["rebuild"],
                dataset_factory=_OpenedDatasetFactory(raw),
                with_report=False,
                chunk_observer=observer,
                allow_logical=True,
            )
            _publish_partitions(selection, applied.cache._indexes)
            selection.finish()
            run._complete()
        except BaseException as exc:
            run._fail(exc)
        finally:
            if applied is not None:
                applied.cache._lease.close()

    run._start(build)
    return run


def _observe_chunk(
    fragments: _ArrowDecisionFragments,
    selection: _LiveSelection,
    chunk: _FilterChunk,
) -> None:
    fragments.write(chunk)
    selection.publish(chunk)


def _source_view(
    source: Any,
    *,
    input_id: str | None,
) -> tuple[FilterBase, tuple[Selection, ...], DatasetUniverse | None]:
    from .api import FilteredDataset

    if isinstance(source, SelectionView):
        return (
            filter_base(
                source.universe.dataset,
                input_id=input_id,
                allow_logical=True,
            ),
            source.selections,
            source.universe,
        )
    if isinstance(source, DatasetUniverse):
        return (
            filter_base(
                source.dataset,
                input_id=input_id,
                allow_logical=True,
            ),
            (),
            source,
        )
    if isinstance(source, FilteredDataset):
        raw = filter_universe(source.base)
        identity = filter_identity(raw, input_id=input_id)
        universe = _universe(raw, identity)
        selected = frozenset(source.iter_indices())
        decisions = tuple(
            universe.global_index(index) in selected for index in range(len(universe))
        )
        return raw, (DecisionSet(universe, decisions).select(True),), universe
    return (
        filter_base(source, input_id=input_id, allow_logical=True),
        (),
        None,
    )


def _universe(
    dataset: FilterBase,
    identity: Mapping[str, Any],
) -> DatasetUniverse:
    if isinstance(dataset, SampleIdentity):
        return DatasetUniverse(dataset)
    namespace = filter_identity_key(identity)
    return DatasetUniverse(
        dataset,
        sample_identity=_IndexSampleIdentity(dataset, namespace),
    )


def _rebase_upstream(
    upstream: tuple[Selection, ...],
    universe: DatasetUniverse,
) -> tuple[Selection, ...]:
    return tuple(
        selection if selection.universe is universe else selection.rebase(universe)
        for selection in upstream
    )


def _labels(values: Any) -> frozenset[str] | None:
    if values is None:
        return None
    from .api import normalized_labels

    return frozenset(normalized_labels(values))


def _ready_selection(
    universe: DatasetUniverse,
    partitions: Mapping[str, Sequence[int]],
    labels: frozenset[str] | None,
) -> StaticSelection:
    decisions: list[str | None] = [None] * len(universe)
    positions = _global_positions(universe)
    for partition_label, indexes in partitions.items():
        for global_index in indexes:
            try:
                position = positions[int(global_index)]
            except KeyError as exc:
                raise ValueError(
                    f"cached filter index is outside the dataset universe: {global_index}"
                ) from exc
            if decisions[position] is not None:
                raise ValueError(f"cached filter index is duplicated: {global_index}")
            decisions[position] = partition_label
    if any(decision is None for decision in decisions):
        raise RuntimeError("ready filter cache does not cover the complete universe.")
    selected = (
        frozenset(cast(str, decision) for decision in decisions)
        if labels is None
        else labels
    )
    return DecisionSet(
        universe,
        tuple(cast(str, decision) for decision in decisions),
    ).select(*selected)


def _publish_partitions(
    selection: _LiveSelection,
    partitions: Mapping[str, Sequence[int]],
) -> None:
    selection.publish(_FilterChunk(partitions=partitions, metrics=()))


def _global_positions(universe: DatasetUniverse) -> dict[int, int]:
    positions: dict[int, int] = {}
    for index in range(len(universe)):
        global_index = universe.global_index(index)
        if global_index in positions:
            raise ValueError(f"duplicate global_index {global_index} in universe.")
        positions[global_index] = index
    return positions


def _position(index: int, length: int) -> int:
    position = operator.index(index)
    if position < 0:
        position += length
    if position < 0 or position >= length:
        raise IndexError("universe index out of range")
    return position


__all__ = ["FilterRun"]
