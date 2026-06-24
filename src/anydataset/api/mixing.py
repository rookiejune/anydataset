from __future__ import annotations

from collections import deque
import queue
import random
from dataclasses import dataclass
import threading
from typing import Iterator, Sequence

from ..samples import Sample


@dataclass
class SampleStream:
    name: str
    iterator: Iterator[Sample]
    weight: float = 1.0


class WeightedDatasetMixer:
    def __init__(self, streams: Sequence[SampleStream], seed: int | None = None):
        self._streams = _active_streams(streams)
        self._seed = seed

    def __iter__(self) -> Iterator[Sample]:
        rng = random.Random(self._seed)
        active = list(self._streams)

        while active:
            weights = [stream.weight for stream in active]
            selected = rng.choices(range(len(active)), weights=weights, k=1)[0]
            stream = active[selected]
            try:
                yield next(stream.iterator)
            except StopIteration:
                active.pop(selected)


@dataclass(frozen=True)
class _StreamItem:
    stream_index: int
    sample: Sample


@dataclass(frozen=True)
class _StreamDone:
    stream_index: int


@dataclass(frozen=True)
class _StreamError:
    stream_index: int
    error: BaseException


class PrefetchingDatasetMixer:
    def __init__(
        self,
        streams: Sequence[SampleStream],
        seed: int | None = None,
        queue_size: int = 256,
        wait_timeout: float | None = None,
    ):
        if queue_size <= 0:
            raise ValueError("queue_size must be positive.")
        self._streams = _active_streams(streams)
        self._seed = seed
        self._queue_size = queue_size
        self._wait_timeout = wait_timeout

    def __iter__(self) -> Iterator[Sample]:
        rng = random.Random(self._seed)
        output = queue.Queue(maxsize=self._queue_size)
        stop_event = threading.Event()
        threads = [
            threading.Thread(
                target=_produce_stream_events,
                args=(stream_index, stream.iterator, output, stop_event),
                daemon=True,
            )
            for stream_index, stream in enumerate(self._streams)
        ]
        for thread in threads:
            thread.start()

        active = [True for _ in self._streams]
        buffers = [deque() for _ in self._streams]

        try:
            while any(active) or any(buffers):
                _drain_available_events(output, active, buffers)
                ready_indices = [
                    index for index, buffer in enumerate(buffers) if buffer
                ]
                if not ready_indices:
                    if not any(active):
                        break
                    event = _get_next_event(output, self._wait_timeout)
                    _handle_stream_event(event, active, buffers)
                    continue

                selected = _choose_ready_stream(rng, self._streams, ready_indices)
                yield buffers[selected].popleft()
        finally:
            stop_event.set()
            for thread in threads:
                thread.join(timeout=0.1)


def _active_streams(streams: Sequence[SampleStream]) -> list[SampleStream]:
    if not streams:
        raise ValueError("WeightedDatasetMixer requires at least one stream.")
    for stream in streams:
        if stream.weight < 0:
            raise ValueError(f"Stream {stream.name!r} has negative weight.")
    active = [stream for stream in streams if stream.weight > 0]
    if not active:
        raise ValueError("At least one stream must have a positive weight.")
    return active


def _produce_stream_events(
    stream_index: int,
    iterator: Iterator[Sample],
    output,
    stop_event: threading.Event,
) -> None:
    try:
        for sample in iterator:
            if stop_event.is_set():
                return
            _put_stream_event(output, _StreamItem(stream_index, sample), stop_event)
    except BaseException as exc:
        _put_stream_event(output, _StreamError(stream_index, exc), stop_event)
    finally:
        _put_stream_event(output, _StreamDone(stream_index), stop_event)


def _put_stream_event(output, event, stop_event: threading.Event) -> None:
    while not stop_event.is_set():
        try:
            output.put(event, timeout=0.1)
            return
        except queue.Full:
            continue


def _get_next_event(output, wait_timeout: float | None):
    try:
        return output.get(timeout=wait_timeout)
    except queue.Empty as exc:
        raise TimeoutError("Timed out waiting for a prefetched sample.") from exc


def _drain_available_events(output, active: list[bool], buffers: list[deque]) -> None:
    while True:
        try:
            event = output.get_nowait()
        except queue.Empty:
            return
        _handle_stream_event(event, active, buffers)


def _handle_stream_event(event, active: list[bool], buffers: list[deque]) -> None:
    if isinstance(event, _StreamItem):
        buffers[event.stream_index].append(event.sample)
        return
    if isinstance(event, _StreamDone):
        active[event.stream_index] = False
        return
    if isinstance(event, _StreamError):
        raise event.error
    raise TypeError(f"Unknown stream event: {event!r}")


def _choose_ready_stream(
    rng: random.Random,
    streams: Sequence[SampleStream],
    ready_indices: Sequence[int],
) -> int:
    weights = [streams[index].weight for index in ready_indices]
    return rng.choices(list(ready_indices), weights=weights, k=1)[0]
