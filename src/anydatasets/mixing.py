from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Iterator, Sequence

from .samples import Sample


@dataclass
class SampleStream:
    name: str
    iterator: Iterator[Sample]
    weight: float = 1.0


class WeightedDatasetMixer:
    def __init__(self, streams: Sequence[SampleStream], seed: int | None = None):
        if not streams:
            raise ValueError("WeightedDatasetMixer requires at least one stream.")
        for stream in streams:
            if stream.weight < 0:
                raise ValueError(f"Stream {stream.name!r} has negative weight.")
        if all(stream.weight == 0 for stream in streams):
            raise ValueError("At least one stream must have a positive weight.")

        self._streams = list(streams)
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
