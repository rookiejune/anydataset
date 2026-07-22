from __future__ import annotations

import random
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from typing import Any

from torch.utils.data import DataLoader, Sampler

from .._sharding import runtime_shard, validate_shard
from ..types.item import AudioView, ImageView, Modality, Role, TextView, View
from .reader import StoreDataset

ViewRef = tuple[Role, Modality, View]


@dataclass(frozen=True)
class _Group:
    start: int
    stop: int

    def __len__(self) -> int:
        return self.stop - self.start


class StoreLocalBatchSampler(Sampler[list[int]]):
    def __init__(
        self,
        dataset: StoreDataset,
        *,
        batch_size: int,
        views: Iterable[ViewRef] | None = None,
        drop_last: bool = False,
        shuffle: bool = False,
        seed: int = 0,
        epoch: int = 0,
        num_replicas: int | None = None,
        rank: int | None = None,
    ) -> None:
        if not isinstance(dataset, StoreDataset):
            raise TypeError("dataset must be a StoreDataset.")
        self.dataset = dataset
        self.batch_size = _positive_int("batch_size", batch_size)
        self.views = _view_refs(dataset, views)
        if not isinstance(drop_last, bool):
            raise TypeError("drop_last must be a bool.")
        if not isinstance(shuffle, bool):
            raise TypeError("shuffle must be a bool.")
        self.drop_last = drop_last
        self.shuffle = shuffle
        self.seed = _int("seed", seed)
        self.epoch = _non_negative_int("epoch", epoch)
        self.num_replicas, self.rank = _rank(num_replicas, rank)
        self._groups: tuple[_Group, ...] | None = None

    def __iter__(self) -> Iterator[list[int]]:
        for position, batch in enumerate(self._all_batches()):
            if position % self.num_replicas == self.rank:
                yield batch

    def __len__(self) -> int:
        count = sum(
            _batch_count(len(group), self.batch_size, self.drop_last)
            for group in self._payload_groups()
        )
        if self.rank >= count:
            return 0
        return (count - 1 - self.rank) // self.num_replicas + 1

    def set_epoch(self, epoch: int) -> None:
        self.epoch = _non_negative_int("epoch", epoch)

    def _all_batches(self) -> Iterator[list[int]]:
        groups = list(self._payload_groups())
        if self.shuffle:
            rng = random.Random(self.seed + self.epoch)
            rng.shuffle(groups)
        else:
            rng = None

        for group in groups:
            indexes = list(range(group.start, group.stop))
            if rng is not None:
                rng.shuffle(indexes)
            for start in range(0, len(indexes), self.batch_size):
                batch = indexes[start : start + self.batch_size]
                if len(batch) == self.batch_size or not self.drop_last:
                    yield batch

    def _payload_groups(self) -> tuple[_Group, ...]:
        if self._groups is None:
            self._groups = _payload_groups(self.dataset, self.views)
        return self._groups


def store_local_loader(
    dataset: StoreDataset,
    *,
    batch_size: int,
    views: Iterable[ViewRef] | None = None,
    drop_last: bool = False,
    shuffle: bool = False,
    seed: int = 0,
    epoch: int = 0,
    num_replicas: int | None = None,
    rank: int | None = None,
    **loader_kwargs: Any,
) -> DataLoader:
    conflicts = {
        "batch_sampler",
        "batch_size",
        "drop_last",
        "sampler",
        "shuffle",
    } & loader_kwargs.keys()
    if conflicts:
        names = ", ".join(sorted(conflicts))
        raise ValueError(f"store_local_loader owns loader kwargs: {names}.")

    return DataLoader(
        dataset,
        batch_sampler=StoreLocalBatchSampler(
            dataset,
            batch_size=batch_size,
            views=views,
            drop_last=drop_last,
            shuffle=shuffle,
            seed=seed,
            epoch=epoch,
            num_replicas=num_replicas,
            rank=rank,
        ),
        **loader_kwargs,
    )


def _payload_groups(
    dataset: StoreDataset,
    views: tuple[ViewRef, ...],
) -> tuple[_Group, ...]:
    groups: list[_Group] = []
    current_key: tuple[tuple[ViewRef, str], ...] | None = None
    start = 0

    for index in range(len(dataset)):
        key = _payload_key(dataset, views, index)
        if current_key is None:
            current_key = key
            start = index
            continue
        if key != current_key:
            groups.append(_Group(start=start, stop=index))
            current_key = key
            start = index

    if current_key is not None:
        groups.append(_Group(start=start, stop=len(dataset)))
    return tuple(groups)


def _payload_key(
    dataset: StoreDataset,
    views: tuple[ViewRef, ...],
    index: int,
) -> tuple[tuple[ViewRef, str], ...]:
    sample = dataset.samples[index]
    sample_refs = frozenset(ref for ref, _meta in sample.items)
    key: list[tuple[ViewRef, str]] = []
    for view in views:
        if view[:2] not in sample_refs:
            continue
        entry = dataset.views[view].entries_by_index[index]
        if entry is None:
            raise ValueError(
                f"Store view {view!r} is missing sample_index {index}."
            )
        key.append((view, entry.shard))
    if not key:
        raise ValueError(
            f"Store sample_index {index} has no payload shard for selected views."
        )
    return tuple(key)


def _view_refs(
    dataset: StoreDataset,
    views: Iterable[ViewRef] | None,
) -> tuple[ViewRef, ...]:
    selected = tuple(dataset.views if views is None else views)
    if not selected:
        raise ValueError("views must contain at least one store view.")
    available = frozenset(dataset.views)
    for view in selected:
        _validate_view_ref(view)
        if view not in available:
            raise KeyError(f"Store dataset does not contain view {view!r}.")
    return selected


def _validate_view_ref(view: object) -> None:
    if not isinstance(view, tuple) or len(view) != 3:
        raise TypeError("views entries must be (Role, Modality, View) tuples.")
    role, modality, key = view
    if not isinstance(role, Role):
        raise TypeError("store view role must be a Role.")
    if not isinstance(modality, Modality):
        raise TypeError("store view modality must be a Modality.")
    if modality is Modality.AUDIO:
        if not isinstance(key, AudioView):
            raise TypeError("audio store views must use AudioView values.")
        return
    if modality is Modality.IMAGE:
        if not isinstance(key, ImageView):
            raise TypeError("image store views must use ImageView values.")
        return
    if modality is Modality.TEXT:
        if not isinstance(key, TextView):
            raise TypeError("text store views must use TextView values.")
        return
    raise ValueError(f"Unsupported modality: {modality!r}.")


def _rank(
    num_replicas: int | None,
    rank: int | None,
) -> tuple[int, int]:
    if num_replicas is None and rank is None:
        shard = runtime_shard()
        return shard.rank_count, shard.rank_index
    if num_replicas is None or rank is None:
        raise ValueError("num_replicas and rank must be set together.")
    count = _positive_int("num_replicas", num_replicas)
    index = _non_negative_int("rank", rank)
    validate_shard(count, index)
    return count, index


def _batch_count(size: int, batch_size: int, drop_last: bool) -> int:
    full = size // batch_size
    if drop_last or size % batch_size == 0:
        return full
    return full + 1


def _positive_int(name: str, value: int) -> int:
    value = _int(name, value)
    if value <= 0:
        raise ValueError(f"{name} must be positive.")
    return value


def _non_negative_int(name: str, value: int) -> int:
    value = _int(name, value)
    if value < 0:
        raise ValueError(f"{name} must be non-negative.")
    return value


def _int(name: str, value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer.")
    return value
