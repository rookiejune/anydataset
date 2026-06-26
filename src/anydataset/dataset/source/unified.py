from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Any, Iterator

from ... import types
from ...store.manifest import ViewRef
from ...store.reader import StoreDataset, read_store_dataset
from ...types import item
from ..errors import MissingModalityError, RoleLike

if TYPE_CHECKING:
    from ...cache import CacheManifest


class UnifiedDatasetSource:
    default_task = types.Task.AUDIO_CODEC
    map_style = True

    def __init__(self, views: Sequence[ViewRef] | None = None):
        self.views = None if views is None else tuple(views)

    def prepare(self, spec: types.Spec, cache: CacheManifest) -> StoreDataset:
        return read_store_dataset(
            spec.path,
            split=spec.split,
            cache_path=cache.cache_path,
            views=self.views,
        )

    def iter_samples(self, state: StoreDataset) -> Iterator[item.Sample]:
        yield from state

    def num_samples(self, state: StoreDataset) -> int:
        return len(state)

    def sample_at(self, state: StoreDataset, index: int) -> item.Sample:
        return state.sample_at(index)

    def audio(
        self,
        row: Mapping[tuple[item.Role, item.Modality], Any],
        role: RoleLike = None,
    ) -> item.AudioItem:
        ref = (_role_value(role), item.Modality.AUDIO)
        audio = row.get(ref)
        if audio is None:
            raise MissingModalityError(item.Modality.AUDIO, role)
        if not isinstance(audio, item.AudioItem):
            raise TypeError("audio modality must be an AudioItem.")
        return audio

    def text(
        self,
        row: Mapping[tuple[item.Role, item.Modality], Any],
        role: RoleLike = None,
    ) -> item.TextItem:
        ref = (_role_value(role), item.Modality.TEXT)
        text = row.get(ref)
        if text is None:
            raise MissingModalityError(item.Modality.TEXT, role)
        if not isinstance(text, item.TextItem):
            raise TypeError("text modality must be a TextItem.")
        return text


def _role_value(role: RoleLike) -> item.Role:
    if role is None:
        return item.Role.DEFAULT
    if isinstance(role, item.Role):
        return role
    return item.Role(role)
