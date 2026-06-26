from __future__ import annotations

import hashlib
import json
import os
import shutil
import threading
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any

from torch.utils.data import Dataset

from .._sharding import validate_shard
from ..cache import CacheManifest, FileLock
from ..store.jsonio import read_json, write_json
from ..types import Spec
from ..types.item import Sample
from .abc import AnyDataset


type FilterPredicate = Callable[[Sample], bool]
type JsonScalar = str | int | float | bool | None
type JsonConfig = tuple[tuple[str, JsonConfigValue], ...]
type JsonConfigValue = JsonScalar | JsonConfig | tuple[JsonConfigValue, ...]
type FrozenJsonValue = JsonScalar | MappingProxyType[str, FrozenJsonValue] | tuple[FrozenJsonValue, ...]
type FrozenJsonItems = tuple[tuple[str, _FrozenJsonValue], ...]
type _FrozenJsonValue = JsonScalar | _FrozenJsonObject | tuple[_FrozenJsonValue, ...]


@dataclass(frozen=True)
class _FrozenJsonObject:
    items: FrozenJsonItems


class FilterRule:
    __slots__ = ("_config", "_config_map", "_name", "_predicate", "_version")

    def __init__(
        self,
        name: str,
        version: str,
        config: JsonConfig,
        predicate: FilterPredicate,
    ) -> None:
        _validate_string("name", name)
        _validate_string("version", version)
        if not isinstance(config, tuple):
            raise TypeError("config must be tuple pairs.")
        if not callable(predicate):
            raise TypeError("predicate must be callable.")
        frozen = _freeze_json_object(config)
        _json_text(_plain_json_object(frozen))
        self._name = name
        self._version = version
        self._config = frozen
        self._config_map = _json_object_view(frozen)
        self._predicate = predicate

    def __repr__(self) -> str:
        return (
            "FilterRule("
            f"name={self.name!r}, "
            f"version={self.version!r}, "
            f"config={self.config!r}, "
            f"predicate={self.predicate!r}"
            ")"
        )

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, FilterRule):
            return NotImplemented
        return (
            self.name,
            self.version,
            self._config,
            self.predicate,
        ) == (
            other.name,
            other.version,
            other._config,
            other.predicate,
        )

    def __hash__(self) -> int:
        return hash((self.name, self.version, self._config, self.predicate))

    @property
    def name(self) -> str:
        return self._name

    @property
    def version(self) -> str:
        return self._version

    @property
    def predicate(self) -> FilterPredicate:
        return self._predicate

    @property
    def config(self) -> MappingProxyType[str, FrozenJsonValue]:
        return self._config_map

    @property
    def identity(self) -> Mapping[str, Any]:
        return {
            "name": self.name,
            "version": self.version,
            "config": _plain_json_object(self._config),
        }

    @property
    def id(self) -> str:
        return _stable_hash(self.identity)


class FilteredDataset(Dataset):
    __slots__ = ("_base", "_cache_path", "_indices", "_rule")

    def __init__(
        self,
        base: AnyDataset,
        indices: Sequence[int],
        rule: FilterRule,
        cache_path: Path,
    ) -> None:
        if not isinstance(base, AnyDataset):
            raise TypeError("base must be an AnyDataset.")
        if not isinstance(rule, FilterRule):
            raise TypeError("rule must be a FilterRule.")
        self._base = base
        self._indices = tuple(indices)
        self._rule = rule
        self._cache_path = Path(cache_path)

    def __repr__(self) -> str:
        return (
            "FilteredDataset("
            f"base={self.base!r}, "
            f"indices={self.indices!r}, "
            f"rule={self.rule!r}, "
            f"cache_path={self.cache_path!r}"
            ")"
        )

    @property
    def base(self) -> AnyDataset:
        return self._base

    @property
    def indices(self) -> tuple[int, ...]:
        return self._indices

    @property
    def rule(self) -> FilterRule:
        return self._rule

    @property
    def cache_path(self) -> Path:
        return self._cache_path

    @property
    def spec(self) -> Spec:
        return self.base.spec

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, index: int) -> Sample:
        return self.base[self.indices[index]]

    def iter_shard(self, num_shards: int, shard_id: int):
        validate_shard(num_shards, shard_id)
        for index in range(shard_id, len(self), num_shards):
            yield self[index]


def cached_filter(dataset: AnyDataset, rule: FilterRule) -> FilteredDataset:
    if not isinstance(dataset, AnyDataset):
        raise TypeError("dataset must be an AnyDataset.")
    if not isinstance(rule, FilterRule):
        raise TypeError("rule must be a FilterRule.")

    cache = dataset.cache_manager.prepare(dataset.spec)
    base_count = len(dataset)
    expected = _metadata(dataset.spec, base_count, rule)
    cache_path = _filter_path(cache, rule)

    if _is_ready(cache_path, expected):
        return FilteredDataset(dataset, _read_indices(cache_path), rule, cache_path)

    lock_path = _lock_path(cache, rule)
    with FileLock(lock_path):
        if _is_ready(cache_path, expected):
            return FilteredDataset(dataset, _read_indices(cache_path), rule, cache_path)
        indices = _collect_indices(dataset, rule)
        _write_cache(cache_path, expected, indices)
        return FilteredDataset(dataset, indices, rule, cache_path)


def _metadata(spec: Spec, base_count: int, rule: FilterRule) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "base": {
            "spec_id": spec.id,
            "sample_count": base_count,
        },
        "rule": dict(rule.identity),
    }


def _filter_path(cache: CacheManifest, rule: FilterRule) -> Path:
    return cache.cache_path / "filters" / rule.id


def _lock_path(cache: CacheManifest, rule: FilterRule) -> Path:
    return cache.cache_path / "filters" / f".{rule.id}.lock"


def _is_ready(path: Path, expected: Mapping[str, Any]) -> bool:
    if not (path / ".ready").is_file():
        return False
    if not (path / "indices.parquet").is_file():
        return False
    metadata_path = path / "rule.json"
    if not metadata_path.is_file():
        return False
    return read_json(metadata_path) == expected


def _collect_indices(dataset: AnyDataset, rule: FilterRule) -> tuple[int, ...]:
    indices: list[int] = []
    for index in range(len(dataset)):
        keep = rule.predicate(dataset[index])
        if not isinstance(keep, bool):
            raise TypeError("filter predicate must return bool.")
        if keep:
            indices.append(index)
    return tuple(indices)


def _write_cache(
    path: Path,
    metadata: Mapping[str, Any],
    indices: Sequence[int],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.parent / f".{path.name}.{os.getpid()}.{threading.get_ident()}.tmp"
    if tmp.exists():
        shutil.rmtree(tmp)
    tmp.mkdir(parents=True)
    try:
        write_json(tmp / "rule.json", dict(metadata))
        _write_indices(tmp / "indices.parquet", indices)
        (tmp / ".ready").write_text("ready\n", encoding="utf-8")
        if path.exists():
            shutil.rmtree(path)
        os.replace(tmp, path)
    except Exception:
        if tmp.exists():
            shutil.rmtree(tmp)
        raise


def _read_indices(path: Path) -> tuple[int, ...]:
    _, pq = _pyarrow()
    table = pq.read_table(path / "indices.parquet", columns=["index"])
    return tuple(int(index) for index in table.column("index").to_pylist())


def _write_indices(path: Path, indices: Sequence[int]) -> None:
    pa, pq = _pyarrow()
    table = pa.Table.from_pydict(
        {"index": indices},
        schema=pa.schema([("index", pa.int64())]),
    )
    pq.write_table(table, path)


def _stable_hash(value: Mapping[str, Any]) -> str:
    return hashlib.sha256(_json_text(value).encode("utf-8")).hexdigest()[:16]


def _json_text(value: Mapping[str, Any]) -> str:
    try:
        return json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":"))
    except (TypeError, ValueError) as exc:
        raise TypeError("filter rule config must be JSON serializable.") from exc


def _plain_json_object(value: _FrozenJsonObject) -> dict[str, Any]:
    return {key: _plain_json_value(item) for key, item in value.items}


def _plain_json_value(value: _FrozenJsonValue) -> Any:
    if isinstance(value, _FrozenJsonObject):
        return _plain_json_object(value)
    if isinstance(value, tuple):
        return [_plain_json_value(item) for item in value]
    return value


def _json_object_view(
    value: _FrozenJsonObject,
) -> MappingProxyType[str, FrozenJsonValue]:
    return MappingProxyType(
        {key: _json_value_view(item) for key, item in value.items}
    )


def _json_value_view(value: _FrozenJsonValue) -> FrozenJsonValue:
    if isinstance(value, _FrozenJsonObject):
        return _json_object_view(value)
    if isinstance(value, tuple):
        return tuple(_json_value_view(item) for item in value)
    return value


def _freeze_json_object(value: JsonConfig) -> _FrozenJsonObject:
    items: list[tuple[str, _FrozenJsonValue]] = []
    seen: set[str] = set()
    for pair in value:
        if not isinstance(pair, tuple) or len(pair) != 2:
            raise TypeError("filter rule config must use (key, value) pairs.")
        key, item = pair
        if not isinstance(key, str):
            raise TypeError("filter rule config keys must be strings.")
        if key in seen:
            raise ValueError("filter rule config keys must be unique.")
        seen.add(key)
        items.append((key, _freeze_json_value(item)))
    return _FrozenJsonObject(tuple(sorted(items, key=lambda pair: pair[0])))


def _freeze_json_value(value: JsonConfigValue) -> _FrozenJsonValue:
    if isinstance(value, str | int | float | bool) or value is None:
        return value
    if _looks_like_json_object(value):
        return _freeze_json_object(value)
    if isinstance(value, tuple):
        return tuple(_freeze_json_value(item) for item in value)
    raise TypeError("filter rule config must be JSON serializable.")


def _looks_like_json_object(value: object) -> bool:
    if not isinstance(value, tuple):
        return False
    return all(
        isinstance(item, tuple) and len(item) == 2 and isinstance(item[0], str)
        for item in value
    )


def _validate_string(name: str, value: str) -> None:
    if not isinstance(value, str):
        raise TypeError(f"{name} must be a string.")
    if value == "":
        raise ValueError(f"{name} must not be empty.")


def _pyarrow():
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise ImportError("Cached filters require pyarrow.") from exc
    return pa, pq
