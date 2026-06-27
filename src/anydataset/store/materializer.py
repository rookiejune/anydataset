from __future__ import annotations

import logging
import multiprocessing
import os
import pickle
import shutil
import time
import traceback
from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path
from queue import Empty
from tempfile import TemporaryDirectory
from typing import Any, Literal, cast

from .._sharding import validate_shard
from ..types.item import (
    AudioItem,
    AudioView,
    ImageItem,
    ImageView,
    Item,
    Modality,
    Sample,
    TextItem,
    TextView,
    View,
)
from ..view import Provider, ViewProvider
from .parts import DatasetPartWriter, commit_store_parts
from .writer import DEFAULT_MAX_SHARD_SAMPLES, DatasetWriter

type DatasetFactory = Callable[[], Any]
type ProviderFactory = Callable[[str], Provider]
type Devices = Literal["auto"] | str | Iterable[str]

_PROGRESS_INTERVAL = 1.0


@dataclass
class ViewMaterializer:
    output_dir: str | Path
    split: str | None = None
    max_shard_samples: int = DEFAULT_MAX_SHARD_SAMPLES

    @property
    def dataset_id(self) -> str:
        return _dataset_id(self.output_dir)

    def write(
        self,
        *,
        dataset_factory: DatasetFactory,
        provider_factory: ProviderFactory,
        devices: Devices = "auto",
    ) -> Path:
        resolved = resolve_devices(devices)
        if len(resolved) == 1:
            return self._write_single(dataset_factory(), provider_factory(resolved[0]))
        return self.write_parallel(
            dataset_factory=dataset_factory,
            provider_factory=provider_factory,
            devices=resolved,
        )

    def write_parallel(
        self,
        *,
        dataset_factory: DatasetFactory,
        provider_factory: ProviderFactory,
        devices: Devices = "auto",
    ) -> Path:
        resolved = resolve_devices(devices)
        if len(resolved) == 1:
            return self._write_single(dataset_factory(), provider_factory(resolved[0]))

        _validate_spawn_factory("dataset_factory", dataset_factory)
        _validate_spawn_factory("provider_factory", provider_factory)
        output_dir = _prepare_parallel_output_dir(Path(self.output_dir).expanduser())
        logs_dir = output_dir / "logs"
        logs_dir.mkdir(parents=True, exist_ok=True)
        with TemporaryDirectory(
            prefix=f".{output_dir.name}-parts-",
            dir=str(output_dir.parent),
        ) as tmpdir:
            parts_dir = Path(tmpdir)
            self._run_parallel_parts(
                dataset_factory=dataset_factory,
                provider_factory=provider_factory,
                devices=resolved,
                parts_dir=parts_dir,
                logs_dir=logs_dir,
            )
            log_backup = self._stash_logs(logs_dir)
            try:
                result = self.commit_parts(parts_dir)
            finally:
                self._restore_logs(log_backup)
            return result

    def _write_single(self, dataset: Iterable[Sample], provider: Provider) -> Path:
        return DatasetWriter(
            self.output_dir,
            dataset_id=self.dataset_id,
            split=self.split,
            max_shard_samples=self.max_shard_samples,
        ).write(self._samples(dataset, provider))

    def write_part(
        self,
        dataset: Any,
        provider: Provider,
        *,
        parts_dir: str | Path,
        num_shards: int,
        shard_id: int,
    ) -> Path:
        return DatasetPartWriter(
            Path(parts_dir) / f"part-{shard_id:05d}",
            dataset_id=self.dataset_id,
            split=self.split,
            shard_id=shard_id,
            num_shards=num_shards,
            max_shard_samples=self.max_shard_samples,
        ).write(
            (
                (index, _with_providers(sample, provider))
                for index, sample in iter_indexed_shard(dataset, num_shards, shard_id)
            )
        )

    def commit_parts(self, parts_dir: str | Path) -> Path:
        return commit_store_parts(
            self.output_dir,
            parts_dir,
            dataset_id=self.dataset_id,
            split=self.split,
        )

    def _samples(self, dataset: Iterable[Sample], provider: Provider):
        for sample in dataset:
            yield _with_providers(sample, provider)

    def _samples_with_progress(
        self,
        dataset: Any,
        provider: Provider,
        *,
        num_shards: int,
        shard_id: int,
        progress: multiprocessing.Queue,
    ):
        pending = 0
        last_flush = time.monotonic()
        try:
            for index, sample in iter_indexed_shard(dataset, num_shards, shard_id):
                yield index, _with_providers(sample, provider)
                pending += 1
                now = time.monotonic()
                if now - last_flush >= _PROGRESS_INTERVAL:
                    _put_progress(progress, _Progress(shard_id, pending, False, None))
                    pending = 0
                    last_flush = now
        finally:
            if pending:
                _put_progress(progress, _Progress(shard_id, pending, False, None))

    def _run_parallel_parts(
        self,
        *,
        dataset_factory: DatasetFactory,
        provider_factory: ProviderFactory,
        devices: tuple[str, ...],
        parts_dir: Path,
        logs_dir: Path,
    ) -> None:
        context = multiprocessing.get_context("spawn")
        progress = context.Queue()
        total = None
        workers = [
            context.Process(
                target=_materialize_worker,
                args=(
                    _WorkerConfig(
                        output_dir=Path(self.output_dir),
                        dataset_id=self.dataset_id,
                        split=self.split,
                        max_shard_samples=self.max_shard_samples,
                        parts_dir=parts_dir,
                        logs_dir=logs_dir,
                        device=device,
                        num_shards=len(devices),
                        shard_id=shard_id,
                    ),
                    dataset_factory,
                    provider_factory,
                    progress,
                ),
                name=f"anydataset-materialize-{shard_id}",
            )
            for shard_id, device in enumerate(devices)
        ]
        for worker in workers:
            worker.start()
        try:
            _watch_workers(workers, progress, total=total)
        except Exception:
            for worker in workers:
                if worker.is_alive():
                    worker.terminate()
            for worker in workers:
                worker.join()
            raise
        for worker in workers:
            worker.join()

        failed = [worker for worker in workers if worker.exitcode != 0]
        if failed:
            details = ", ".join(
                f"{worker.name} exited {worker.exitcode}" for worker in failed
            )
            raise RuntimeError(f"View materialization workers failed: {details}.")

    def _stash_logs(self, logs_dir: Path) -> Path | None:
        if not logs_dir.exists():
            return None
        output_dir = Path(self.output_dir).expanduser()
        backup = output_dir.parent / f".{output_dir.name}-logs-{os.getpid()}"
        if backup.exists():
            shutil.rmtree(backup)
        os.replace(logs_dir, backup)
        return backup

    def _restore_logs(self, backup: Path | None) -> None:
        if backup is None:
            return
        output_dir = Path(self.output_dir).expanduser()
        target = output_dir / "logs"
        if target.exists():
            shutil.rmtree(target)
        os.replace(backup, target)


@dataclass(frozen=True)
class _WorkerConfig:
    output_dir: Path
    dataset_id: str
    split: str | None
    max_shard_samples: int
    parts_dir: Path
    logs_dir: Path
    device: str
    num_shards: int
    shard_id: int


@dataclass(frozen=True)
class _Progress:
    shard_id: int
    samples: int
    done: bool
    error: str | None


def resolve_devices(devices: Devices) -> tuple[str, ...]:
    if devices == "auto":
        count = _cuda_device_count()
        if count > 0:
            return tuple(f"cuda:{index}" for index in range(count))
        return ("cpu",)
    if isinstance(devices, str):
        resolved = (devices,)
    else:
        resolved = tuple(devices)
    if not resolved:
        raise ValueError("devices must not be empty.")
    return resolved


def _cuda_device_count() -> int:
    try:
        import torch
    except ImportError:
        return 0
    if not torch.cuda.is_available():
        return 0
    return torch.cuda.device_count()


def _validate_spawn_factory(name: str, factory: Callable[..., Any]) -> None:
    try:
        pickle.dumps(factory)
    except Exception as exc:
        raise TypeError(
            f"{name} must be picklable for multi-device materialization."
        ) from exc


def _prepare_parallel_output_dir(output_dir: Path) -> Path:
    if output_dir.exists():
        if not output_dir.is_dir():
            raise ValueError(f"Target path exists and is not a directory: {output_dir}")
        entries = [entry for entry in output_dir.iterdir() if entry.name != "logs"]
        if entries:
            raise ValueError(f"Target directory must be empty: {output_dir}")
        logs_dir = output_dir / "logs"
        if logs_dir.exists():
            shutil.rmtree(logs_dir)
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(exist_ok=True)
    return output_dir


def _materialize_worker(
    config: _WorkerConfig,
    dataset_factory: DatasetFactory,
    provider_factory: ProviderFactory,
    progress: multiprocessing.Queue,
) -> None:
    logger = _worker_logger(config.logs_dir, config.shard_id)
    logger.info(
        "starting shard %s/%s on %s",
        config.shard_id,
        config.num_shards,
        config.device,
    )
    try:
        dataset = dataset_factory()
        provider = provider_factory(config.device)
        DatasetPartWriter(
            config.parts_dir / f"part-{config.shard_id:05d}",
            dataset_id=config.dataset_id,
            split=config.split,
            shard_id=config.shard_id,
            num_shards=config.num_shards,
            max_shard_samples=config.max_shard_samples,
        ).write(
            ViewMaterializer(
                config.output_dir,
                split=config.split,
                max_shard_samples=config.max_shard_samples,
            )._samples_with_progress(
                dataset,
                provider,
                num_shards=config.num_shards,
                shard_id=config.shard_id,
                progress=progress,
            )
        )
    except Exception:
        error = traceback.format_exc()
        logger.error("worker failed\n%s", error)
        _put_progress(progress, _Progress(config.shard_id, 0, True, error))
        raise
    logger.info("finished shard %s", config.shard_id)
    _put_progress(progress, _Progress(config.shard_id, 0, True, None))


def _worker_logger(logs_dir: Path, shard_id: int) -> logging.Logger:
    logs_dir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger(f"anydataset.materializer.{os.getpid()}.{shard_id}")
    logger.setLevel(logging.INFO)
    logger.propagate = False
    handler = logging.FileHandler(
        logs_dir / f"part-{shard_id:05d}.log",
        encoding="utf-8",
    )
    handler.setFormatter(
        logging.Formatter("%(asctime)s %(levelname)s %(processName)s %(message)s")
    )
    logger.handlers.clear()
    logger.addHandler(handler)
    return logger


def _watch_workers(
    workers: list[multiprocessing.Process],
    progress: multiprocessing.Queue,
    *,
    total: int | None,
) -> None:
    done = 0
    with _progress_bar(total) as bar:
        while done < len(workers):
            try:
                message = progress.get(timeout=0.2)
            except Empty:
                if _dead_worker(workers):
                    raise RuntimeError("View materialization worker exited early.")
                continue
            if not isinstance(message, _Progress):
                continue
            if message.samples:
                bar.update(message.samples)
            if message.done:
                done += 1
                if message.error is not None:
                    raise RuntimeError(
                        f"View materialization worker {message.shard_id} failed.\n"
                        f"{message.error}"
                    )


def _dead_worker(workers: list[multiprocessing.Process]) -> bool:
    return any(worker.exitcode not in (None, 0) for worker in workers)


def _progress_bar(total: int | None):
    try:
        from tqdm.auto import tqdm
    except ImportError:
        return _NullProgressBar()
    return tqdm(total=total, unit="sample", desc="materialize views")


class _NullProgressBar:
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        return None

    def update(self, count: int) -> None:
        return None


def _put_progress(progress: multiprocessing.Queue, message: _Progress) -> None:
    progress.put(message)


def iter_indexed_shard(
    dataset: Any,
    num_shards: int,
    shard_id: int,
) -> Iterator[tuple[int, Sample]]:
    validate_shard(num_shards, shard_id)
    iter_indexed = getattr(dataset, "iter_indexed_shard", None)
    if callable(iter_indexed):
        yield from iter_indexed(num_shards, shard_id)
        return

    iter_shard = getattr(dataset, "iter_shard", None)
    if callable(iter_shard):
        for local_index, sample in enumerate(iter_shard(num_shards, shard_id)):
            yield shard_id + local_index * num_shards, sample
        return

    native_shard = getattr(dataset, "shard", None)
    if callable(native_shard):
        for local_index, sample in enumerate(
            native_shard(num_shards=num_shards, index=shard_id)
        ):
            yield shard_id + local_index * num_shards, sample
        return

    if hasattr(dataset, "__len__") and hasattr(dataset, "__getitem__"):
        for index in range(shard_id, len(dataset), num_shards):
            yield index, dataset[index]
        return

    for index, sample in enumerate(dataset):
        if index % num_shards == shard_id:
            yield index, sample


def _with_providers(
    sample: Sample,
    provider: Provider,
) -> Sample:
    modality = _output_modality(provider.output)
    return {
        ref: _with_provider(item, provider)
        for ref, item in sample.items()
        if ref[1] is modality
    }


def _with_provider(
    item: Item,
    provider: Provider,
) -> Item:
    match item:
        case AudioItem():
            provider = _audio_provider(provider)
            return _with_view(
                item,
                provider.output,
                provider(item.views),
            )
        case ImageItem():
            provider = _image_provider(provider)
            return _with_view(
                item,
                provider.output,
                provider(item.views),
            )
        case TextItem():
            provider = _text_provider(provider)
            return _with_view(
                item,
                provider.output,
                provider(item.views),
            )
    raise TypeError(f"Unsupported materializer item: {type(item).__name__}.")


def _with_view(
    item: Item,
    view: View,
    value: Any,
) -> Item:
    match item:
        case AudioItem():
            if not isinstance(view, AudioView):
                raise TypeError("audio item materializer output must be an AudioView.")
            return AudioItem(
                views=_views(view, value),
                meta=item.meta,
            )
        case ImageItem():
            if not isinstance(view, ImageView):
                raise TypeError("image item materializer output must be an ImageView.")
            return ImageItem(
                views=_views(view, value),
                meta=item.meta,
            )
        case TextItem():
            if not isinstance(view, TextView):
                raise TypeError("text item materializer output must be a TextView.")
            return TextItem(
                views=_views(view, value),
                meta=item.meta,
            )
    raise TypeError(f"Unsupported materializer item: {type(item).__name__}.")


def _audio_provider(provider: Provider) -> ViewProvider[AudioView]:
    if not isinstance(provider.output, AudioView):
        raise TypeError("audio item materializer output must be an AudioView.")
    return cast(ViewProvider[AudioView], provider)


def _image_provider(provider: Provider) -> ViewProvider[ImageView]:
    if not isinstance(provider.output, ImageView):
        raise TypeError("image item materializer output must be an ImageView.")
    return cast(ViewProvider[ImageView], provider)


def _text_provider(provider: Provider) -> ViewProvider[TextView]:
    if not isinstance(provider.output, TextView):
        raise TypeError("text item materializer output must be a TextView.")
    return cast(ViewProvider[TextView], provider)


def _output_modality(view: View) -> Modality:
    if isinstance(view, AudioView):
        return Modality.AUDIO
    if isinstance(view, ImageView):
        return Modality.IMAGE
    if isinstance(view, TextView):
        return Modality.TEXT
    raise TypeError("materializer output must be an AudioView, ImageView, or TextView.")


def _views[ViewT](view: ViewT, value: Any) -> dict[ViewT, Any]:
    return {view: value}


def _dataset_id(output_dir: str | Path) -> str:
    return Path(output_dir).expanduser().name or "dataset"
