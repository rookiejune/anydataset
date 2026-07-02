from __future__ import annotations

import argparse
import csv
import json
import multiprocessing
import os
import statistics
import tempfile
import time
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from anydataset import FilterRule, Spec
from anydataset.types import (
    AudioItem,
    AudioView,
    Modality,
    Role,
    Sample,
    TextItem,
    TextView,
)
from anydataset._parallel import (
    indexed_loader as runtime_indexed_loader,
    map_style_indexed_loader,
)
from anydataset._write_pipeline import BackgroundWriteSink
from anydataset.dataset.source.sharded_csv import ShardedCsvSource
from anydataset.store.parts import DatasetPartWriter, commit_store_parts
from anydataset.store.reader import read_store_dataset
from anydataset.store.writer import DatasetWriter
from torch.utils.data import DataLoader, Dataset


@dataclass(frozen=True)
class Measurement:
    seconds: float
    detail: dict[str, Any]


def main() -> None:
    args = parse_args()
    output = {
        "store_commit": run_repeated(
            lambda root: bench_store_commit(
                root,
                samples=args.store_samples,
                parts=args.store_parts,
                text_views=args.store_text_views,
                audio_views=args.store_audio_views,
                max_shard_samples=args.store_max_shard_samples,
            ),
            repeats=args.repeats,
        ),
        "sharded_csv": run_repeated(
            lambda root: bench_sharded_csv(
                root,
                csv_shards=args.csv_shards,
                files_per_shard=args.csv_files_per_shard,
                rows_per_file=args.csv_rows_per_file,
                num_shards=args.csv_num_shards,
                shard_id=args.csv_shard_id,
            ),
            repeats=args.repeats,
        ),
        "store_reader": bench_store_reader_variants(args),
        "indexed_loader": bench_indexed_loader_variants(args),
        "filter_parallel": run_repeated(
            lambda root: bench_filter_parallel(
                root,
                samples=args.filter_samples,
                devices=args.filter_devices,
                batch_size=args.filter_batch_size,
                num_workers=args.filter_num_workers,
                prefetch_factor=args.filter_prefetch_factor,
                commit_samples=args.filter_commit_samples,
                payload_bytes=args.filter_payload_bytes,
            ),
            repeats=args.repeats,
        ),
        "writer_pipeline": bench_writer_pipeline_variants(args),
    }
    print(json.dumps(output, ensure_ascii=True, indent=2, sort_keys=True))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark anydataset hot paths.")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--store-samples", type=int, default=2_000)
    parser.add_argument("--store-parts", type=int, default=4)
    parser.add_argument("--store-text-views", type=int, default=2)
    parser.add_argument("--store-audio-views", type=int, default=1)
    parser.add_argument("--store-max-shard-samples", type=int, default=512)
    parser.add_argument("--csv-shards", type=int, default=4)
    parser.add_argument("--csv-files-per-shard", type=int, default=4)
    parser.add_argument("--csv-rows-per-file", type=int, default=2_000)
    parser.add_argument("--csv-num-shards", type=int, default=8)
    parser.add_argument("--csv-shard-id", type=int, default=3)
    parser.add_argument("--indexed-samples", type=int, default=10_000)
    parser.add_argument("--indexed-batch-size", type=int, default=32)
    parser.add_argument("--indexed-num-workers", type=int, default=0)
    parser.add_argument("--indexed-prefetch-factor", type=int, default=None)
    parser.add_argument("--indexed-payload-bytes", type=int, default=32)
    parser.add_argument("--indexed-num-shards", type=int, default=1)
    parser.add_argument("--indexed-shard-id", type=int, default=0)
    parser.add_argument(
        "--indexed-variants",
        default="runtime,map_default,map_spawn,map_fork",
        help="Comma-separated variants: runtime,map_default,map_spawn,map_fork.",
    )
    parser.add_argument("--filter-samples", type=int, default=2_000)
    parser.add_argument("--filter-devices", type=int, default=2)
    parser.add_argument("--filter-batch-size", type=int, default=32)
    parser.add_argument("--filter-num-workers", type=int, default=0)
    parser.add_argument("--filter-prefetch-factor", type=int, default=None)
    parser.add_argument("--filter-commit-samples", type=int, default=100_000)
    parser.add_argument("--filter-payload-bytes", type=int, default=32)
    parser.add_argument("--writer-jobs", type=int, default=256)
    parser.add_argument("--writer-payload-bytes", type=int, default=64 * 1024)
    parser.add_argument("--writer-workers", type=int, default=1)
    parser.add_argument("--writer-prefetch", type=int, default=None)
    parser.add_argument("--writer-producer-delay-ms", type=float, default=0.0)
    parser.add_argument(
        "--writer-variants",
        default="inline,thread,process_spawn,process_fork",
        help=(
            "Comma-separated variants: inline,thread,process_spawn,process_fork."
        ),
    )
    return parser.parse_args()


def run_repeated(
    benchmark: Callable[[Path], Measurement],
    *,
    repeats: int,
) -> dict[str, Any]:
    if repeats <= 0:
        raise ValueError("repeats must be positive.")
    measurements = []
    for repeat in range(repeats):
        with tempfile.TemporaryDirectory(prefix="anydataset-bench-") as tmpdir:
            root = Path(tmpdir)
            measurement = benchmark(root)
            measurements.append(measurement)
    seconds = [measurement.seconds for measurement in measurements]
    return {
        "seconds": {
            "min": min(seconds),
            "median": statistics.median(seconds),
            "max": max(seconds),
            "runs": seconds,
        },
        "detail": measurements[-1].detail,
    }


def bench_store_commit(
    root: Path,
    *,
    samples: int,
    parts: int,
    text_views: int,
    audio_views: int,
    max_shard_samples: int,
) -> Measurement:
    parts_dir = root / "parts"
    output_dir = root / "dataset"
    for part_id in range(parts):
        DatasetPartWriter(
            parts_dir / f"part-{part_id:05d}",
            dataset_id="bench-store",
            split="train",
            shard_id=part_id,
            num_shards=parts,
            max_shard_samples=max_shard_samples,
        ).write(
            indexed_samples(
                samples=samples,
                parts=parts,
                part_id=part_id,
                text_views=text_views,
                audio_views=audio_views,
            )
        )

    start = time.perf_counter()
    commit_store_parts(output_dir, parts_dir, dataset_id="bench-store", split="train")
    seconds = time.perf_counter() - start
    return Measurement(
        seconds=seconds,
        detail={
            "samples": samples,
            "parts": parts,
            "views": text_views + audio_views,
            "max_shard_samples": max_shard_samples,
        },
    )


def indexed_samples(
    *,
    samples: int,
    parts: int,
    part_id: int,
    text_views: int,
    audio_views: int,
) -> Iterator[tuple[int, Sample]]:
    for index in range(part_id, samples, parts):
        yield index, sample(index, text_views=text_views, audio_views=audio_views)


def sample(index: int, *, text_views: int, audio_views: int) -> Sample:
    output: dict[tuple[Role, Modality], Any] = {}
    for offset, role in enumerate(text_roles(text_views)):
        output[role, Modality.TEXT] = TextItem(
            views={TextView.TEXT: f"text-{offset}-{index}"}
        )
    for offset, role in enumerate(audio_roles(audio_views)):
        output[role, Modality.AUDIO] = AudioItem(
            views={AudioView.WAVEFORM: ([[float(index + offset)]], 16_000)}
        )
    return output


def text_roles(count: int) -> tuple[Role, ...]:
    return roles(count)


def audio_roles(count: int) -> tuple[Role, ...]:
    return roles(count)


def roles(count: int) -> tuple[Role, ...]:
    available = (Role.DEFAULT, Role.SOURCE, Role.TARGET)
    if count > len(available):
        raise ValueError(f"view role count must be <= {len(available)}.")
    return available[:count]


def bench_sharded_csv(
    root: Path,
    *,
    csv_shards: int,
    files_per_shard: int,
    rows_per_file: int,
    num_shards: int,
    shard_id: int,
) -> Measurement:
    csv_root = root / "csv"
    write_csv_dataset(
        csv_root,
        csv_shards=csv_shards,
        files_per_shard=files_per_shard,
        rows_per_file=rows_per_file,
    )
    dataset = ShardedCsvSource().prepare(
        Spec(source="sharded_csv", path=str(csv_root)),
        root / "cache",
    )
    expected_rows = csv_shards * files_per_shard * rows_per_file

    start = time.perf_counter()
    rows = sum(1 for _index, _row in dataset.iter_indexed_shard(num_shards, shard_id))
    seconds = time.perf_counter() - start
    return Measurement(
        seconds=seconds,
        detail={
            "rows": expected_rows,
            "selected_rows": rows,
            "csv_shards": csv_shards,
            "files_per_shard": files_per_shard,
            "rows_per_file": rows_per_file,
            "num_shards": num_shards,
            "shard_id": shard_id,
        },
    )


def bench_store_reader_variants(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "lazy_all": run_repeated(
            lambda root: bench_store_reader(root, args, "lazy_all"),
            repeats=args.repeats,
        ),
        "preload_all": run_repeated(
            lambda root: bench_store_reader(root, args, "preload_all"),
            repeats=args.repeats,
        ),
        "preload_one": run_repeated(
            lambda root: bench_store_reader(root, args, "preload_one"),
            repeats=args.repeats,
        ),
    }


def bench_store_reader(
    root: Path,
    args: argparse.Namespace,
    mode: str,
) -> Measurement:
    output_dir = root / "reader"
    DatasetWriter(
        output_dir,
        dataset_id="bench-reader",
        split="train",
        max_shard_samples=args.store_max_shard_samples,
    ).write(
        sample(
            index,
            text_views=args.store_text_views,
            audio_views=args.store_audio_views,
        )
        for index in range(args.store_samples)
    )
    selected = ((Role.DEFAULT, Modality.AUDIO, AudioView.WAVEFORM),)

    start = time.perf_counter()
    match mode:
        case "lazy_all":
            dataset = read_store_dataset(output_dir)
        case "preload_all":
            dataset = read_store_dataset(output_dir, preload=True)
        case "preload_one":
            dataset = read_store_dataset(output_dir, views=selected, preload=True)
        case _:
            raise KeyError(mode)
    seconds = time.perf_counter() - start
    return Measurement(
        seconds=seconds,
        detail={
            "samples": args.store_samples,
            "views": args.store_text_views + args.store_audio_views,
            "loaded_views": len(dataset.views._cache),
            "mode": mode,
        },
    )


def bench_indexed_loader_variants(args: argparse.Namespace) -> dict[str, Any]:
    variants = indexed_variants(args.indexed_variants)
    return {
        variant: run_repeated(
            lambda _root, variant=variant: bench_indexed_loader(
                variant,
                samples=args.indexed_samples,
                batch_size=args.indexed_batch_size,
                num_workers=args.indexed_num_workers,
                prefetch_factor=args.indexed_prefetch_factor,
                payload_bytes=args.indexed_payload_bytes,
                num_shards=args.indexed_num_shards,
                shard_id=args.indexed_shard_id,
            ),
            repeats=args.repeats,
        )
        for variant in variants
    }


def indexed_variants(value: str) -> tuple[str, ...]:
    allowed = {"runtime", "map_default", "map_spawn", "map_fork"}
    output = tuple(item.strip() for item in value.split(",") if item.strip())
    unknown = sorted(set(output) - allowed)
    if unknown:
        raise ValueError(f"Unknown indexed loader variants: {unknown}.")
    if len(output) == 0:
        raise ValueError("indexed_variants must contain at least one variant.")
    if "map_fork" in output and "fork" not in multiprocessing.get_all_start_methods():
        return tuple(variant for variant in output if variant != "map_fork")
    return output


def bench_indexed_loader(
    variant: str,
    *,
    samples: int,
    batch_size: int,
    num_workers: int,
    prefetch_factor: int | None,
    payload_bytes: int,
    num_shards: int,
    shard_id: int,
) -> Measurement:
    factory = SyntheticDatasetFactory(samples=samples, payload_bytes=payload_bytes)
    loader = make_indexed_loader(
        variant,
        factory=factory,
        samples=samples,
        batch_size=batch_size,
        num_workers=num_workers,
        prefetch_factor=prefetch_factor,
        num_shards=num_shards,
        shard_id=shard_id,
    )

    start = time.perf_counter()
    selected = 0
    checksum = 0
    for batch in loader:
        for index, row in batch:
            selected += 1
            checksum += int(index)
            checksum += len(row[Role.DEFAULT, Modality.TEXT].views[TextView.TEXT])
    seconds = time.perf_counter() - start
    return Measurement(
        seconds=seconds,
        detail={
            "variant": variant,
            "samples": samples,
            "selected_samples": selected,
            "checksum": checksum,
            "batch_size": batch_size,
            "num_workers": num_workers,
            "prefetch_factor": prefetch_factor,
            "payload_bytes": payload_bytes,
            "num_shards": num_shards,
            "shard_id": shard_id,
        },
    )


def bench_filter_parallel(
    root: Path,
    *,
    samples: int,
    devices: int,
    batch_size: int,
    num_workers: int,
    prefetch_factor: int | None,
    commit_samples: int,
    payload_bytes: int,
) -> Measurement:
    store_dir = root / "filter-store"
    DatasetWriter(store_dir, dataset_id="bench-filter", split="train").write(
        synthetic_sample(index, payload_bytes) for index in range(samples)
    )
    factory = StoreDatasetFactory(store_dir)
    rule = FilterRule(name="bench_modulo", factory=ModuloFilterFactory(modulo=3))
    previous_home = os.environ.get("ANYDATASET_HOME")
    os.environ["ANYDATASET_HOME"] = str(root / "home")
    try:
        start = time.perf_counter()
        result = rule.apply(
            dataset_factory=factory,
            device=tuple(f"cpu:{index}" for index in range(devices)),
            batch_size=batch_size,
            num_workers=num_workers,
            prefetch_factor=prefetch_factor,
            commit_samples=commit_samples,
        )
        seconds = time.perf_counter() - start
    finally:
        if previous_home is None:
            os.environ.pop("ANYDATASET_HOME", None)
        else:
            os.environ["ANYDATASET_HOME"] = previous_home
    return Measurement(
        seconds=seconds,
        detail={
            "samples": samples,
            "devices": devices,
            "batch_size": batch_size,
            "num_workers": num_workers,
            "prefetch_factor": prefetch_factor,
            "commit_samples": commit_samples,
            "payload_bytes": payload_bytes,
            "counts": dict(result.counts),
        },
    )


def bench_writer_pipeline_variants(args: argparse.Namespace) -> dict[str, Any]:
    variants = writer_variants(args.writer_variants)
    return {
        variant: run_repeated(
            lambda root, variant=variant: bench_writer_pipeline(
                root,
                variant,
                jobs=args.writer_jobs,
                payload_bytes=args.writer_payload_bytes,
                workers=args.writer_workers,
                prefetch=args.writer_prefetch,
                producer_delay_ms=args.writer_producer_delay_ms,
            ),
            repeats=args.repeats,
        )
        for variant in variants
    }


def writer_variants(value: str) -> tuple[str, ...]:
    allowed = {"inline", "thread", "process_spawn", "process_fork"}
    output = tuple(item.strip() for item in value.split(",") if item.strip())
    unknown = sorted(set(output) - allowed)
    if unknown:
        raise ValueError(f"Unknown writer pipeline variants: {unknown}.")
    if len(output) == 0:
        raise ValueError("writer_variants must contain at least one variant.")
    if "process_fork" in output and "fork" not in multiprocessing.get_all_start_methods():
        return tuple(variant for variant in output if variant != "process_fork")
    return output


def bench_writer_pipeline(
    root: Path,
    variant: str,
    *,
    jobs: int,
    payload_bytes: int,
    workers: int,
    prefetch: int | None,
    producer_delay_ms: float,
) -> Measurement:
    output_dir = root / "writer" / variant
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = synthetic_payload(0, payload_bytes).encode("utf-8")
    write_jobs = tuple(
        WriteJob(path=output_dir / f"part-{index:06d}.bin", payload=payload)
        for index in range(jobs)
    )

    start = time.perf_counter()
    match variant:
        case "inline":
            for job in write_jobs:
                write_payload_job(job)
                maybe_sleep(producer_delay_ms)
        case "thread":
            run_thread_writer(
                write_jobs,
                workers=workers,
                prefetch=prefetch,
                producer_delay_ms=producer_delay_ms,
            )
        case "process_spawn":
            run_process_writer(
                write_jobs,
                workers=workers,
                prefetch=prefetch,
                producer_delay_ms=producer_delay_ms,
                start_method="spawn",
            )
        case "process_fork":
            run_process_writer(
                write_jobs,
                workers=workers,
                prefetch=prefetch,
                producer_delay_ms=producer_delay_ms,
                start_method="fork",
            )
        case _:
            raise KeyError(variant)
    seconds = time.perf_counter() - start

    total_bytes = sum(path.stat().st_size for path in output_dir.iterdir())
    return Measurement(
        seconds=seconds,
        detail={
            "variant": variant,
            "jobs": jobs,
            "payload_bytes": payload_bytes,
            "total_bytes": total_bytes,
            "workers": workers,
            "prefetch": prefetch,
            "producer_delay_ms": producer_delay_ms,
        },
    )


def run_process_writer(
    jobs: tuple["WriteJob", ...],
    *,
    workers: int,
    prefetch: int | None,
    producer_delay_ms: float,
    start_method: str,
) -> None:
    with BackgroundWriteSink(
        write_payload_job,
        workers=workers,
        max_pending=prefetch,
        start_method=start_method,
        backend="process",
    ) as sink:
        for job in jobs:
            sink.submit(job)
            maybe_sleep(producer_delay_ms)


def run_thread_writer(
    jobs: tuple["WriteJob", ...],
    *,
    workers: int,
    prefetch: int | None,
    producer_delay_ms: float,
) -> None:
    if workers <= 0:
        for job in jobs:
            write_payload_job(job)
            maybe_sleep(producer_delay_ms)
        return
    with BackgroundWriteSink(
        write_payload_job,
        workers=workers,
        max_pending=prefetch,
        start_method="spawn",
        backend="thread",
    ) as sink:
        for job in jobs:
            sink.submit(job)
            maybe_sleep(producer_delay_ms)


def maybe_sleep(milliseconds: float) -> None:
    if milliseconds > 0:
        time.sleep(milliseconds / 1000)


@dataclass(frozen=True)
class WriteJob:
    path: Path
    payload: bytes


def write_payload_job(job: WriteJob) -> None:
    job.path.write_bytes(job.payload)


def make_indexed_loader(
    variant: str,
    *,
    factory: "SyntheticDatasetFactory",
    samples: int,
    batch_size: int,
    num_workers: int,
    prefetch_factor: int | None,
    num_shards: int,
    shard_id: int,
) -> DataLoader:
    if variant == "runtime":
        if num_shards != 1 or shard_id != 0:
            raise ValueError("runtime variant only supports indexed_num_shards=1.")
        return runtime_indexed_loader(
            factory,
            batch_size=batch_size,
            num_workers=num_workers,
            prefetch_factor=prefetch_factor,
        )

    with rank_environment(num_shards=num_shards, shard_id=shard_id):
        return map_style_indexed_loader(
            factory,
            sample_count=samples,
            batch_size=batch_size,
            num_workers=num_workers,
            prefetch_factor=prefetch_factor,
            start_method=indexed_variant_start_method(variant),
        )


def indexed_variant_start_method(variant: str) -> str:
    if variant == "map_default":
        return "spawn"
    if variant == "map_spawn":
        return "spawn"
    if variant == "map_fork":
        return "fork"
    raise ValueError(f"Unsupported indexed loader variant: {variant}.")


def indexed_collate(batch: list[tuple[int, Sample]]) -> tuple[tuple[int, Sample], ...]:
    return tuple(batch)


@dataclass(frozen=True)
class SyntheticDatasetFactory:
    samples: int
    payload_bytes: int

    def __call__(self) -> "SyntheticDataset":
        return SyntheticDataset(samples=self.samples, payload_bytes=self.payload_bytes)


@dataclass(frozen=True)
class SyntheticDataset(Dataset):
    samples: int
    payload_bytes: int

    def __len__(self) -> int:
        return self.samples

    def __getitem__(self, index: int) -> Sample:
        if index < 0:
            index += len(self)
        if index < 0 or index >= len(self):
            raise IndexError("synthetic dataset index out of range.")
        return {
            (Role.DEFAULT, Modality.TEXT): TextItem(
                views={TextView.TEXT: synthetic_payload(index, self.payload_bytes)}
            )
        }


@dataclass(frozen=True)
class StoreDatasetFactory:
    root: Path

    def __call__(self):
        return read_store_dataset(self.root)


@dataclass(frozen=True)
class ModuloFilterFactory:
    modulo: int

    def __call__(self) -> "ModuloFilter":
        return ModuloFilter(modulo=self.modulo)


@dataclass(frozen=True)
class ModuloFilter:
    modulo: int

    def __call__(self, sample: Sample) -> str:
        text = sample[Role.DEFAULT, Modality.TEXT].views[TextView.TEXT]
        index = int(text.split("-", maxsplit=2)[1])
        return f"bucket_{index % self.modulo}"


class rank_environment:
    def __init__(self, *, num_shards: int, shard_id: int) -> None:
        self.values = {
            "WORLD_SIZE": str(num_shards),
            "RANK": str(shard_id),
        }
        self.previous: dict[str, str | None] = {}

    def __enter__(self):
        for key, value in self.values.items():
            self.previous[key] = os.environ.get(key)
            os.environ[key] = value
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        for key, value in self.previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def synthetic_payload(index: int, payload_bytes: int) -> str:
    prefix = f"text-{index}-"
    if len(prefix) >= payload_bytes:
        return prefix
    return prefix + ("x" * (payload_bytes - len(prefix)))


def synthetic_sample(index: int, payload_bytes: int) -> Sample:
    return {
        (Role.DEFAULT, Modality.TEXT): TextItem(
            views={TextView.TEXT: synthetic_payload(index, payload_bytes)}
        )
    }


def write_csv_dataset(
    root: Path,
    *,
    csv_shards: int,
    files_per_shard: int,
    rows_per_file: int,
) -> None:
    row_id = 0
    for shard_id in range(csv_shards):
        shard_dir = root / f"shard_{shard_id}"
        shard_dir.mkdir(parents=True)
        for file_id in range(files_per_shard):
            path = shard_dir / f"{file_id}.csv"
            with path.open("w", encoding="utf-8", newline="") as file:
                writer = csv.DictWriter(file, fieldnames=("id", "text"))
                writer.writeheader()
                for _ in range(rows_per_file):
                    writer.writerow({"id": row_id, "text": f"text-{row_id}"})
                    row_id += 1


if __name__ == "__main__":
    main()
