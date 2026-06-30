from __future__ import annotations

import argparse
import csv
import json
import multiprocessing
import statistics
import tempfile
import time
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from anydataset import AudioItem, AudioView, Modality, Role, Sample, Spec, TextItem, TextView
from anydataset._parallel import indexed_loader as runtime_indexed_loader
from anydataset.dataset.source.sharded_csv import ShardedCsvSource
from anydataset.store.parts import DatasetPartWriter, commit_store_parts
from torch.utils.data import DataLoader, Dataset, Sampler


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
        "indexed_loader": bench_indexed_loader_variants(args),
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
    validate_positive("store_samples", samples)
    validate_positive("store_parts", parts)
    validate_positive("store_text_views", text_views)
    validate_positive("store_audio_views", audio_views)
    validate_positive("store_max_shard_samples", max_shard_samples)
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
    validate_positive("csv_shards", csv_shards)
    validate_positive("csv_files_per_shard", files_per_shard)
    validate_positive("csv_rows_per_file", rows_per_file)
    validate_positive("csv_num_shards", num_shards)
    if shard_id < 0 or shard_id >= num_shards:
        raise ValueError("csv_shard_id must satisfy 0 <= shard_id < csv_num_shards.")
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


def bench_indexed_loader_variants(args: argparse.Namespace) -> dict[str, Any]:
    validate_positive("indexed_samples", args.indexed_samples)
    validate_positive("indexed_batch_size", args.indexed_batch_size)
    validate_non_negative("indexed_num_workers", args.indexed_num_workers)
    validate_positive("indexed_payload_bytes", args.indexed_payload_bytes)
    validate_positive("indexed_num_shards", args.indexed_num_shards)
    if args.indexed_shard_id < 0 or args.indexed_shard_id >= args.indexed_num_shards:
        raise ValueError(
            "indexed_shard_id must satisfy 0 <= indexed_shard_id < indexed_num_shards."
        )
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

    context = indexed_loader_context(variant, num_workers)
    kwargs: dict[str, Any] = {
        "batch_size": batch_size,
        "collate_fn": indexed_collate,
        "num_workers": num_workers,
        "sampler": GlobalIndexSampler(
            samples=samples,
            num_shards=num_shards,
            shard_id=shard_id,
        ),
    }
    if num_workers > 0:
        if prefetch_factor is not None:
            kwargs["prefetch_factor"] = prefetch_factor
        if context is not None:
            kwargs["multiprocessing_context"] = context
    return DataLoader(MapIndexedDataset(factory), **kwargs)


def indexed_loader_context(variant: str, num_workers: int):
    if num_workers == 0:
        return None
    if variant == "map_default":
        return None
    if variant == "map_spawn":
        return multiprocessing.get_context("spawn")
    if variant == "map_fork":
        return multiprocessing.get_context("fork")
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


class MapIndexedDataset(Dataset):
    def __init__(self, dataset_factory: Callable[[], Dataset]) -> None:
        self.dataset_factory = dataset_factory
        self._dataset: Dataset | None = None

    def __getstate__(self) -> dict[str, Any]:
        return {"dataset_factory": self.dataset_factory}

    def __setstate__(self, state: dict[str, Any]) -> None:
        self.dataset_factory = state["dataset_factory"]
        self._dataset = None

    @property
    def dataset(self) -> Dataset:
        if self._dataset is None:
            self._dataset = self.dataset_factory()
        return self._dataset

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int) -> tuple[int, Sample]:
        return index, self.dataset[index]


@dataclass(frozen=True)
class GlobalIndexSampler(Sampler[int]):
    samples: int
    num_shards: int
    shard_id: int

    def __post_init__(self) -> None:
        validate_positive("samples", self.samples)
        validate_positive("num_shards", self.num_shards)
        if self.shard_id < 0 or self.shard_id >= self.num_shards:
            raise ValueError("shard_id must satisfy 0 <= shard_id < num_shards.")

    def __iter__(self) -> Iterator[int]:
        return iter(range(self.shard_id, self.samples, self.num_shards))

    def __len__(self) -> int:
        if self.shard_id >= self.samples:
            return 0
        return (self.samples - 1 - self.shard_id) // self.num_shards + 1


def synthetic_payload(index: int, payload_bytes: int) -> str:
    prefix = f"text-{index}-"
    if len(prefix) >= payload_bytes:
        return prefix
    return prefix + ("x" * (payload_bytes - len(prefix)))


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


def validate_positive(name: str, value: int) -> None:
    if value <= 0:
        raise ValueError(f"{name} must be positive.")


def validate_non_negative(name: str, value: int) -> None:
    if value < 0:
        raise ValueError(f"{name} must be non-negative.")


if __name__ == "__main__":
    main()
