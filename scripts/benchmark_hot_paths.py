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
