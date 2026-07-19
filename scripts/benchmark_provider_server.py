"""Benchmark direct and ProviderServer calls with a real LongCat provider."""

from __future__ import annotations

import argparse
import gc
import json
import math
import statistics
import tempfile
import time
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Any, Callable

import torch
import torchaudio

from anydataset.dataset.collate import Batch, FieldGroup, FieldRef
from anydataset.provider.longcat import LongCatProvider
from anydataset.provider_service import (
    ProviderServer,
    RemoteProvider,
    _ProviderCommand,
    _request,
)
from anydataset.types import (
    AudioItem,
    AudioView,
    Modality,
    Role,
)


@dataclass(frozen=True)
class LongCatFactory:
    cache_dir: Path

    def __call__(self, device: str) -> LongCatProvider:
        return LongCatProvider(
            cache_dir=self.cache_dir,
            device=device,
            local_files_only=True,
        )


def main() -> None:
    args = parse_args()
    waveform, sample_rate = torchaudio.load(args.audio)
    max_samples = max(1, round(args.seconds * sample_rate))
    views = {
        AudioView.WAVEFORM: (
            waveform[..., :max_samples].contiguous(),
            sample_rate,
        )
    }
    batch = audio_batch(*views[AudioView.WAVEFORM], size=args.batch_size)
    factory = LongCatFactory(args.cache_dir)

    direct_load_start = time.perf_counter()
    provider = factory(args.device)
    direct_load_seconds = time.perf_counter() - direct_load_start
    direct_call = request(provider, views, batch)
    direct = measure(
        direct_call,
        warmup=args.warmup,
        repeats=args.repeats,
    )
    direct_output = direct.pop("output")
    del direct_call, provider
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    with tempfile.TemporaryDirectory(prefix="anydataset-provider-bench-") as tmpdir:
        address = Path(tmpdir) / "provider.sock"
        server = ProviderServer(
            address=address,
            provider_factory=factory,
            device=args.device,
            start_method=args.start_method,
            startup_timeout=args.startup_timeout,
        )
        remote_load_start = time.perf_counter()
        server.start()
        remote_load_seconds = time.perf_counter() - remote_load_start
        try:
            remote = RemoteProvider(AudioView.LONGCAT, address)
            remote_call = request(remote, views, batch)
            remote_result = measure(
                remote_call,
                warmup=args.warmup,
                repeats=args.repeats,
            )
            connection_baseline = measure(
                partial(
                    _request,
                    address,
                    None,
                    _ProviderCommand.PING,
                    None,
                ),
                warmup=args.warmup,
                repeats=args.repeats,
            )
        finally:
            server.stop()

    remote_output = remote_result.pop("output")
    connection_baseline.pop("output")
    if not outputs_equal(direct_output, remote_output):
        raise RuntimeError("Direct and remote LongCat outputs differ.")
    remote_overhead_ms = remote_result["median_ms"] - direct["median_ms"]
    report = {
        "audio": str(args.audio),
        "audio_seconds": views[AudioView.WAVEFORM][0].shape[-1] / sample_rate,
        "cache_dir": str(args.cache_dir),
        "device": args.device,
        "batch_size": args.batch_size,
        "start_method": args.start_method,
        "warmup": args.warmup,
        "repeats": args.repeats,
        "direct_load_seconds": direct_load_seconds,
        "remote_load_seconds": remote_load_seconds,
        "direct": direct,
        "remote": remote_result,
        "remote_overhead_ms": remote_overhead_ms,
        "remote_overhead_ratio": remote_overhead_ms / direct["median_ms"],
        "connection_baseline": connection_baseline,
        "output_shapes": output_shapes(direct_output),
    }
    print(json.dumps(report, indent=2, sort_keys=True))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark real LongCat calls through ProviderServer.",
    )
    parser.add_argument("--audio", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seconds", type=positive_float, default=2.0)
    parser.add_argument("--batch-size", type=positive_int, default=1)
    parser.add_argument("--warmup", type=non_negative_int, default=2)
    parser.add_argument("--repeats", type=positive_int, default=20)
    parser.add_argument("--start-method", choices=("spawn", "fork"), default="spawn")
    parser.add_argument("--startup-timeout", type=positive_float, default=600.0)
    args = parser.parse_args()
    if not args.audio.is_file():
        parser.error(f"audio file does not exist: {args.audio}")
    if not args.cache_dir.is_dir():
        parser.error(f"cache directory does not exist: {args.cache_dir}")
    return args


def measure(
    call: Callable[[], Any],
    *,
    warmup: int,
    repeats: int,
) -> dict[str, Any]:
    output = None
    for _ in range(warmup):
        output = call()
    elapsed = []
    for _ in range(repeats):
        start = time.perf_counter()
        output = call()
        elapsed.append((time.perf_counter() - start) * 1000)
    ordered = sorted(elapsed)
    p95_index = min(len(ordered) - 1, math.ceil(len(ordered) * 0.95) - 1)
    return {
        "output": output,
        "min_ms": min(elapsed),
        "median_ms": statistics.median(elapsed),
        "p95_ms": ordered[p95_index],
        "max_ms": max(elapsed),
        "runs_ms": elapsed,
    }


def audio_batch(
    waveform: torch.Tensor,
    sample_rate: int,
    *,
    size: int,
) -> Batch | None:
    if size == 1:
        return None
    ref = (Role.DEFAULT, Modality.AUDIO)
    field = FieldRef(ref, FieldGroup.VIEWS, AudioView.WAVEFORM)
    waveforms = waveform.unsqueeze(0).expand(size, *waveform.shape).contiguous()
    rates = torch.full((size,), sample_rate, dtype=torch.int64)
    mask = torch.ones(
        (size, waveform.shape[-1]),
        dtype=torch.bool,
    )
    return Batch(
        sample={
            ref: AudioItem(
                views={AudioView.WAVEFORM: (waveforms, rates)},
            )
        },
        masks={field: mask},
    )


def request(
    provider: Any,
    views: dict[AudioView, Any],
    batch: Batch | None,
) -> Callable[[], Any]:
    if batch is None:
        return partial(provider, views)
    return partial(provider.call_batch, batch)


def outputs_equal(left: Any, right: Any) -> bool:
    if isinstance(left, torch.Tensor) and isinstance(right, torch.Tensor):
        return torch.equal(left, right)
    if isinstance(left, (list, tuple)) and isinstance(right, (list, tuple)):
        return len(left) == len(right) and all(
            outputs_equal(left_item, right_item)
            for left_item, right_item in zip(left, right)
        )
    return left == right


def output_shapes(output: Any) -> list[list[int]]:
    if isinstance(output, torch.Tensor):
        return [list(output.shape)]
    if isinstance(output, (list, tuple)):
        return [
            list(item.shape)
            for item in output
            if isinstance(item, torch.Tensor)
        ]
    raise TypeError("LongCat benchmark output must contain tensors.")


def positive_float(value: str) -> float:
    result = float(value)
    if not math.isfinite(result) or result <= 0:
        raise argparse.ArgumentTypeError("value must be a finite positive number")
    return result


def non_negative_int(value: str) -> int:
    result = int(value)
    if result < 0:
        raise argparse.ArgumentTypeError("value must be non-negative")
    return result


def positive_int(value: str) -> int:
    result = non_negative_int(value)
    if result == 0:
        raise argparse.ArgumentTypeError("value must be positive")
    return result


if __name__ == "__main__":
    main()
