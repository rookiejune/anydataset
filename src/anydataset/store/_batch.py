"""Batch provider helpers for materialization.

The module batches samples, validates batched provider outputs, and retries
CUDA out-of-memory batches by splitting them into smaller calls.
"""

from __future__ import annotations

import gc
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from typing import Any

import torch

from ..dataset.collate import Batch, collate_fn
from ..types.item import (
    AudioReq,
    ImageReq,
    Item,
    Modality,
    Requirement,
    Role,
    Sample,
    TextReq,
)
from ._modality import modality_inputs, role_items, with_modality_view
from ._types import ModalityProviderLike, output_modality
from ._view import with_view


def sample_batches(
    samples: Iterable[Sample],
    batch_size: int,
) -> Iterator[tuple[Sample, ...]]:
    batch: list[Sample] = []
    for sample in samples:
        batch.append(sample)
        if len(batch) == batch_size:
            yield tuple(batch)
            batch = []
    if batch:
        yield tuple(batch)


def indexed_sample_batches(
    samples: Iterable[tuple[int, Sample]],
    batch_size: int,
) -> Iterator[tuple[tuple[int, Sample], ...]]:
    batch: list[tuple[int, Sample]] = []
    for sample in samples:
        batch.append(sample)
        if len(batch) == batch_size:
            yield tuple(batch)
            batch = []
    if batch:
        yield tuple(batch)


def with_resilient_batch_provider(
    samples: Sequence[Sample],
    call: Callable[[Sequence[Sample]], Sequence[Sample]],
) -> Iterator[Sample]:
    try:
        yield from call(samples)
    except Exception as exc:
        oom = _is_oom_error(exc)
        if len(samples) <= 1 or not oom:
            raise
        _release_exception(exc)
        _clear_cuda_cache()
        midpoint = len(samples) // 2
        yield from with_resilient_batch_provider(samples[:midpoint], call)
        yield from with_resilient_batch_provider(samples[midpoint:], call)


def with_batch_view_provider(
    samples: Sequence[Sample],
    provider: Any,
) -> Iterator[Sample]:
    output = provider.output
    modality = output_modality(output)
    refs = _batch_view_refs(samples, modality)
    outputs: list[dict[tuple[Role, Modality], Item]] = [{} for _ in samples]
    if refs:
        schema = {ref: _input_requirement(samples, ref) for ref in refs}
        batch = collate_fn(schema)(samples)
        values_by_ref = _ref_batch_outputs(
            _call_batch(provider, batch),
            refs,
            len(samples),
            provider_kind="view",
        )
    else:
        values_by_ref = {}
    for ref, values in values_by_ref.items():
        validate_batch_outputs(values, len(samples))
        for index, (sample, value) in enumerate(zip(samples, values, strict=True)):
            outputs[index][ref] = with_view(sample[ref], output, value)
    yield from outputs


def with_batch_modality_provider(
    samples: Sequence[Sample],
    provider: ModalityProviderLike,
) -> Iterator[Sample]:
    output = provider.output
    out_modality = output_modality(output)
    reference_role = getattr(provider, "reference_role", None)
    roles = _batch_modality_input_roles(samples, out_modality, reference_role)
    outputs: list[dict[tuple[Role, Modality], Item]] = [{} for _ in samples]
    input_refs = {
        role: _batch_modality_input_ref(samples, role, out_modality)
        for role in roles
    }
    if input_refs:
        schema_refs = tuple(input_refs.values()) + _reference_refs(
            reference_role,
            out_modality,
        )
        schema = {ref: _input_requirement(samples, ref) for ref in schema_refs}
        batch = collate_fn(schema)(samples)
        values_by_ref = _ref_batch_outputs(
            _call_batch(provider, batch),
            tuple(input_refs.values()),
            len(samples),
            provider_kind="modality",
        )
    else:
        values_by_ref = {}
    for role, input_ref in input_refs.items():
        values = values_by_ref[input_ref]
        ref = (role, out_modality)
        for index, value in enumerate(values):
            outputs[index][ref] = with_modality_view(output, value)
    yield from outputs


def _batch_view_refs(
    samples: Sequence[Sample],
    modality: Modality,
) -> tuple[tuple[Role, Modality], ...]:
    if not samples:
        return ()
    refs = _sorted_refs(ref for ref in samples[0] if ref[1] is modality)
    for sample in samples:
        if _sorted_refs(ref for ref in sample if ref[1] is modality) != refs:
            raise ValueError("Batch samples must share provider input references.")
    return refs


def _batch_modality_input_roles(
    samples: Sequence[Sample],
    output: Modality,
    reference_role: Role | None,
) -> tuple[Role, ...]:
    if not samples:
        return ()
    roles = tuple(
        sorted(
            (
                role
                for role, _ in modality_inputs(
                    role_items(samples[0]),
                    output,
                    reference_role,
                )
            ),
            key=lambda role: role.value,
        )
    )
    for sample in samples:
        sample_roles = tuple(
            sorted(
                (
                    role
                    for role, _ in modality_inputs(
                        role_items(sample),
                        output,
                        reference_role,
                    )
                ),
                key=lambda role: role.value,
            )
        )
        if sample_roles != roles:
            raise ValueError("Batch samples must share modality provider input roles.")
    return roles


def _batch_modality_input_ref(
    samples: Sequence[Sample],
    role: Role,
    output: Modality,
) -> tuple[Role, Modality]:
    refs: list[tuple[Role, Modality]] = []
    for sample in samples:
        inputs = tuple(
            (ref_role, modality)
            for ref_role, modality in sample
            if ref_role is role and modality is not output
        )
        if len(inputs) != 1:
            names = ", ".join(sorted(modality.value for _, modality in inputs))
            raise ValueError(
                f"Role {role.value!r} needs exactly one input modality when "
                f"materializing {output.value!r}; got {names or 'none'}."
            )
        refs.append(inputs[0])
    ref = refs[0]
    if any(value != ref for value in refs):
        raise ValueError("Batch samples must share modality provider input references.")
    return ref


def _reference_refs(
    reference_role: Role | None,
    output: Modality,
) -> tuple[tuple[Role, Modality], ...]:
    if reference_role is None:
        return ()
    return ((reference_role, output),)


def _input_requirement(
    samples: Sequence[Sample],
    ref: tuple[Role, Modality],
) -> Requirement:
    views: set[Any] = set()
    meta: set[Any] = set()
    for sample in samples:
        sample_item = sample[ref]
        views.update(sample_item.views)
        meta.update(sample_item.meta)

    match ref[1]:
        case Modality.AUDIO:
            return AudioReq.from_iter(views, meta)
        case Modality.IMAGE:
            return ImageReq.from_iter(views, meta)
        case Modality.TEXT:
            return TextReq.from_iter(views, meta)
    raise TypeError(f"Unsupported sample reference: {ref!r}.")


def _sorted_refs(
    refs: Iterable[tuple[Role, Modality]],
) -> tuple[tuple[Role, Modality], ...]:
    return tuple(sorted(refs, key=lambda ref: (ref[0].value, ref[1].value)))


def _call_batch(
    provider: Any,
    batch: Batch,
) -> Sequence[Any] | Mapping[tuple[Role, Modality], Sequence[Any]]:
    try:
        call_batch = provider.call_batch
    except AttributeError as exc:
        raise TypeError("batch_size > 1 requires provider.call_batch().") from exc
    return call_batch(batch)


def _ref_batch_outputs(
    values: Sequence[Any] | Mapping[tuple[Role, Modality], Sequence[Any]],
    refs: Sequence[tuple[Role, Modality]],
    sample_count: int,
    *,
    provider_kind: str,
) -> Mapping[tuple[Role, Modality], Sequence[Any]]:
    if len(refs) == 1 and not isinstance(values, Mapping):
        validate_batch_outputs(values, sample_count)
        return {refs[0]: values}
    if not isinstance(values, Mapping):
        raise TypeError(
            f"Batch {provider_kind} providers with multiple input references must return "
            "a mapping from reference to outputs."
        )

    expected = set(refs)
    actual = set(values)
    if actual != expected:
        missing = ", ".join(_ref_name(ref) for ref in sorted(expected - actual))
        extra = ", ".join(_ref_name(ref) for ref in sorted(actual - expected))
        details = []
        if missing:
            details.append(f"missing {missing}")
        if extra:
            details.append(f"unexpected {extra}")
        raise ValueError(
            f"Batch {provider_kind} provider returned outputs for the wrong references"
            f" ({'; '.join(details)})."
        )

    for ref, outputs in values.items():
        if isinstance(outputs, Mapping) or not isinstance(outputs, Sequence):
            raise TypeError(
                f"Batch {provider_kind} provider outputs for {_ref_name(ref)} "
                "must be a sequence."
            )
        validate_batch_outputs(outputs, sample_count)
    return values


def _ref_name(ref: tuple[Role, Modality]) -> str:
    role, modality = ref
    return f"({role.value}, {modality.value})"


def validate_batch_outputs(values: Sequence[Any], expected: int) -> None:
    if len(values) != expected:
        raise ValueError(
            f"Batch provider returned {len(values)} outputs for {expected} samples."
        )


def _is_oom_error(error: BaseException) -> bool:
    if isinstance(error, torch.OutOfMemoryError):
        return True
    if not isinstance(error, RuntimeError):
        return False
    message = str(error).lower()
    return "out of memory" in message or "cuda error: out of memory" in message


def _release_exception(error: BaseException) -> None:
    pending: list[BaseException] = [error]
    seen: set[int] = set()
    while pending:
        current = pending.pop()
        if id(current) in seen:
            continue
        seen.add(id(current))
        if current.__cause__ is not None:
            pending.append(current.__cause__)
        if current.__context__ is not None:
            pending.append(current.__context__)
        current.__traceback__ = None
        current.__cause__ = None
        current.__context__ = None


def _clear_cuda_cache() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
