from __future__ import annotations

from .types import Preset, Source, SourceKey, Spec, source_key


def resolve_dataset(dataset: str | Preset | Spec) -> Spec:
    if isinstance(dataset, Spec):
        return dataset
    if isinstance(dataset, Preset):
        return dataset.spec()
    if isinstance(dataset, str):
        return resolve_shorthand(dataset)
    raise TypeError("dataset must be a string, Preset or Spec.")


def resolve_shorthand(shorthand: str) -> Spec:
    source, body = split_source_prefix(shorthand)
    if source is not None:
        path, split = split_name_and_split(body)
        if not path:
            raise ValueError(
                f"{source_key(source)} dataset shorthand must include a path."
            )
        return Spec(source=source, path=path, split=split)

    name, split = split_name_and_split(shorthand)
    try:
        preset = Preset(name)
    except ValueError as exc:
        raise KeyError(
            f"Unknown dataset preset {name!r}. Use a registered source shorthand "
            "such as `hf://`, `hf-disk://` or `store://` for raw specs."
        ) from exc
    return preset.spec(split=split)


def split_source_prefix(shorthand: str) -> tuple[SourceKey | None, str]:
    if shorthand.startswith("hf://"):
        return Source.HF, shorthand[len("hf://") :]
    if shorthand.startswith("store://"):
        return Source.STORE, shorthand[len("store://") :]
    if "://" in shorthand:
        from .dataset.source import has_source

        source, body = shorthand.split("://", 1)
        if not has_source(source):
            raise KeyError(f"Unknown dataset source: {source!r}.")
        return source, body
    return None, shorthand


def split_name_and_split(value: str) -> tuple[str, str | None]:
    bracket_depth = 0
    for index in range(len(value) - 1, -1, -1):
        char = value[index]
        if char == "]":
            bracket_depth += 1
            continue
        if char == "[" and bracket_depth > 0:
            bracket_depth -= 1
            continue
        if char == ":" and bracket_depth == 0:
            if windows_drive_colon(value, index):
                continue
            name, split = value[:index], value[index + 1 :]
            return name, split
    return value, None


def windows_drive_colon(value: str, index: int) -> bool:
    return (
        index == 1
        and len(value) > 2
        and value[0].isalpha()
        and value[2] in {"/", "\\"}
    )
