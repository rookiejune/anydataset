from __future__ import annotations

from anydataset.api.spec import DatasetSpec


def esc50_spec(
    split: str = "train",
    streaming: bool = True,
) -> DatasetSpec:
    return DatasetSpec(
        source="huggingface",
        path="ashraq/esc50",
        name="esc50",
        split=split,
        load_options={
            "streaming": streaming,
        },
    )
