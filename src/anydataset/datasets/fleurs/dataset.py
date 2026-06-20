from __future__ import annotations

from anydataset.api.spec import DatasetSpec


def fleurs_spec(
    language: str = "en_us",
    split: str = "train",
    streaming: bool = True,
) -> DatasetSpec:
    return DatasetSpec(
        source="huggingface",
        path="google/fleurs",
        name="fleurs",
        split=split,
        load_options={
            "config_name": language,
            "streaming": streaming,
        },
    )
