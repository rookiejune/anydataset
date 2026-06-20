from __future__ import annotations

from anydataset.api.spec import DatasetSpec


def nsynth_spec(
    config_name: str = "instrument",
    split: str = "train",
    streaming: bool = True,
) -> DatasetSpec:
    return DatasetSpec(
        source="huggingface",
        path="confit/nsynth-parquet",
        name="nsynth",
        split=split,
        load_options={
            "config_name": config_name,
            "streaming": streaming,
        },
    )
