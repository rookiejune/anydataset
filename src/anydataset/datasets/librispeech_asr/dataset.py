from __future__ import annotations

from anydataset.api.spec import DatasetSpec


def librispeech_asr_spec(
    config_name: str = "clean",
    split: str = "train.100",
    streaming: bool = True,
) -> DatasetSpec:
    return DatasetSpec(
        source="huggingface",
        path="openslr/librispeech_asr",
        name="librispeech_asr",
        split=split,
        load_options={
            "config_name": config_name,
            "streaming": streaming,
        },
    )
