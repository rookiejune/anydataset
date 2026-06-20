from dataclasses import dataclass
from typing import Any, Mapping


@dataclass(frozen=True)
class Sample:
    data: Mapping[str, Any]
    dataset_name: str
    sample_index: int
