from __future__ import annotations

import tarfile
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import torch

from anydataset.store.reader import read_store_dataset
from anydataset.store.writer import DatasetWriter
from anydataset.types import AudioItem, AudioView, Modality, Role


class PayloadCacheTest(unittest.TestCase):
    def test_payload_lookup_uses_cached_tarinfo_index(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            output = Path(tmpdir) / "store"
            ref = (Role.DEFAULT, Modality.AUDIO)
            DatasetWriter(output, dataset_id="payload-index").write(
                [
                    {
                        ref: AudioItem(
                            views={
                                AudioView.WAVEFORM: (
                                    torch.tensor([[float(index)]]),
                                    16_000,
                                )
                            }
                        )
                    }
                    for index in range(2)
                ]
            )
            dataset = read_store_dataset(output)

            with mock.patch.object(
                tarfile.TarFile,
                "_getmember",
                side_effect=AssertionError("linear tar member lookup"),
            ):
                first = dataset[0][ref].views[AudioView.WAVEFORM][0]
                second = dataset[1][ref].views[AudioView.WAVEFORM][0]

        self.assertTrue(torch.equal(first, torch.tensor([[0.0]])))
        self.assertTrue(torch.equal(second, torch.tensor([[1.0]])))


if __name__ == "__main__":
    unittest.main()
