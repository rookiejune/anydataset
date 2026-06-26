import tempfile
import unittest
from pathlib import Path

import torch

from anydataset import (
    AnyDataset,
    AudioItem,
    AudioView,
    Modality,
    Role,
    Source,
    Spec,
)
from anydataset.store import DatasetWriter, ViewMaterializer, read_store_dataset
from anydataset.store.paths import view_dir


class ViewMaterializerTest(unittest.TestCase):
    def test_materializer_writes_only_provider_output_by_default(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            source = root / "source"
            target = root / "target"
            waveform = torch.tensor([[1.0, 2.0, 3.0]])
            dataset = _source_dataset(source, root, [_audio_sample(waveform)])

            ViewMaterializer(target, dataset_id="toy-audio", split="train").write(
                dataset,
                _Provider(offset=10),
            )

            stored = read_store_dataset(target)
            sample = _read_sample(target, root)

            self.assertEqual(
                set(stored.views),
                {(Role.DEFAULT, Modality.AUDIO, AudioView.LONGCAT)},
            )
            self.assertFalse(
                view_dir(
                    target,
                    (Role.DEFAULT, Modality.AUDIO, AudioView.WAVEFORM),
                ).exists()
            )
            self.assertEqual(set(sample[Role.DEFAULT, Modality.AUDIO].views), {AudioView.LONGCAT})
            self.assertTrue(
                torch.equal(
                    sample[Role.DEFAULT, Modality.AUDIO]
                    .views[AudioView.LONGCAT]["semantic_codes"],
                    torch.tensor([[11, 12, 13]]),
                )
            )

    def test_materializer_can_copy_inputs(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            source = root / "source"
            target = root / "target"
            waveform = torch.tensor([[1.0, 2.0, 3.0]])
            dataset = _source_dataset(source, root, [_audio_sample(waveform)])

            ViewMaterializer(
                target,
                dataset_id="toy-audio",
                split="train",
                copy_inputs=True,
            ).write(dataset, _Provider())

            sample = _read_sample(target, root)
            audio = sample[Role.DEFAULT, Modality.AUDIO]

            self.assertEqual(set(audio.views), {AudioView.WAVEFORM, AudioView.LONGCAT})
            loaded_waveform, sample_rate = audio.views[AudioView.WAVEFORM]
            self.assertTrue(torch.equal(loaded_waveform, waveform))
            self.assertEqual(sample_rate, 4)

    def test_materializer_processes_multiple_roles(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            source = root / "source"
            target = root / "target"
            dataset = _source_dataset(
                source,
                root,
                [
                    {
                        (Role.SOURCE, Modality.AUDIO): AudioItem(
                            views={AudioView.WAVEFORM: (torch.tensor([[1.0, 2.0]]), 4)}
                        ),
                        (Role.TARGET, Modality.AUDIO): AudioItem(
                            views={AudioView.WAVEFORM: (torch.tensor([[3.0, 4.0]]), 4)}
                        ),
                    }
                ],
            )

            ViewMaterializer(target, dataset_id="toy-audio").write(
                dataset,
                _Provider(offset=10),
            )

            stored = read_store_dataset(target)
            sample = _read_sample(target, root)

            self.assertEqual(
                set(stored.views),
                {
                    (Role.SOURCE, Modality.AUDIO, AudioView.LONGCAT),
                    (Role.TARGET, Modality.AUDIO, AudioView.LONGCAT),
                },
            )
            self.assertTrue(
                torch.equal(
                    sample[Role.SOURCE, Modality.AUDIO]
                    .views[AudioView.LONGCAT]["semantic_codes"],
                    torch.tensor([[11, 12]]),
                )
            )
            self.assertTrue(
                torch.equal(
                    sample[Role.TARGET, Modality.AUDIO]
                    .views[AudioView.LONGCAT]["semantic_codes"],
                    torch.tensor([[13, 14]]),
                )
            )


def _source_dataset(path: Path, root: Path, samples):
    DatasetWriter(path, dataset_id="toy-audio", split="train").write(samples)
    return AnyDataset(
        Spec(source=Source.STORE, path=str(path), split="train"),
        cache_root=root / "cache-source",
    )


def _read_sample(path: Path, root: Path):
    dataset = AnyDataset(
        Spec(source=Source.STORE, path=str(path), split="train"),
        cache_root=root / "cache-target",
    )
    return dataset[0]


def _audio_sample(waveform: torch.Tensor):
    return {
        (Role.DEFAULT, Modality.AUDIO): AudioItem(
            views={AudioView.WAVEFORM: (waveform, 4)},
        )
    }


class _Provider:
    output = AudioView.LONGCAT

    def __init__(self, *, offset=0):
        self.offset = offset

    def __call__(self, views):
        waveform, _ = views[AudioView.WAVEFORM]
        return {"semantic_codes": waveform.to(torch.int64) + int(self.offset)}


if __name__ == "__main__":
    unittest.main()
