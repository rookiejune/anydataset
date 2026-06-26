from io import BytesIO
import shutil
import tarfile
import tempfile
import unittest
from pathlib import Path

import torch

from anydataset import (
    AnyDataset,
    AudioItem,
    AudioKey,
    AudioView,
    Modality,
    Role,
    Source,
    Spec,
    ViewRef,
)
from anydataset.store import DatasetWriter, ViewInput, ViewMaterializer
from anydataset.store.jsonio import read_json
from anydataset.store.manifest import DatasetManifest
from anydataset.store.manifestio import read_view_manifest
from anydataset.store.paths import dataset_json_path, view_dir, view_shard_path


class ViewMaterializerTest(unittest.TestCase):
    def test_materializer_writes_only_new_view_by_default(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            source = root / "source"
            target = root / "target"
            waveform = torch.tensor([[1.0, 2.0, 3.0]])
            DatasetWriter(source, dataset_id="toy-audio", split="train").write(
                [_audio_sample(waveform)]
            )
            input_ref = ViewRef(Modality.AUDIO, AudioView.WAVEFORM)
            output_ref = ViewRef(Modality.AUDIO, AudioView.LONGCAT)

            def provider(view: ViewInput):
                return {
                    "semantic_codes": view.value.to(torch.int64) + 10,
                    "sample_id": view.sample.sample_id,
                }

            ViewMaterializer(
                input_dir=source,
                output_dir=target,
                input_ref=input_ref,
                output_ref=output_ref,
                transform=provider,
                provider_name="toy_longcat",
                provider_version="1",
                config={"offset": 10},
            ).write()

            manifest = DatasetManifest.from_dict(read_json(dataset_json_path(target)))
            revision = next(
                selection.revision
                for selection in manifest.views
                if selection.ref == output_ref
            )
            entry = next(read_view_manifest(target, output_ref, revision))

            self.assertEqual({selection.ref for selection in manifest.views}, {output_ref})
            self.assertFalse(view_dir(target, input_ref, "raw").exists())
            self.assertEqual(entry.ref, output_ref)
            with tarfile.open(
                view_shard_path(target, output_ref, entry.revision, entry.shard),
                "r",
            ) as archive:
                payload = archive.extractfile(entry.key).read()
            loaded = torch.load(BytesIO(payload), map_location="cpu")
            self.assertTrue(
                torch.equal(loaded["semantic_codes"], torch.tensor([[11, 12, 13]]))
            )
            self.assertEqual(entry.provenance["name"], "toy_longcat")
            self.assertEqual(entry.provenance["config"], {"offset": 10})

    def test_materializer_can_register_new_view_in_place(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            source = root / "source"
            waveform = torch.tensor([[1.0, 2.0, 3.0]])
            DatasetWriter(source, dataset_id="toy-audio", split="train").write(
                [_audio_sample(waveform)]
            )
            input_ref = ViewRef(Modality.AUDIO, AudioView.WAVEFORM)
            output_ref = ViewRef(Modality.AUDIO, AudioView.LONGCAT)

            ViewMaterializer(
                input_dir=source,
                output_dir=source,
                input_ref=input_ref,
                output_ref=output_ref,
                transform=lambda view: {"semantic_codes": view.value.to(torch.int64)},
                provider_name="toy_longcat",
                provider_version="1",
            ).write()

            manifest = DatasetManifest.from_dict(read_json(dataset_json_path(source)))
            revision = next(
                selection.revision
                for selection in manifest.views
                if selection.ref == output_ref
            )
            entry = next(read_view_manifest(source, output_ref, revision))

            self.assertEqual(
                {selection.ref for selection in manifest.views},
                {input_ref, output_ref},
            )
            self.assertTrue(view_dir(source, input_ref, "raw").exists())
            self.assertTrue(view_dir(source, output_ref, revision).exists())
            with tarfile.open(
                view_shard_path(source, output_ref, entry.revision, entry.shard),
                "r",
            ) as archive:
                payload = archive.extractfile(entry.key).read()
            loaded = torch.load(BytesIO(payload), map_location="cpu")
            self.assertTrue(torch.equal(loaded["semantic_codes"], torch.tensor([[1, 2, 3]])))

    def test_materializer_self_contained_mode_copies_existing_views(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            source = root / "source"
            target = root / "target"
            waveform = torch.tensor([[1.0, 2.0, 3.0]])
            DatasetWriter(source, dataset_id="toy-audio", split="train").write(
                [_audio_sample(waveform)]
            )
            input_ref = ViewRef(Modality.AUDIO, AudioView.WAVEFORM)
            output_ref = ViewRef(Modality.AUDIO, AudioView.LONGCAT)

            ViewMaterializer(
                input_dir=source,
                output_dir=target,
                input_ref=input_ref,
                output_ref=output_ref,
                transform=lambda view: {"semantic_codes": view.value.to(torch.int64)},
                provider_name="toy_longcat",
                provider_version="1",
                mode="self_contained",
            ).write()
            shutil.rmtree(source)

            dataset = AnyDataset(
                Spec(source=Source.UNIFIED, path=str(target), split="train"),
                cache_root=root / "cache",
            )
            sample = dataset[0]
            manifest = DatasetManifest.from_dict(read_json(dataset_json_path(target)))

            self.assertEqual(
                {selection.ref for selection in manifest.views},
                {input_ref, output_ref},
            )
            audio = sample[Role.DEFAULT, Modality.AUDIO]
            self.assertTrue(torch.equal(audio.views[AudioView.WAVEFORM], waveform))


def _audio_sample(waveform: torch.Tensor):
    return {
        (Role.DEFAULT, Modality.AUDIO): AudioItem(
            views={AudioView.WAVEFORM: waveform},
            required={AudioKey.SAMPLE_RATE: 4},
        )
    }


if __name__ == "__main__":
    unittest.main()
