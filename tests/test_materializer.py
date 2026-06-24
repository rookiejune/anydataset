import shutil
import tempfile
import unittest
from io import BytesIO
from pathlib import Path
import tarfile

import torch

from anydataset import (
    AnyDataset,
    AudioKey,
    AudioView,
    DatasetSpec,
    ModalityKey,
    Task,
    ViewRef,
)
from anydataset.samples import Sample
from anydataset.store import (
    DatasetManifest,
    DatasetWriter,
    ViewInput,
    ViewManifestEntry,
    ViewMaterializer,
    dataset_json_path,
    read_json,
    read_jsonl,
    view_manifest_path,
    view_shard_path,
)


class ViewMaterializerTest(unittest.TestCase):
    def test_materializer_adds_new_view_and_output_is_self_contained(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            source = root / "source"
            target = root / "target"
            waveform = torch.tensor([[1.0, 2.0, 3.0]])
            DatasetWriter(source, dataset_id="toy-audio", split="train").write(
                [
                    Sample(
                        data={
                            ModalityKey.AUDIO: {
                                AudioKey.SAMPLE_RATE: 4,
                                AudioKey.VIEWS: {
                                    AudioView.WAVEFORM: waveform,
                                },
                            },
                        },
                        dataset_name="source:train",
                        sample_index=0,
                    )
                ]
            )
            input_ref = ViewRef(ModalityKey.AUDIO, AudioView.WAVEFORM)
            output_ref = ViewRef(ModalityKey.AUDIO, AudioView.LONGCAT)

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
            shutil.rmtree(source)

            dataset = AnyDataset(
                datasets=DatasetSpec(source="unified", path=str(target), name="target"),
                task=Task.AUDIO_CODEC,
                cache_dir=root / "cache",
            )
            sample = next(iter(dataset))
            manifest = DatasetManifest.from_dict(read_json(dataset_json_path(target)))
            longcat_revision = next(
                selection.revision for selection in manifest.views if selection.ref == output_ref
            )
            longcat_entry = ViewManifestEntry.from_dict(
                next(read_jsonl(view_manifest_path(target, output_ref, longcat_revision)))
            )

            self.assertTrue(torch.equal(sample.data[ModalityKey.AUDIO][AudioKey.VIEWS][AudioView.WAVEFORM], waveform))
            self.assertEqual({selection.ref for selection in manifest.views}, {input_ref, output_ref})
            self.assertEqual(longcat_entry.ref, output_ref)
            with tarfile.open(
                view_shard_path(target, output_ref, longcat_entry.revision, longcat_entry.shard),
                "r",
            ) as archive:
                payload = archive.extractfile(longcat_entry.key).read()
            loaded = torch.load(BytesIO(payload), map_location="cpu")
            self.assertTrue(torch.equal(loaded["semantic_codes"], torch.tensor([[11, 12, 13]])))
            self.assertEqual(longcat_entry.provenance["name"], "toy_longcat")
            self.assertEqual(longcat_entry.provenance["config"], {"offset": 10})

    def test_materializer_can_replace_selected_revision_for_existing_view(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            source = root / "source"
            target = root / "target"
            DatasetWriter(source, dataset_id="toy-audio", split="train").write(
                [
                    Sample(
                        data={
                            ModalityKey.AUDIO: {
                                AudioKey.SAMPLE_RATE: 4,
                                AudioKey.VIEWS: {
                                    AudioView.WAVEFORM: torch.tensor([[1.0]]),
                                },
                            },
                        },
                        dataset_name="source:train",
                        sample_index=0,
                    )
                ]
            )
            ref = ViewRef(ModalityKey.AUDIO, AudioView.WAVEFORM)

            ViewMaterializer(
                input_dir=source,
                output_dir=target,
                input_ref=ref,
                output_ref=ref,
                transform=lambda view: view.value * 2,
                provider_name="double_waveform",
                provider_version="1",
            ).write()

            dataset = AnyDataset(
                datasets=DatasetSpec(source="unified", path=str(target), name="target"),
                task=Task.AUDIO_CODEC,
                cache_dir=root / "cache",
            )
            sample = next(iter(dataset))
            manifest = DatasetManifest.from_dict(read_json(dataset_json_path(target)))

            self.assertEqual(len(manifest.views), 1)
            self.assertNotEqual(manifest.views[0].revision, "raw")
            self.assertTrue(
                torch.equal(
                    sample.data[ModalityKey.AUDIO][AudioKey.VIEWS][AudioView.WAVEFORM],
                    torch.tensor([[2.0]]),
                )
            )


if __name__ == "__main__":
    unittest.main()
