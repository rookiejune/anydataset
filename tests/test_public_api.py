from __future__ import annotations

import json
import subprocess
import sys

import anydataset
import anydataset.dataset as dataset
import anydataset.dataset.source as source
import anydataset.filter as filter_api
import anydataset.provider as provider
import anydataset.provider_service as provider_service
import anydataset.quality as quality
import anydataset.presets as presets
import anydataset.runtime as runtime
import anydataset.store as store
import anydataset.types as any_types


def _assert_public_all(module, expected: list[str], *, private_allowed: set[str] | None = None) -> None:
    private_allowed = private_allowed or set()
    assert module.__all__ == expected
    for name in module.__all__:
        assert hasattr(module, name), name
        if name not in private_allowed:
            assert not name.startswith("_"), name


def _assert_not_exported(module, names: list[str]) -> None:
    for name in names:
        assert name not in module.__all__
        assert not hasattr(module, name), name


def test_top_level_public_api_boundary() -> None:
    _assert_public_all(
        anydataset,
        [
            "AnyDataset",
            "FilterRule",
            "IterableAnyDataset",
            "Lang",
            "Preset",
            "Source",
            "Spec",
            "__version__",
            "anydataset_home",
            "register_source",
            "remap_lang",
            "resolve_dataset",
        ],
        private_allowed={"__version__"},
    )
    _assert_not_exported(
        anydataset,
        [
            "has_source",
            "create_source",
            "source_exists",
            "DatasetWriter",
            "read_store_dataset",
        ],
    )


def test_source_public_api_boundary() -> None:
    _assert_public_all(
        source,
        [
            "DatasetSource",
            "HuggingFaceDiskSource",
            "HuggingFaceFilesSource",
            "HuggingFaceSource",
            "ShardedCsvSource",
            "ShardingSource",
            "StoreSource",
            "TsvSource",
            "register_source",
        ],
    )
    _assert_not_exported(
        source,
        [
            "FSD50KSource",
            "FSD50KDataset",
            "ShardedCsvDataset",
            "TsvDataset",
            "SourceFactory",
            "prepare_hf",
            "prepare_hf_disk",
            "for_source",
            "has_source",
            "create_source",
            "source_exists",
        ],
    )


def test_types_public_api_boundary() -> None:
    _assert_public_all(
        any_types,
        [
            "AudioItem",
            "AudioMeta",
            "AudioReq",
            "AudioView",
            "ImageItem",
            "ImageMeta",
            "ImageReq",
            "ImageView",
            "Item",
            "ItemTransform",
            "Lang",
            "Modality",
            "Meta",
            "Preset",
            "Reference",
            "Requirement",
            "Role",
            "Sample",
            "Schema",
            "SemanticAcousticView",
            "Source",
            "SourceKey",
            "Spec",
            "TextItem",
            "TextMeta",
            "TextReq",
            "TextView",
            "Transforms",
            "View",
            "item",
            "remap_lang",
        ],
    )


def test_presets_public_api_boundary() -> None:
    _assert_public_all(
        presets,
        [
            "CIFAR10",
            "CommonVoice",
            "ESC50",
            "FSD50K",
            "Fleurs",
            "LibriSpeechASR",
            "MNIST",
            "NSynth",
            "WMT19",
        ],
    )
    _assert_not_exported(presets, ["preset_spec", "create_map_preset"])


def test_presets_load_concrete_modules_lazily() -> None:
    code = """
import json
import sys
import anydataset.presets as presets

prefix = "anydataset.presets."
before = sorted(
    name for name in sys.modules
    if name.startswith(prefix) and name != f"{prefix}registry"
)
presets.MNIST
after = sorted(
    name for name in sys.modules
    if name.startswith(prefix) and name != f"{prefix}registry"
)
print(json.dumps({"before": before, "after": after}))
"""
    output = subprocess.check_output([sys.executable, "-c", code], text=True)
    loaded = json.loads(output)

    assert loaded["before"] == []
    assert loaded["after"] == ["anydataset.presets.mnist"]


def test_dataset_public_api_boundary() -> None:
    _assert_public_all(
        dataset,
        [
            "AnyDataset",
            "AudioBatch",
            "Batch",
            "FieldGroup",
            "FieldRef",
            "IterableAnyDataset",
            "IndexSelection",
            "MapStyleABC",
            "Morphology",
            "SpeechBatch",
            "SpeechGridBatch",
            "SpeechGridView",
            "audio_collate",
            "build_toy_audio_dataset",
            "build_toy_speech_dataset",
            "build_toy_speech_grid",
            "collate_fn",
            "field_lengths",
            "load_audio_file",
            "prepare_audio",
            "speech_collate",
            "speech_grid_batch",
            "speech_grid_collate",
        ],
    )


def test_filter_public_api_boundary() -> None:
    _assert_public_all(
        filter_api,
        [
            "BatchFilterPredicate",
            "DatasetFactory",
            "FilterApplyKwargs",
            "FilterApplyReport",
            "FilterApplyResult",
            "FilterDecision",
            "FilterFactory",
            "FilteredDataset",
            "FilterLabel",
            "FilterPredicate",
            "FilterRule",
            "RejectReplaceDataset",
            "cleanup_filter_generations",
        ],
    )
    _assert_not_exported(filter_api, ["filter_identity", "filter_path"])


def test_quality_public_api_boundary() -> None:
    _assert_public_all(
        quality,
        [
            "Bicleaner",
            "ChineseGEC",
            "QualityChain",
            "QualityLabel",
            "QualityRule",
            "Rule",
            "Scorer",
            "SpeechQuality",
            "SpeechQualityProfile",
            "TextAcceptability",
            "TextQuality",
            "TextQualityProfile",
            "TranslationQuality",
            "TranslationQualityProfile",
        ],
    )


def test_runtime_public_api_boundary() -> None:
    _assert_public_all(runtime, ["AutoStartMethod", "Runtime"])


def test_store_public_api_boundary() -> None:
    _assert_public_all(
        store,
        [
            "DatasetWriter",
            "BatchModalityProvider",
            "BatchModalityTransform",
            "BatchOutput",
            "BatchViewProvider",
            "BatchViewTransform",
            "FunctionModalityProvider",
            "FunctionViewProvider",
            "ModalityMaterializer",
            "MaterializationStatus",
            "StoreFilesInUseError",
            "cleanup_store_files",
            "lease_store_files",
            "migrate_store",
            "ModalityProvider",
            "ModalityTransform",
            "Provider",
            "SampleMaterializer",
            "validate_store_payloads",
            "validate_store_view_payloads",
            "ViewMaterializer",
            "ViewProvider",
            "ViewTransform",
        ],
    )
    _assert_not_exported(
        store,
        [
            "read_store_dataset",
            "read_store_manifest",
            "DatasetPartWriter",
            "DatasetFragmentWriter",
            "commit_store_parts",
            "commit_store_fragments",
            "sample_manifest_writer",
            "read_view_manifest",
            "payload_groups",
        ],
    )


def test_provider_public_api_boundary() -> None:
    _assert_public_all(
        provider,
        [
            "CodecProvider",
            "LongCatProvider",
            "MossTTSProvider",
            "QwenTTSProvider",
            "WhisperASRProvider",
        ],
    )
    _assert_not_exported(
        provider,
        [
            "AudioProvider",
            "ProviderServer",
            "RemoteProviderFactory",
        ],
    )


def test_provider_service_public_api_boundary() -> None:
    _assert_public_all(
        provider_service,
        [
            "ProviderServer",
            "RemoteFilterFactory",
            "RemoteFilterPredicate",
            "RemoteProvider",
            "RemoteProviderError",
            "RemoteProviderFactory",
        ],
    )
    _assert_not_exported(
        provider_service,
        [
            "_ProviderCommand",
            "_ProviderError",
            "_ProviderRequest",
            "_ProviderResponse",
            "_ProviderServerConfig",
            "_provider_error",
            "_serve_connection",
            "_accept_connection",
        ],
    )
