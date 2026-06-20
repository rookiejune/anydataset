# TODO

已定下来的设计放在 [docs/design.md](docs/design.md)。这里仅保留接下来要做或需要拍板的事项。

## P1：文档和默认策略

- [ ] HuggingFace streaming 默认策略：
  - 内置大音频/parquet 数据集默认 `streaming=True`。
  - 通用 `hf://...` 不强行 streaming，用户通过 `dataset_map` 或 helper spec 显式设置。
