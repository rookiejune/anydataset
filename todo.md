# TODO

已定下来的设计放在 [docs/design.md](docs/design.md)。这里仅保留接下来要做或需要拍板的事项。

## P0：统一格式最小闭环

- [ ] 固化统一格式目录和 manifest schema：
  - 一个输出目录必须是自包含 dataset，不依赖原始路径、父 dataset 或任何外部引用。
  - MVP 先使用 `dataset.json`、`samples.jsonl` 和 `manifest.jsonl`，暂不引入 parquet。
  - 顶层目录直接使用 `audio/`、`text/` 等模态名，不额外套 `modalities/`。
  - role 是可空结构化字段；空 role 不写成 `default` 目录，例如默认音频路径是 `audio/views/waveform/<revision>/`。
  - 非空 role 写在模态下面，例如 `audio/source/views/waveform/<revision>/`、`audio/target/views/longcat/<revision>/`。
  - `views` 是保留路径段，role 不能命名为 `views`。
  - `dataset.json` 记录 schema version、dataset id、split、样本数、每个 `(modality, role, view_key)` 选用的 revision、创建配置和输入 provenance。
  - 全局 sample manifest 记录稳定 `sample_id`、原始 `dataset_name` / `sample_index`、可审计来源、modality、role、duration、sample_rate、label、text 等轻量字段。
  - 每个 view revision 目录包含 `view.json`、`manifest.parquet` 或 `manifest.jsonl`、`shards/*.tar` 和 `.ready`。
  - `view.json` 和 view manifest 都记录结构化的 `modality`、`role`、`view_key` 和 `revision`，路径只是持久化细节。
  - view manifest 记录 `sample_id`、shard/key、shape、dtype、checksum 和 provider provenance。
- [x] 新增 store schema / path helper：
  - 建议新建 `anydataset.store` 或 `anydataset.view_store` 包。
  - 复用 `ViewRef` 表达 `(modality, role, view_key)`，不要在 writer / reader 里手写路径字符串。
  - 定义 `DatasetManifest`、`SampleManifestEntry`、`ViewManifestEntry` 等小 dataclass。
  - 增加 JSON 读写 helper，写入时使用临时文件加原子 rename。
- [x] 实现 `DatasetWriter` MVP：
  - 输入接受逻辑上已经加载好的 canonical dataset iterator，每条 sample 包含全部可用模态、key 和已有 view。
  - 写入到新的空目录；目标目录已存在且非空时直接报错，不做静默合并。
  - 先支持默认 role 的 audio views：`waveform` 和 `file`。
  - `DatasetWriter` 负责分配稳定 `sample_id`、写全局 sample manifest、写选择保留的 view manifest 和 shard payload。
  - 写入流程使用临时目录加原子 rename；只有全局 manifest、view manifest、checksum 都完成后才写 `.ready`。
  - 写入失败时留下可删除的临时目录，不产生半 ready 的 dataset。
- [x] 实现统一格式读取器 MVP：
  - 新增 `UnifiedDatasetAdapter`，读取 `dataset.json`、sample manifest 和指定 view manifest。
  - 支持通过 `source="unified"` 或 `unified:///path:split` 读取统一格式目录。
  - MVP 支持默认 role 的 `audio.views.waveform` 和 `audio.views.file`；file view 解包到 cache 并返回本地路径。
  - 读取时只加载已 ready 的 view；显式指定缺失 view 直接报错，不静默生成。
  - 支持按 sample manifest 顺序迭代，并保留现有 `iter_indexed_samples(...)` sharding 语义。
- [x] 补统一格式 MVP 测试：
  - 写入器 smoke test：从小型 canonical iterator 写出完整 dataset，并校验 manifest、view manifest、payload 和 `.ready`。
  - 读取器 smoke test：按指定 views 读取，缺失 view 报错，sharding 后 sample 不重复。
  - 写出再读回 smoke test：`samples -> DatasetWriter -> UnifiedDatasetAdapter -> samples`。

## P1：View Materializer

- [x] 实现统一格式 view materializer：
  - 输入是一个已有统一格式 dataset 和目标新路径，输出仍是完整自包含统一格式 dataset。
  - materializer 显式声明输入 `(modality, role, view_key)`、输出 `(modality, role, view_key)`、provider 配置和 provider 版本；缺少输入 view 时立即报错。
  - 生成新 view 时同时复制或重写需要保留的旧 views，最终目标路径不引用旧路径。
  - 新 view 的 revision 由 provider 名称、版本、配置和输入 view revisions 共同决定。
  - 普通训练读取默认 CPU-only，不在 DataLoader worker 中隐式加载 GPU codec。
- [x] 补 materializer 测试：
  - materializer smoke test：从已有统一格式 dataset 生成带 toy `longcat` 的新路径，并确认新路径不依赖旧路径。
  - existing view replacement smoke test：用新 revision 替换 dataset 当前选择的 `waveform` view。
- [x] 增加 LongCat view provider / materializer：
  - 输入接受统一格式 dataset，每条 sample 至少有一个可用 audio view、`audio.sample_rate`，以及可选的 `text`。
  - 使用 `anytrain.codec.longcat.LongCatAudioCodec` 或外部传入的 LongCat codec，生成 `longcat` view，包含 `semantic_codes` 和 `acoustic_codes`。
  - LongCat 依赖保持 optional/lazy import，普通 `anydataset` 安装不拉取 LongCat 或 `anytrain[longcat]`。
  - provider smoke test 使用外部 fake codec 验证 materializer 输出；真实权重 smoke 依赖本地或远程 LongCat ckpt。
- [x] 用真实 LongCat 权重跑一次短音频 smoke：
  - 2026-06-25 在 121 的 GPU 0 上使用 `LongCat-Audio-Codec/demos/org/common.wav` 截 1 秒，输出 `semantic_codes=(1, 17)`、`acoustic_codes=(1, 1, 17)`。
- [ ] DAC 这类不区分 semantic/acoustic 的 codec 后续作为并列 view provider 单独设计。

## P1：训练迭代和 Sharding

- [ ] 重新设计训练态 epoch / sharding 语义：
  - `AnyDataset` / `DatasetSource` 提供统一的 `set_epoch(epoch)` 入口；epoch 语义属于 dataset / iteration strategy，不依赖 PyTorch sampler。
  - PyTorch sampler 只适用于 map-style dataset 的 index 顺序；`IterableDataset` 路径下不要要求用户自己区分或编写 sampler。
  - 保留 `.shard(num_shards, shard_id)` 作为固定、可复现的显式分区 API，用于 eval、离线处理或明确的 rank 分区；训练默认路径需要 epoch-aware shuffle / shard 分配，避免每个 rank 或 worker 永远只看到固定 residue class。
  - `DataLoader(num_workers>0)` 和 DDP 的 rank / worker 切分继续在 dataset 内部处理；普通 worker 不要求用户显式 `.shard()`。
  - 需要考虑 `persistent_workers=True` 时 epoch 如何传播到 worker 内的 dataset 拷贝，必要时使用共享状态或明确限制。
  - 为 Lightning 提供 callback/helper，在 `on_train_epoch_start` 显式调用 `dataset.set_epoch(trainer.current_epoch)`；不能假设 Lightning 会自动调用 dataset 的 `set_epoch`。
- [ ] 为 adapter 增加能力声明，按能力选择最合适的训练迭代策略：
  - 纯 streaming：使用 buffer shuffle、数据源顺序 shuffle 和流式 rank / worker split。
  - 有文件、row group 或 sample manifest：先按 `seed + epoch` shuffle 轻量 manifest / shard 列表，再按 rank / worker 分配，payload 保持懒加载。
  - 支持稳定 `__len__ + __getitem__` 的 backend 可额外暴露 map-style dataset；只有这种出口才交给 `DistributedSampler` 或自定义 sampler。

## P2：后续优化

- [ ] 评估将 `samples.jsonl` / `manifest.jsonl` 替换或补充为 parquet。
- [ ] 评估 WebDataset tar shards、`.npy`、`.npz`、`safetensors` 等 payload backend。
- [ ] 设计更通用的 normalizer / transform / pipeline，逐步替代 formatter 作为主链路概念。
