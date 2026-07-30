# anydataset Agents

## 架构边界

- `Spec` 只描述物理数据源：source、path、split、version 和 load options。
- `Source` 只负责 prepare 和 raw row iteration，不在底层猜测任务或字段名。
- `Preset` 负责把内置数据集映射到具体 `Spec`，并通过 `parse_fn` 把 raw row 转成 canonical `Sample`。
- `Sample` 统一使用 `Mapping[tuple[Role, Modality], Item]`，不要恢复旧的 wrapper / `.data` 结构。
- `AnyDataset` 表示 map-style 数据集；`IterableAnyDataset` 表示 iterable 数据集。
- `MultipleAnyDataset` 只组合已经构造好的 dataset，迭代顺序交给 `IterationStrategy`；组合本身不作为 filter cache 身份。
- store 的公开入口是 `DatasetWriter`、`ViewMaterializer`、
  `ModalityMaterializer`、provider 类型和 `Source.STORE`。part、fragment、manifest
  读写与 commit helper 都是内部实现。
- `dataset.morphology` 提供跨下游复用的 audio 样本形态契约：`audio` / `speech` /
  `speech_grid` 的 batch、utterance collate 和 grid view。它不持有 Lightning
  DataModule、Hydra、LBA 或 teacher 训练附件；那些留在 anycodec / speech-to-speech。
  `speech` 强制 waveform + text，speaker 仅为可选 meta；speaker 对照轴属于
  `speech_grid`。`SpeechGridBatch` 为 `[B, S, Text, C, time]`，轴标签是
  `tuple[tuple[str | None, ...], ...]`（每 sample 独立；`None` 表示未知但保留
  轴对应）；**不**默认 flatten，也不从 Speech batch 事后聚合。

## 开发约定

- 新增内置数据集时，在 `src/anydataset/presets/` 下增加 preset class，并按数据集类型注册到 `AnyDataset.preset()` 或 `IterableAnyDataset.preset()`。
- 具体数据集字段映射写在 preset 的 `parse_fn` 或清晰 helper 里，不写进 source 层。
- 不新增旧式适配器、格式化器、流包装器、规格别名或模态别名。
- 不做静默兼容旧 manifest 结构；格式变更时通过 schema version 和显式迁移处理。
- PyTorch `DataLoader` 的 worker/LBA/Lightning 装配由调用方显式配置；形态 collate
  （`audio_collate` / `speech_collate` / `speech_grid_collate` / `SpeechGridView`）
  属于可选公开契约，不为缺失 speaker/text 静默补默认值。
- 大数据集默认用 streaming 时，把选择放到 preset 的 `Spec.load_options` 里。

## 多 worker 和多卡

- source 如果原生支持 sharding，优先使用 source 原生 shard；否则用 index modulo 作为保底。
- source prepare cache path 只由物理 `Spec` 决定，不能因为 task、schema 或 sample metadata 不同而分裂。
- filter cache path 由 dataset identity 和 `FilterRule.name` 决定；物理 dataset 使用 class、`Spec` 和 store provenance，业务管理的输入用显式 `input_id` 版本化。
- `RejectReplaceDataset` 只做低拒绝率的在线 CPU 安全网，不进入 filter cache identity；主力质量筛选仍用 `FilteredDataset`。
