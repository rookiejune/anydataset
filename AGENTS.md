# anydataset Agents

## 架构边界

- `Spec` 只描述物理数据源：source、path、split、version 和 load options。
- `Source` 只负责 prepare 和 raw row iteration，不在底层猜测任务或字段名。
- `Preset` 负责把内置数据集映射到具体 `Spec`，并通过 `parse_fn` 把 raw row 转成 canonical `Sample`。
- `Sample` 统一使用 `Mapping[tuple[Role, Modality], Item]`，不要恢复旧的 wrapper / `.data` 结构。
- `AnyDataset` 表示 map-style 数据集；`IterableAnyDataset` 表示 iterable 数据集。
- `MultipleAnyDataset` 只组合已经构造好的 dataset，迭代顺序交给 `IterationStrategy`；组合本身不作为 filter cache 身份。
- store 的公开入口是 `DatasetWriter`、`ViewMaterializer`、`ViewRef` 和 `Source.STORE`。

## 开发约定

- 新增内置数据集时，在 `src/anydataset/presets/` 下增加 preset class，并在 `Preset.create()` 中注册。
- 具体数据集字段映射写在 preset 的 `parse_fn` 或清晰 helper 里，不写进 source 层。
- 不新增旧式适配器、格式化器、流包装器、规格别名或模态别名。
- 不做静默兼容旧 manifest 结构；格式变更时通过 schema version 和显式迁移处理。
- PyTorch `DataLoader` 的 batching 和 collate 由调用方显式配置，库内 dataset 只产出单条 canonical `Sample`。
- 大数据集默认用 streaming 时，把选择放到 preset 的 `Spec.load_options` 里。

## 多 worker 和多卡

- source 如果原生支持 sharding，优先使用 source 原生 shard；否则用 index modulo 作为保底。
- source prepare cache path 只由物理 `Spec` 决定，不能因为 task、schema 或 sample metadata 不同而分裂。
- filter cache path 由 dataset identity 和 `FilterRule.name` 决定；单 spec dataset 使用 class + spec id，merged dataset 使用排序后的 child identity。
