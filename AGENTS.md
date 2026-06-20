# anydataset Agents

## 架构边界

- `tasks/<task_name>/` 定义任务的统一 canonical sample schema 和 per-sample formatter。
- task 只接受 canonical sample，不猜测外部数据集字段名，不维护 alias 表，不处理某个具体数据集的类别映射。
- `datasets/<dataset_name>/` 定义具体数据集如何下载、缓存、准备和流式读取 raw item。
- `datasets/<dataset_name>/adapters/<task_name>.py` 定义该数据集如何适配某个 task，把 raw item 映射到该 task 的 canonical sample。
- `api/` 负责用户入口、resolver、cache、mixing、iteration strategy 和 dataset/task 的组装，不写具体任务或具体数据集的业务规则。
- 顶层 `anydataset.*` 可以保留便捷导出，但新实现应优先放在上述目录中。

## Dataset / Strategy / Formatter

- `DatasetSource` 只负责单个 resolved `DatasetSpec` 的 prepare/cache、task adapter、per-sample formatter、shard 和 sample iteration。
- `AnyDataset` 持有一个或多个 `DatasetSource`，对外仍表现为普通 `IterableDataset`。
- `AnyDataset.shard()` 应调用每个 child dataset 的 `shard()`。
- 不在 `AnyDataset` 上提供 `.dataloader()`；用户需要 batching 时直接使用 PyTorch `DataLoader` 和自己的 `collate_fn`。
- 多数据源迭代顺序由 `IterationStrategy` 决定，例如顺序、轮询、加权随机。
- `SampleFormatter` 只负责改写单条 `Sample`；不要在 formatter 中做 pad、stack 或 dataclass batch collate。
- 采样率、声道数和 clip 截断属于对应 per-sample formatter，例如 `AudioCodecFormatter`。

## 音频 codec 约定

- 音频 codec task 的 canonical sample 字段是 `waveform`、`sample_rate` 和可选 `text`。
- 是否有 `text` 由对应 `datasets/<dataset_name>/adapters/audio_codec.py` 决定；task 层不按 music/speech/environment 分类。
- 采样率、声道数和 clip 截断属于 `tasks/audio_codec` 的 per-sample formatter；padding/collate 不属于 formatter。
- 数据集字段名、嵌套 audio 对象和转录文本字段名属于对应 `datasets/<dataset_name>/adapters/audio_codec.py`。
- 第一批内置音频数据集是 `fleurs`、`librispeech_asr`、`esc50`、`nsynth` 和 `fsd50k`；只保留这些规范名字，不额外增加短 alias。
- HF/parquet 分片数据集默认开启 `streaming=True`，让后台 queue 可以边下载边消费。
- FSD50K 使用 HF repo 中的单个 wav 文件清单，按 shard 逐个下载；FMA-small 这种大 zip 数据集不要伪装成 streaming，除非先有稳定的分片来源。

## 开发要求

- 新增数据集时，优先建立 `datasets/<dataset_name>/`，不要把数据集特例写进 task。
- 新增任务时，优先建立 `tasks/<task_name>/`，并为需要支持的数据集补对应 dataset adapter。
- API 层只组合，不内联 task 或 dataset 的映射规则。
- 保持小步重构；如果移动模块，尽量用兼容 wrapper 避免外部导入一次性断裂。

## 流式读取和队列

- 大数据集不应在主训练迭代路径上同步下载、解码和取样。
- API 层应使用 bounded queue 做后台预取：dataset stream 负责生产 sample，mixer 优先从已有样本的队列里取。
- 单个慢数据源未就绪时，不应阻塞其它已有样本的数据源。
- 如果后续需要慢数据源不阻塞快数据源，应实现 prefetching iteration strategy，而不是恢复内部 dataloader/collator。
- 队列只能吸收生产/消费速度的短期波动；如果所有 producer 都没有产出，或训练长期快于数据生产，消费侧仍然需要等待或超时失败，不能伪造 batch。

## 多 worker 和多卡

- `IterableDataset` 在 PyTorch `DataLoader(num_workers>0)` 和 DDP 多进程下会被复制，必须在 API 层自动 sharding，避免 worker 或 rank 重复消费同一批样本。
- 全局 shard id 应按 `rank * num_workers + worker_id` 计算，`num_shards = world_size * num_workers`。
- DDP 下优先读取 `torch.distributed` 的 rank/world size；未初始化时可以回退到 `RANK` 和 `WORLD_SIZE` 环境变量。
- dataset adapter 如果支持原生 sharding，应优先在 adapter 内实现，减少远程下载和过滤浪费；API 层的 index modulo sharding 是保底行为。
- PyTorch `DataLoader` 的 `batch_size` 和 `collate_fn` 由外部训练代码控制；库内 dataset 只产出单条 `Sample`。
- DDP 训练中各 rank 的 batch 数可能因有限数据源分片不均而不同；需要严格同步步数时应使用 `drop_last=True` 或上层固定 `steps_per_epoch`。
- 多 worker / 多 rank 可以同时创建 dataset stream，但同一个 raw dataset cache path 的首次 `adapter.prepare()` 必须由跨进程文件锁保护，只有一个进程做首次下载或 materialize，其它进程等待 `.ready` 后从 cache 构造自己的 manifest。
- cache path 应只由 raw dataset 的 source/path/name/split/version/options/loader 类型决定，不能因为 task adapter 或 sample metadata 不同而分裂成多个下载目录。
- `adapter.prepare()` 的约定是：如果需要写共享 cache，应在返回前完成 cache materialize；如果只是远程流式读取，应自己实现原生 sharding 或接受 API 层 modulo sharding 的带宽浪费。
