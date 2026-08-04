# 设计说明

`anydataset` 的核心目标是把物理数据来源、数据集字段映射和训练时字段选择分开。数据集读取阶段尽量保留事实，训练阶段再由用户用 schema 明确声明需求。

## 边界

- `Spec` 只描述物理数据源，包括 source、path、split、version 和 load options。
- `Source` 只负责 prepare 和 raw row iteration，不猜测任务、字段名或语义。
- `Preset` 负责把内置数据集映射到 `Spec`，并把 raw row 转成 canonical `Sample`。
- `Sample` 使用 `(Role, Modality) -> Item` 表达逻辑结构。
- `Schema` 使用 `(Role, Modality) -> Requirement` 表达一次训练或读取真正需要的 view 和 meta。
- `collate_fn(schema)` 只按照 schema 整理 batch，不为缺失字段补隐式默认值。

Store reader 对 manifest 和 payload shard 使用带 fingerprint 的进程内句柄缓存；调用方可以
通过 `StoreDataset.close()` 或 context manager 显式释放资源。写入完成的 store 还可以包含
`payload-groups.json` 和每个 tar shard 的 offset index，它们都只是可失效的读取优化
metadata，不改变核心 manifest schema；旧 store 或 sidecar 不可用时 reader 回退到完整
manifest/tar 扫描。

Store payload 默认使用 PyTorch safe weights-only 反序列化，支持 tensor、基础容器和
字符串等常见 view 值。读取包含自定义 Python 对象的旧 store 时，调用方必须确认 store
来自可信来源，并显式传入 `unsafe_pickle_payloads=True` 打开 pickle 反序列化边界。需要
更快的发布校验时可显式选择 integrity `fast` 或 `normal`，默认 `full` 会读取
manifest 引用的 payload body，并拒绝 tar shard 中 manifest 外的额外 payload member。

## Schema 心智模型

`Role` 描述 item 在样本里的位置，例如 `DEFAULT`、`SOURCE`、`TARGET`。同一个样本里有多份同模态数据时，用 role 区分，而不是发明新的字段名。

`Modality` 描述数据类型，例如 `AUDIO`、`TEXT`、`IMAGE`。它决定 item 的类型和可用 view/meta 枚举。

`View` 描述同一份数据的表示方式，例如音频的 waveform、file、LongCat codes、DAC codes 或 speaker-axis metadata。新的编码或派生结果应优先作为 view，而不是改写原始 item。每个 view 必须有稳定且唯一的值类型；例如 `AudioView.WAVEFORM` 始终是 `(waveform_tensor, sample_rate)`，不能在 grouped 数据集中临时变成 `{speaker_id: waveform}` 这类 mapping。需要表达 speaker 轴顺序时使用独立 view，例如 `AudioView.SPEAKERS`。

`Meta` 描述旁信息，例如 label、labels、language。meta 必须在 schema 中显式声明后才会进入 batch。

## Preset 和 Schema

Preset 应该尽量保留数据集天然提供的信息。例如语音到语音翻译数据集可以同时产出 source audio、target audio、source text 和 target text。是否把这些字段用于训练，由用户 schema 决定。

当前内置 preset 是 `MNIST`、`CIFAR10`、`FLEURS`、`LIBRISPEECH_ASR`、
`COMMON_VOICE`、`ESC50`、`NSYNTH`、`FSD50K` 和 `WMT19`，全部为 map-style，通过
`AnyDataset.preset()` 创建。新增 preset 时只把物理 `Spec` 和 raw row 到
canonical `Sample` 的映射放进 preset；过滤、模型编码、训练采样权重等业务规则留在
调用方或更高层模块。Hugging Face `streaming=True` 被显式拒绝。

`FSD50K` 只接受可选的 Hugging Face `revision`，默认值为
`main`。revision 同时进入 `Spec`/source cache identity、Hub 文件列表 URL 和
`hf_hub_download`，保证列举与下载来自同一版本；空 revision 和其他未知 load option
会显式报错。

核心 API 不内置任务枚举。即使是图像分类、基础 audio codec 或机器翻译文本对这类常见布局，也应通过显式 schema helper 表达；组合型、研究型或仍在快速变化的任务尤其不应由底层猜测。

## Source 注册

`Source` 枚举和 source 注册器只表达物理来源类别，不表达具体数据集。具体内置数据集
（例如 `FSD50K`）应放在 preset 附近：preset 负责选择通用物理 source、填充
`Spec`，并把 raw row 解析成 canonical `Sample`。source 层不能为了某个具体数据集
猜测任务、字段名或模态语义。

`Source` 枚举表达核心内置物理来源类别：`HF`、`HF_DISK`、`HF_FILES` 和
`STORE`。source 注册器还可以挂载字符串 key；当前内置字符串 key 有 `tsv` 和
`sharded_csv`。

- `hf-files` 面向 Hugging Face Hub 上的文件树，列举并下载 repo 文件，raw row 只包含
  repo、revision、相对路径和本地缓存路径等物理信息。它不解码音频、不生成 canonical
  字段，也不内置具体数据集规则；例如 `FSD50K` preset 通过
  `path_template="clips/{split}"` 和 `suffixes=(".wav",)` 使用它，再在 preset parser
  中加载 waveform。
- `tsv` 面向本地表格调试和 Common Voice 本地包，读取文件路径、
  `<path>/<split>.tsv`，或按 `subdirs` load option 的顺序读取各子目录下的同名
  split。TSV 保持为可读的事实来源；source prepare 与 `sharded_csv` 共用
  delimited→Parquet 缓存逻辑，在物理 `Spec` cache 下按文件生成 Parquet part，
  读取侧通过 row group 提供 map-style 随机访问，并声明 `ShardingSource`。
  `root_field` 在读取时注入，不写入 Parquet 列。`prepare_workers` 语义与
  `sharded_csv` 相同，且不进入 `Spec.id`。Common Voice 默认只选择最新
  `cv-corpus-*`，语言目录来自该 corpus；如果旧 corpus 有最新 corpus 缺失的语言，
  preset 显式报错，调用方应手动整理或建立符号链接。
- `sharded_csv` 面向已经物理分片的 CSV 目录，读取
  `shard_<index>/<number>.csv` 数字文件名，设置 split 时读取
  `<path>/<split>/shard_<index>/<number>.csv`。非数字 CSV 文件名会被忽略并写
  warning。CSV 保持为可读的事实来源；source prepare 默认在物理 `Spec` cache 下按文件
  并行生成 Parquet part，最后原子提交 manifest。未变化的 part 会复用，变化的 CSV
  只重建对应 part。并发 prepare 只允许一个进程构建，其他进程等待已提交 manifest；
  构建进程退出时由下一个进程接管，超时则显式报错。读取侧通过 Parquet row group
  提供 map-style 随机访问，多设备和 DataLoader worker 由全局 sample index sampler
  分片，不重复扫描全部 CSV。`load_options.prepare_workers=0/1` 可显式禁用 process
pool，默认值保留自动并行策略。`prepare_workers` 只影响转换并行度，不进入 `Spec.id`
或 prepare cache 身份。

这些字符串 source 可以直接写在 `Spec(source=...)` 里，也可以通过
`resolve_dataset("tsv://...")` 或 `resolve_dataset("sharded_csv://...")`
解析。新 source 只应负责 prepare 和 raw row iteration，不把字段语义塞进 source。

需要为 iterable scan 提供高效全局分片时，source 可以实现
`ShardingSource.iter_shard(dataset, *, num_shards, shard_id)`。输出必须是
精确的全局 modulo shard：每项是 `(sample_index, raw_row)`，`sample_index` 从
`shard_id` 开始并按 `num_shards` 稠密递增。入口会拒绝非二元 tuple、bool/非整数索引和
已产出序列中的缺口、重复或错 shard 索引。单个未知长度 iterable shard 无法独立证明
末尾完整覆盖或 row/index 对应关系，这两项仍由 source 契约负责，并在已知 sample count
的下游 commit 中继续校验。这个契约由 source 显式声明；raw dataset 上恰好存在
`shard()` 或 `iter_shard()` 不构成 opt-in，避免把 native shard 内从零开始的
局部编号误当成全局 cache 对齐键。

`hf-disk`、`store`、`tsv` 和 `sharded_csv` 的 prepared dataset 支持按全局下标随机读取，
因此实现该契约且不扫描其他 shard。Hugging Face `streaming=True` 在 `Source.HF`
入口显式拒绝；需要可索引本地数据时使用 `Source.HF_DISK`。
dataset 层的 `iter_shard()` 是唯一的全局 modulo shard 语义，产出
`(sample_index, sample)`。`IterableAnyDataset` 不会机会主义地调用 raw dataset
的 `shard()`。

Map-style `iter_runtime_shard` 在多卡时会丢弃不能被 `rank_count` 整除的尾部样本；
iterable 路径不做该截断。

## 派生 View

派生表示应通过 provider 和 `ViewMaterializer` 生成。典型流程是：

```text
base store -> provider -> standalone store -> schema selects derived view
```

例如 LongCat codes 是 `AudioView.LONGCAT`。Codec view 的单样本值统一为整数 Tensor
`[frame, codebook]`，collate 后为 `[batch, frame, codebook]`，mask 为
`[batch, frame]`。K 个码本必须完整、有序保存；数据层不区分 semantic / acoustic
codebook。经 `CodecProvider` 生成时，第 k 列的每个 id 必须满足
`0 <= id < codebook_sizes[k]`；provider 在输出设备上逐码本检查该值域，越界时在
写入前显式报错。store manifest 不保存 `codebook_sizes`，因此直接读取 store 时不做该
值域校验；已有 view 进入 collate 时也只检查通用 tensor 契约。具体码本语义属于 codec
或下游任务。旧的
`{"semantic_codes": ..., "acoustic_codes": ...}` mapping 不属于当前 codec view
契约，进入 collate 时应显式报错。

Preset 不负责加载 codec，也不应该把 LongCat 逻辑塞进 raw row parse。Preset 只需要
产出可被 provider 消费的音频 view，例如 `AudioView.WAVEFORM` 或 `AudioView.FILE`。

`ViewMaterializer` 和 `ModalityMaterializer` 都直接发布 standalone store。默认只保留
provider 输出；需要携带输入 view/meta 时，通过 `keep_schema` 显式选择。这样训练和
filter 都只读取一个 store，不存在运行时 base/delta overlay，也不依赖两个 dataset
的 sample index 顺序一致。需要多个派生 view 时，按阶段对上一个 standalone store
继续物化，并显式保留下一阶段所需字段。

`write()` 支持按 part 并行物化，`num_shards` 控制写进程数，`num_workers` 控制每个
写进程内部的 DataLoader workers；并行写入时调用方应提供可 pickle 的
`dataset_factory`。
store 的 map-style `__getitem__` 保持全局下标随机访问语义，不隐式改变采样顺序。
训练时如果需要避免样本级 shuffle 频繁跨 tar shard，仍使用统一的
`dataset.dataloader(..., shuffle=...)` 入口。底层 `StoreDataset` 通过私有 `_shuffle`
生成 payload-shard-local 读取计划：先按已选择 view manifest 的 shard group 排序或
shuffle，再在 group 内 shuffle 样本；batch planner 只在同一个 group 内组 batch。
`sharded_csv` 和 `tsv` 使用 Parquet row group 作为同类 index group，避免全量样本
index list 和跨 row group 随机读取；其他 map-style dataset 使用有界连续 index group。planner
只维护 `planning_window` 个候选，DDP 按 `distributed_plan_window` 个 plan 的有界 chunk
同步并只裁剪 rank-local 最终 batch 尾部，不修改通用 `iter_shard()` 的 modulo 契约。
默认 DDP 同步窗口保持较小，以降低 cost lookup 或 batch packing 较重时的 first-batch
等待；需要定位卡点时可用 `ANYDATASET_DEBUG_DDP_PLANS=1` 打开 chunk 级日志。
reader 默认只接受字段和 Parquet manifest 结构完整的 `schema_version: 3` store；
v2 store 没有 provenance，属于 legacy compatibility 格式，只能通过显式
`legacy_policy="warn"` 或 `"allow"` 读取。`warn` 会发出明确的
`RuntimeWarning`，默认 `reject` 保持发布和 cache-sensitive 路径的安全边界。
v2 缺失的 provenance 只能按空值参与 identity，不能静默猜测 input/provider 语义；
发布 store 或生成 cache-sensitive 派生数据前应重新物化或迁移到 v3。新写入的 v3 store
在 dataset manifest 中保存 materializer 的 `input_id` 和 `provider_id`。更早格式必须先显式迁移或重新物化，
不在读取时按 `sample_id` 静默补齐。
离线 `migrate_store(source, output)` 只处理真实存在过的上一版 canonical 布局：dataset
manifest 没有 `schema_version`（也接受显式的 `1`），sample manifest 已包含稠密稳定的
`sample_index`，view manifest 使用 `sample_id`。迁移通过 sample manifest 的唯一 ID
映射生成 v3 view manifest，把引用的 payload shard 复制到独立新目录，完成 v3 结构、
覆盖范围和 tar key 校验后才原子发布。源目录不原地修改。更早的 view revision 目录、
JSONL manifest 或不同 sample item 结构不属于该 v1 契约，只能从原始 canonical dataset
重新物化。

`ViewMaterializer` 默认只写 provider 输出 view；如果 standalone store 需要额外携带原始
item 的少量 view 或 meta，调用方通过 `keep_schema` 显式声明。`keep_schema` 使用现有
`Schema`/`Requirement` 语义，只复制声明的字段；如果声明的 view 和 provider 输出冲突，
materializer 必须报错，不静默覆盖。

`ViewMaterializer` 和 `ModalityMaterializer` 默认使用可续跑的 fragment 流水线。
库会把完成的 provider batch 聚合到 checkpoint chunk 后写成 ready fragment，并按全局
`sample_index` 跳过已完成样本；所有样本覆盖后再原子提交最终 store 并清理 resume
目录。fragment 仍使用普通 store 校验，损坏或语义不匹配时显式报错，不静默丢弃。
多设备 materializer 只负责为每个设备启动独立进程并按 rank 分片，不初始化
`torch.distributed` process group；需要 collective 的 provider 负责显式创建和释放
自己的 process group，并定义各 rank 的同步契约。
`commit_samples` 控制 checkpoint 粒度，默认是 `max(batch_size, 1024)`，限制尚未落盘的
provider 输出数量；需要更细断点时可以显式调低。最终提交会逐 payload 读取这些 fragment，
按 `max_shard_samples` 流式重新打包 tar，因此 checkpoint 粒度不会决定发布 store 的 shard
数量，也不需要把一个完整目标 shard 的输出同时保留在内存中。只有一个 fragment 时继续
复用原 shard，避免小数据集无谓重写。
调用方可以通过 `max_new_samples` 或显式递增的 `sample_indexes` 配合
`finalize=False`，只完成本轮计划范围并保留同一份完整输入 identity 下的 resume
fragment。最终一次调用不再传范围并使用默认 `finalize=True`，只补齐缺失 index，覆盖
完整输入后原子发布 store。限定范围不允许同时 finalize，避免多设备 worker 生成不完整
part。需要读取阶段性稠密前缀时，可以从 resume fragment 发布独立 snapshot；snapshot
校验 dataset/provider identity，不修改 resume 状态。
resume metadata 与本次运行不兼容时，旧目录会先原子重命名为相邻的
`.<output>.resume.stale-*` 目录，避免在共享文件系统上同步递归
删除大量 fragment 阻塞新任务。运行日志会记录该目录；确认不再需要后由调用方清理。
metadata 使用稳定的 factory 标识，并只记录影响 fragment 语义或格式的配置；device
数量、进程启动方式和 commit 粒度可以在续跑时调整，不会使已完成 fragment 失效。
factory 标识会纳入函数代码、默认参数、closure 值和 callable 实例状态；其中 Tensor
按真实内容生成固定长度摘要，不把权重展开进 resume JSON。调用方还可以设置 `input_id`
和 `provider_id`，为可调用对象无法表达的输入快照或外部模型 checkpoint 提供显式语义版本；
两个 ID 与自动 factory 标识共同参与兼容性判断，不会替代后者。任一标识变化都会隔离
旧 resume 目录，避免复用语义不一致的 fragment。
`write_workers` 控制每个 materializer worker 内的后台写线程数，默认用一个 writer
让 provider 计算和 fragment 落盘重叠；`write_prefetch` 控制待写任务上限。
每个 materializer rank 的 writer 完成 fragment 后，会在 rank 进程内把确定性分配给
自己的 fragment 归并成一个 rank part。主进程最终只归并这些 part，并负责全局索引
覆盖校验、统一 manifest 和原子 ready 标记，不再逐个扫描全部 fragment。
`ModalityMaterializer.roles` 可以显式限制生成角色；未设置时处理全部符合条件的 role。
角色选择属于 materializer 语义 identity，变更后不能复用旧 resume fragment。

重模型 provider 或 filter predicate 可以通过 `ProviderServer`、
`RemoteProviderFactory` 和 `RemoteFilterFactory` 常驻独立进程。
server 进程只拥有 provider 和设备状态；materializer 进程只读 dataset、组 batch、
通过 proxy provider 请求 server，并继续负责 fragment、resume 和 commit。需要隔离
CUDA 与数据读取 worker 时，`Runtime(server_start_method="spawn")` 让调用方把 device 当作
路由键，不在写入或过滤进程里设置 torch device。reader/writer 的 `auto` 始终解析为
spawn；只有调用方验证目标平台和设备边界后，才显式选择 fork。显式 start method 始终
覆盖 `auto`。后台 writer 默认使用 thread
backend，让慢速判定或 provider 计算和落盘重叠，同时避免把 fragment job 通过 process
pipe 序列化传输。filter cache 的写入仍由
filter 层负责，predicate server 只负责慢速判定。

## 派生 Modality

同一 role 下缺失的模态应通过 provider 和 `ModalityMaterializer` 生成。provider 只声明输出 view，materializer 用输出 view 推出输出 modality，并在同一 role 中寻找唯一的非输出 modality 作为输入。

如果输出 modality 已经存在，materializer 必须报错；这条路径只负责补缺失模态，不负责
覆盖或刷新已有数据。如果同一 role 去掉输出 modality 后还剩多个输入 modality，
materializer 也必须报错，调用方应先用 schema 或 transform 明确输入。

provider 可以声明 `reference_role` 作为唯一例外。该 role 必须已经有输出 modality，且不会
再次生成输出；它的 view 会与其他 role 的唯一输入 modality 合并后交给 provider，例如给
目标文本 TTS 提供源 role 的参考音频。reference role 不存在或缺少输出 modality 时显式
报错，其他 role 已经有输出 modality 时仍然报错。

`ModalityMaterializer` 生成的新 item 默认不复制 meta。label、language 等跨模态语义继承必须由调用方显式完成，避免库替用户猜测业务规则。

### Speaker 条件

`TextView.SPEAKERS` 是文本生成语音时使用的 speaker 条件 view。单条未 batch 的
`TextItem` 中它始终是一个非空 speaker id 字符串；collate 后才成为字符串序列。它不是
文本事实 metadata，也不绑定具体 TTS provider。

`SpeakerIdDataset` 通过 `(Role, Modality.TEXT) -> SpeakerAssignment` mapping 为任意
map-style canonical dataset 的一个或多个 text item 增加 speaker view，且不改变样本数量。
每个 text reference 独立选择 assignment：aligned speaker 序列必须与 dataset 等长，cycle
则按样本 index 循环使用非空 speaker 序列。`SpeakerCartesianDataset` 只对一个显式 text
reference 做 text × speakers 展开，避免多 role 笛卡尔积的组合语义被库隐式猜测。

`GroupedSpeakerAudioDataset` 可以把这种 flat speaker grid 按原 text index 聚合回 speaker
轴，并用 `AudioView.SPEAKERS` 和 `AudioView.SPEAKER_LENGTHS` 描述轴顺序和未 padding
长度。flat 长度必须能被 speaker 数整除；同一组内的 source index、speaker 顺序、text
内容、采样率和 waveform channel shape 必须一致，waveform 必须是 `[channel, time]`。
flat audio 如含 `AudioMeta.SPEAKER_ID`，它也必须与 text speaker view 一致。聚合结果只用
`AudioView.SPEAKERS` 表达 speaker 轴，不把 speaker tuple 写进单值 speaker meta。这些物理
store 契约不满足时读取会明确报错。

`SpeakerAudioGrid` 在同一个 text-major flat store 上保留 `text × speaker` 二维语义。
`cells` 暴露原 flat dataset，`rows` 暴露 `GroupedSpeakerAudioDataset`；grid 本身的
`__len__` 和 `__getitem__` 委托给 rows，保持 map-style 单 sample 读取。`select(text=...)`、
`select(speaker=...)` 和 `select()` 分别惰性选择一行、一列和整个网格，也可以同时指定两个
轴选择一个 cell。`SpeakerAudioSelection.load(view=...)` 只读取选择范围并按指定的
`AudioItem.views[view]` 做 padding，返回 `SpeakerAudioBlock`。当选中 cells 只有一个可合并
的 audio view 时可以省略 `view`；存在多个 view 时必须显式指定。block 用 `[text, speaker]`
lengths 保存当前 view padding 前的主 unit 长度。waveform 额外保留 `waveforms` 和
`sample_rate` 便捷属性；`block.audio_view` 只是本次 load 解析出的 view，不是 grid 的配置状态。
block 同时保留
grid 内的 `text_indices`，以及每个 text row 的 `source_indices`、`roles` 和 `texts`。
`SpeakerAudioRow` 定义 row 到原始 source index 和 role 的映射；未显式传入 row specs 时，
grid 使用 row index 和 text ref role。整网格 load 是显式的 eager 操作，调用方负责控制
内存规模。speaker id 在 grid 中必须非空且唯一，选择不隐式重排物理 speaker 轴。

frame codec view 的 block 值是 `[text, speaker, unit, codebook]` Tensor。BiCodec 使用
`AudioView.BICODEC` 的 `semantic` / `acoustic` mapping，两个值分别保留自己的 unit 轴；
speaker lengths 描述可变 semantic unit，固定 acoustic unit 不广播到 semantic frame。

TTS provider 只消费已经存在的 `TextView.TEXT`、`TextView.SPEAKERS` 和可选语言 meta。
例如 Qwen provider 不负责 speaker 分配；调用方先组合 speaker dataset，再使用
`ModalityMaterializer` 为相同 role 增加 audio item。

## 过滤分区

过滤规则通过零参数 factory 创建 predicate；factory 在实际执行过滤的进程里调用。
predicate 直接作用在 dataset 产出的完整 canonical `Sample` 上，返回 bool、字符串、
枚举值或带 metrics 的 `FilterDecision`。库统一归一化为字符串 label，并缓存每个
label 对应的原始样本下标。
predicate 可选实现 `call_batch(samples)`；返回值必须是与输入等长、按位置对应的有序
sequence。未实现时仍逐样本调用 `__call__`。batched predicate 遇到 CUDA OOM 时会先
释放异常引用并清理 CUDA cache，再按原顺序递归二分 batch 重试；每个子 batch 都重新
校验输出契约。非 OOM 不重试，单样本仍 OOM 时原样抛出。

在线 `RejectReplaceDataset` 不是缓存分区的替代品。它只对已经物化的 map-style
dataset 做廉价 CPU reject 替换：顺序 look-ahead，失败后再用 worker 本地 accept
buffer 回填，并在拒绝率偏高时 warning 或硬失败。GPU / 重模型质量规则仍走
`FilterRule.apply`；详见 [`online_filter.md`](online_filter.md)。

多设备过滤按 `Runtime.process_start_method` 启动外层进程，默认值是 `"spawn"`；默认
配置下调用方要显式传入可 pickle 的 `dataset_factory`。库不会把已经构造好的 dataset
实例包进内部闭包再传给子进程。并行读写统一使用“每个 device 一个进程，进程内可选
DataLoader workers”的模型；只解析出一个 device 时不启动外层 device worker。

`FilterRule` 保留 `name` 作为可读名称和缓存 identity 的兼容字段，并支持显式 `rule_id` 和
`version`。三者都会进入缓存路径、metadata、pickle/factory 恢复和 equality；修改其中任一项
都会选择不同缓存。修改 predicate、factory、parse function 或 transforms 时应更新版本。
未提供新字段的旧调用继续使用 name-only cache path。

缓存根目录统一由 `ANYDATASET_HOME` 控制。物理 source prepare cache 写在
`$ANYDATASET_HOME/cache/sources/<spec_id>`，只由 `Spec` 决定。filter cache 写在
`$ANYDATASET_HOME/cache/filters/<dataset_id>/<rule_id>`，其中物理 dataset 的
`dataset_id` 由 dataset class、`Spec` 和 store manifest provenance 决定；filtered view
会把上游 identity、rule 和 label 纳入 identity。对于内容或顺序由业务工程管理的输入，
调用方用非空 `input_id` 版本化整个输入快照，并在输入变化时更新。
store 的 `AudioView.FILE` payload 解包到
`$ANYDATASET_HOME/cache/store-files/<store_id>`，cache identity 包含 view、shard、
payload key 和 shard fingerprint，写入使用同目录原子替换。派生文件被外部清理后会按需
重新解包；dataset 不再为每个访问过的 FILE 样本维护无界进程内路径表。
包含 FILE view 的 store reader 会在
`$ANYDATASET_HOME/cache/store-file-leases/<store_id>.lock` 持有跨进程共享 lease。
`cleanup_store_files(store_root)` 只在取得独占 lease 后删除该 store 的解包目录；存在活动
reader 或 `lease_store_files(store_root)` 显式 lease 时直接报错，不静默跳过。调用方只保留
返回的字符串路径而释放所有 lease 后，路径不再受保证；需要跨 reader 生命周期使用时必须
显式持有 lease，或把文件复制到调用方拥有的目录。缓存不做自动容量淘汰，清理必须由调用方
按物理 store 显式触发，后续访问会重新解包。
物理 dataset 使用自动 identity；业务输入的内容或顺序改变时，调用方必须更新非空
`input_id`。该 ID 补充而不替代自动 class、`Spec`、provenance 和 sample count identity。
`FilterRule.version` 版本化 predicate，`input_id` 版本化输入状态；filtered factory、pickle
重建和链式过滤会继续携带上游 ID。

运行时 warning 和 worker 日志同样由 `ANYDATASET_HOME` 控制，写入
`$ANYDATASET_HOME/logs/<timestamp>-<pid>/`。普通 source warning 按来源写成
`<source>.log`，filter 和 materializer 的多进程 worker 日志分别写在
`filter/part-xxxxx.log` 和 `materializer/part-xxxxx.log`。两者的 rank 0 都把普通 worker
生命周期写到 stdout，ERROR 及以上写到 stderr；filter 额外报告 scan/writer 进度，materializer 报告
reader/provider/writer 进度，便于非交互 job 判断停在哪个阶段。进度写入 `stdout`；
用户级入口不暴露单独的 log root，
嵌套 worker 通过内部配置继承父进程的 run log 目录。

## 质量规则

`anydataset.quality` 下的模块只提供可传给 `FilterRule` 的规则类和 profile 类。
它们不拥有 source、preset、cache root 或训练采样策略。

- `quality.text.TextQuality` 只检查单个文本 item。
- `quality.text.TextAcceptability` 和 `quality.text.ChineseGEC` 是可选模型规则，
  仍按单个 role/text item 输出自己的 label、flags 和指标。
- canonical 文本语言用 `TextMeta.LANG -> Lang` 表达；preset 或 parser 入口用
  `remap_lang(...)` 把数据集原始标签映射成 enum，quality 层不再接收裸字符串。
- `quality.translation.TranslationQuality` 只检查 source/target pair，输出
  `accept` 或 `reject`。
- `quality.rules.QualityChain` 按调用方传入的顺序组合原子规则，并负责
  `accept`、`review`、`reject` 的链式转移。
- `quality.speech.SpeechQuality` 读取 audio item 和同 role text，输出 `accept` 或 `reject`，
  并把阈值命中、缺字段等审计信息放进 `FilterDecision.metrics`。
- `SpeechQuality(codec_provider=provider)` 只读取 `provider.output` 的 frame codes，通过
  `provider.codec.decode()` 评估 codec 重建音频，不回退原始 waveform。同长度 codes 在
  predicate batch 内合并解码，speech evaluator 仍按原样本顺序逐条执行。

如果接入神经网络评估器，模型路径、阈值和版本应体现在 `FilterRule.rule_id`、
`FilterRule.version` 或调用方配置里；filter cache 不会自动推断这些语义变化。
