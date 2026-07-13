# 设计说明

`anydataset` 的核心目标是把物理数据来源、数据集字段映射和训练时字段选择分开。数据集读取阶段尽量保留事实，训练阶段再由用户用 schema 明确声明需求。

## 边界

- `Spec` 只描述物理数据源，包括 source、path、split、version 和 load options。
- `Source` 只负责 prepare 和 raw row iteration，不猜测任务、字段名或语义。
- `Preset` 负责把内置数据集映射到 `Spec`，并把 raw row 转成 canonical `Sample`。
- `Sample` 使用 `(Role, Modality) -> Item` 表达逻辑结构。
- `Schema` 使用 `(Role, Modality) -> Requirement` 表达一次训练或读取真正需要的 view 和 meta。
- `collate_fn(schema)` 只按照 schema 整理 batch，不为缺失字段补隐式默认值。

## Schema 心智模型

`Role` 描述 item 在样本里的位置，例如 `DEFAULT`、`SOURCE`、`TARGET`。同一个样本里有多份同模态数据时，用 role 区分，而不是发明新的字段名。

`Modality` 描述数据类型，例如 `AUDIO`、`TEXT`、`IMAGE`。它决定 item 的类型和可用 view/meta 枚举。

`View` 描述同一份数据的表示方式，例如音频的 waveform、file、LongCat codes 或 DAC codes。新的编码或派生结果应优先作为 view，而不是改写原始 item。

`Meta` 描述旁信息，例如 label、labels、language。meta 必须在 schema 中显式声明后才会进入 batch。

## Preset 和 Task

Preset 应该尽量保留数据集天然提供的信息。例如语音到语音翻译数据集可以同时产出 source audio、target audio、source text 和 target text。是否把这些字段用于训练，由用户 schema 决定。

当前内置 preset 是 `MNIST`、`CIFAR10`、`FLEURS`、`LIBRISPEECH_ASR`、
`COMMON_VOICE`、`ESC50`、`NSYNTH`、`FSD50K` 和 `WMT19`。新增 preset 时只把
物理 `Spec` 和 raw row 到 canonical `Sample` 的映射放进 preset；过滤、模型编码、
训练采样权重等业务规则留在调用方或更高层模块。

`Task` 只适合非常稳定、跨数据集一致的默认 schema，例如图像分类、基础 audio codec 或机器翻译文本对。组合型、研究型或仍在快速变化的任务不应急着进入 `Task` 枚举，用户显式写 schema 更清楚。

## Source 注册

`Source` 枚举只表达核心内置物理来源：`HF`、`HF_DISK` 和 `STORE`。source 注册器还
可以挂载字符串 key；当前内置字符串 key 有 `tsv` 和 `sharded_csv`。

- `tsv` 面向本地表格调试和 Common Voice 本地包，读取文件路径、
  `<path>/<split>.tsv`，或按 `subdirs` load option 的顺序读取各子目录下的同名
  split。Common Voice 默认只选择最新 `cv-corpus-*`，语言目录来自该 corpus；
  如果旧 corpus 有最新 corpus 缺失的语言，preset 显式报错，调用方应手动整理或
  建立符号链接。
- `sharded_csv` 面向已经物理分片的 CSV 目录，读取
  `shard_<index>/<number>.csv` 数字文件名，设置 split 时读取
  `<path>/<split>/shard_<index>/<number>.csv`。非数字 CSV 文件名会被忽略并写
  warning。CSV 保持为可读的事实来源；source prepare 默认在物理 `Spec` cache 下按文件
  并行生成 Parquet part，最后原子提交 manifest。未变化的 part 会复用，变化的 CSV
  只重建对应 part。读取侧通过 Parquet row group 提供 map-style 随机访问，多设备和
  DataLoader worker 由全局 sample index sampler 分片，不重复扫描全部 CSV。

这些字符串 source 可以直接写在 `Spec(source=...)` 里，也可以通过
`resolve_dataset("tsv://...")` 或 `resolve_dataset("sharded_csv://...")`
解析。新 source 只应负责 prepare 和 raw row iteration，不把字段语义塞进 source。

## 派生 View

派生表示应通过 provider 和 `ViewMaterializer` 生成。典型流程是：

```text
base store -> provider -> delta store -> logical merge -> schema selects derived view
```

例如 LongCat codes 是 `AudioView.LONGCAT`。Codec view 的单样本值统一为整数 Tensor
`[frame, codebook]`，collate 后为 `[batch, frame, codebook]`，mask 为
`[batch, frame]`。K 个码本必须完整、有序保存；数据层不区分 semantic / acoustic
codebook，具体码本语义属于 codec 或下游任务。旧的
`{"semantic_codes": ..., "acoustic_codes": ...}` mapping 不属于当前 codec view
契约，进入 collate 时应显式报错。

Preset 不负责加载 codec，也不应该把 LongCat 逻辑塞进 raw row parse。Preset 只需要
产出可被 provider 消费的音频 view，例如 `AudioView.WAVEFORM` 或 `AudioView.FILE`。

`merge()` 是逻辑组合，不修改物理 store。它只接受 map-style dataset，按相同下标取
左右两侧样本并合并 item/view/meta；重复 view 直接报错，重复 metadata 只有值相等
时允许。需要发布自包含 store 时，对合并后的 dataset 显式调用 `write(output_dir)`。
`write()` 支持按 part 并行物化，`num_shards` 控制写进程数，`num_workers` 控制每个
写进程内部的 DataLoader workers；并行写入时调用方应提供可 pickle 的
`dataset_factory`。

store 内部以 `sample_index` 作为样本对齐键。`sample_id` 只用于 manifest 和错误信息
的可读标识，不参与 base store 与 delta store 的对齐；调用方负责保证派生 view 来自
同一顺序的数据集。

`ViewMaterializer` 默认只写 provider 输出 view；如果 delta store 需要额外携带原始
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
`commit_samples` 控制 checkpoint 粒度，默认是 `max(batch_size, 32)`，避免默认可续跑时
产生过多小文件；需要更细断点时可以显式调低。
`write_workers` 控制每个 materializer worker 内的后台写进程数，默认用一个 writer
让 provider 计算和 fragment 落盘重叠；`write_prefetch` 控制待写任务上限。

重模型 provider 或 filter predicate 可以通过 `ProviderServer`、
`RemoteProviderFactory` 和 `RemoteFilterFactory` 常驻独立进程。
server 进程只拥有 provider 和设备状态；materializer 进程只读 dataset、组 batch、
通过 proxy provider 请求 server，并继续负责 fragment、resume 和 commit。需要隔离
CUDA 与数据读取 worker 时，`Runtime(server_start_method="spawn")` 让调用方把 device 当作
路由键，不在写入或过滤进程里设置 torch device；此时 reader worker 默认用 fork 以减少
DataLoader 开销。后台 writer 默认使用 thread backend，让慢速判定或 provider 计算和
落盘重叠，同时避免把 fragment job 通过 process pipe 序列化传输。filter cache 的写入仍由
filter 层负责，predicate server 只负责慢速判定。

## 派生 Modality

同一 role 下缺失的模态应通过 provider 和 `ModalityMaterializer` 生成。provider 只声明输出 view，materializer 用输出 view 推出输出 modality，并在同一 role 中寻找唯一的非输出 modality 作为输入。

如果输出 modality 已经存在，materializer 必须报错；这条路径只负责补缺失模态，不负责覆盖或刷新已有数据。如果同一 role 去掉输出 modality 后还剩多个输入 modality，materializer 也必须报错，调用方应先用 schema 或 transform 明确输入。

`ModalityMaterializer` 生成的新 item 默认不复制 meta。label、language 等跨模态语义继承必须由调用方显式完成，避免库替用户猜测业务规则。

## 过滤分区

过滤规则通过零参数 factory 创建 predicate；factory 在实际执行过滤的进程里调用。
predicate 直接作用在 dataset 产出的完整 canonical `Sample` 上，返回 bool、字符串、
枚举值或带 metrics 的 `FilterDecision`。库统一归一化为字符串 label，并缓存每个
label 对应的原始样本下标。

多设备过滤使用 Python `spawn`，调用方要显式传入可 pickle 的 `dataset_factory`。
库不会把已经构造好的 dataset 实例包进内部闭包再传给子进程。并行读写统一使用
“每个 device 一个进程，进程内可选 DataLoader workers”的模型。

`FilterRule` 的缓存契约只包含 `name`。factory、predicate、parse function 和
transforms 的语义版本不由库检查，调用方应把这些约定写进 `name`。这样 filter
只负责可验证的数据结构、执行设备规划和缓存机制，不把用户业务规则伪装成库能自动
理解的东西。

缓存根目录统一由 `ANYDATASET_HOME` 控制。物理 source prepare cache 写在
`$ANYDATASET_HOME/cache/sources/<spec_id>`，只由 `Spec` 决定。filter cache 写在
`$ANYDATASET_HOME/cache/filters/<dataset_id>/<rule_id>`，其中单物理 dataset 的
`dataset_id` 由 dataset class 和 `Spec.id` 决定，merged map-style dataset 的
`dataset_id` 由排序后的 child identity 决定。`MultipleAnyDataset` 不作为整体建立
filter cache identity；调用方应先对各子 dataset 做过滤或缓存，再组合。

运行时 warning 和 worker 日志同样由 `ANYDATASET_HOME` 控制，写入
`$ANYDATASET_HOME/logs/<timestamp>-<pid>/`。普通 source warning 按来源写成
`<source>.log`，materializer 多进程 worker 日志写在
`materializer/part-xxxxx.log`；rank 0 同时把 worker、provider 加载和物化阶段写到
stderr，便于非交互 job 判断停在哪个阶段。用户级入口不暴露单独的 log root；嵌套 worker 通过
内部配置继承父进程的 run log 目录。

## 质量 Predicate

`anydataset.quality` 下的模块只提供可传给 `FilterRule` 的 predicate 和 profile。
它们不拥有 source、preset、cache root 或训练采样策略。

- `quality.translation` 读取 source/target text，输出 `clean`、`usable`、
  `review`、`reject`，第一版内置 profile 只覆盖 WMT19 `zh-en`。
- `quality.speech` 读取 audio item 和同 role text，输出 `accept` 或 `reject`，
  并把阈值命中、缺字段等审计信息放进 `FilterDecision.metrics`。

如果接入神经网络评估器，模型路径、阈值和版本仍应体现在 `FilterRule.name` 或调用方
配置里；filter cache 不会自动识别这些语义变化。
