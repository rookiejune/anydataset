# 合成 S2ST 数据集设计

本文定义通用合成 speech-to-speech translation（S2ST）数据集的逻辑契约。
目标不是为某个 WMT19 实验增加一条特殊流，而是让任意语言来源通过一个通用翻译模型和
一个 TTS 模型持续发布可训练的数据集。

本文中的 `source` 指数据入口或原始来源；`snapshot` 只用于描述已经发布的不可变数据版本，
不出现在用户设备配置中。

## 目标

- workspace 对外只提供一个数据入口，同时具备 `load`、`status`、`generate` 和 `toy` 能力。
- 每个语言可以声明多个有序来源；同一个物理来源允许重复出现，并保留不同的稳定身份。
- 一个 source family 经过同一个通用翻译模型扩展到其他语言。
- 同一个 source family 的源语言和所有目标语言音频使用同一个不可变 voice condition。
- 支持继续遍历已有来源、增加语言、增加来源和增加 speaker；所有扩充都保持旧样本身份、
  顺序和内容不变。
- 生成过程可流水并行，但训练只读取持续增长的最终数据集，不了解生成依赖图。
- 单卡且没有可访问数据时不启动生成模型，改用 schema-compatible toy 数据测真实训练
  model performance。

## 非目标

- `anytrain` 不拥有 canonical `Sample`、source 调度、snapshot、workspace 路径或设备划分。
- `anydataset` 不加载具体 translator、TTS 或 codec checkpoint，也不依赖 workspace。
- codec 不属于首版 S2ST 生成工厂。最终数据发布 waveform；训练进程在剩余训练设备上按
  现有 runtime 规则编码。以后需要离线 codec view 时，继续使用普通 anydataset
  materialization，不改变 S2ST lineage。
- 首版不把一个 source family 的多个目标语言塞进单个 canonical `Sample`。训练视图仍是
  一条 source/target pair。

## 分层和依赖

依赖方向固定如下：

```text
speech-to-speech ──> workspace ──> anydataset
         │                │             │
         └──────────────> anytrain <────┘ optional providers only
```

### anytrain

`anytrain` 提供不依赖数据集类型的模型能力：

- `UniversalTranslator`：输入字符串和语言字符串，输出翻译字符串。
- `TTSBackend`：输入字符串、语言和 speaker id 或 reference audio，输出 `TTSOutput`。
- `AudioTokenizer` / codec：训练或普通物化所需的 waveform 编码能力。
- 通用的 bounded performance report 和长驻服务生命周期。

这些协议只使用字符串、Tensor、`TTSOutput`、`WaveformReference` 等 anytrain 自身类型。
`anytrain` 不导入 anydataset，也不知道 `Sample`、`Role`、source slot 或 S2ST stage。

### anydataset

通用逻辑放在 `anydataset.synthesis.s2st`：

- source slot、source family、pair coordinate 和 voice condition 的稳定身份。
- source 接纳、语言扩充、speaker future-only 更新和确定性调度。
- 三阶段 revision、精确父 snapshot、彼此隔离的 append-only stage catalog 和恢复校验。
- `pairs` / `sources` dataset projection，以及现有 `Schema` 对 text、waveform 和 codec
  views 的选择。
- schema-compatible toy 数据。

模型执行继续使用现有 `SampleProvider` / `BatchSampleProvider`、`SampleMaterializer` 和
provider service；S2ST 不新增一套互不兼容的 provider 协议。

### workspace

workspace 的 `zhuyin.datasets.s2st` 绑定实际资源：

- 每个语言的具体 dataset factory 和 item reference。
- translator/TTS checkpoint、revision、load options 和 capability adapter。
- lineage root、staging 路径、完整性检查和原子发布位置。
- 对外的 `source()`，以及其全量 `load()`、`status()`、`toy()` 和 producer 描述。

workspace 工厂只向 workspace generation 入口领取精确输入 snapshot 和 staging 输出路径，
写完后交回 workspace 校验和发布。工厂不自行推断 latest parent，也不访问训练 DataModule。

### speech-to-speech

训练工程只负责通过 workspace 的 dataset factory 构造普通 Dataset 和 DataLoader。
anydataset/workspace 的显式 producer 调用向共享路径原子发布不可变 snapshot；训练不分配生产设备、
调度 producer，也不读取 generation DAG、catalog 或 snapshot 元数据。

DataModule 不判断某个样本来自哪个 revision。它持有普通 Dataset；anydataset DataLoader 在新的
batch 规划周期刷新同一对象，下一轮自动纳入已发布的 append-only snapshot。

## 公开逻辑对象

### LanguageSource 和 SourceSlot

一个语言包含有序的 source slots：

```python
LanguageSources(
    language=Lang.ZH,
    sources=(
        SourceSlot(
            name="news-zh-primary",
            dataset=zh_news_factory,
            text=(Role.SOURCE, Modality.TEXT),
        ),
        SourceSlot(
            name="news-zh-repeat",
            dataset=zh_news_factory,
            text=(Role.SOURCE, Modality.TEXT),
        ),
    ),
)
```

`SourceSlot.name` 在 lineage 内唯一且不可变。上例两个 slot 可以指向同一个 dataset factory，
但它们拥有独立 cursor、权重和遍历机会，不能按物理来源去重。

每个被接纳的 source row 使用稳定键：

```text
SourceKey(slot_name, row_index)
```

source dataset 中已有的平行译文只属于原始事实，不自动成为合成 target。所有 target text 都由
配置的 universal translator 生成。

### Source family 和 pair

一个 source row 被接纳后形成一个 source family。family 固定保存：

- `SourceKey`、接纳顺序和源语言。
- 源文本，以及 reference 模式下的源音频。
- 接纳时选择的 voice condition 和 speaker pool revision。
- 当前应生成的目标语言集合。

每个目标方向使用稳定键：

```text
PairKey(SourceKey, target_language)
```

不生成 source language 到自身的 pair。语言数为 `L` 时，一个新 family 最终产生 `L - 1`
条 pair。旧 family 在新增语言后会追加一条指向新语言的 pair，但旧 pair 的 sample id、索引和
payload 都不改变。

### Voice condition

S2ST 配置必须二选一：

1. speaker list：TTS 明确支持 speaker ids。family 接纳时从当前 speaker pool 确定性抽取一个
   speaker，并用同一个 TTS、同一个 speaker 合成源语言和所有目标语言音频。
2. reference audio：每个 source slot 必须声明 audio reference。原始 source audio 直接保留；
   所有目标语言使用该音频作为同一个 TTS 的 reference condition。

speaker 或 reference assignment 一旦写入 source snapshot 就不可修改。这里保证的是 conditioning
一致；实际音色一致程度仍属于 TTS 模型质量，不由数据层伪造保证。

新增 speaker 只产生新的 pool revision，并记录
`effective_from_source_sequence=<next admission>`。它只影响后续接纳的 family；旧 family 在横向补新
语言时继续使用原 speaker。随机选择使用 lineage seed、source key 和 pool revision 的 keyed RNG，
实际结果必须持久化，不能在读取时重新计算。

## 增长规模

配置使用 source family 作为增长单位：

```yaml
data:
  generation:
    initial_sources: 8
    interval_sources: 256
```

- `initial_sources`：首个最终 snapshot 最多接纳的 family 数。正式配置至少覆盖一个完整
  optimizer step；overfit 配置可以更小。
- `interval_sources`：后续每个 revision 最多推进的新增或回填 family 数。

日志同时给出 `added_sources`、`added_pairs`、`total_sources` 和 `total_pairs`。不使用
`initial_samples`，因为新增语言后同一个 family 的 pair 数会变化，扁平 sample 数不能稳定表达
调度工作量。

## 扩充顺序

所有声明顺序都是 lineage 契约的一部分。已有项不能删除、换名或重排；同一 lineage 只允许在尾部
增加 language、source slot 和 speaker。

普通纵向增长使用稳定 round-robin：

1. 按 language 声明顺序轮转。
2. 每个 language 内按 source slot 声明顺序轮转。
3. 每个 slot 使用自己的递增 row cursor；耗尽时记录 `exhausted` 并跳过。

新增语言时优先顺序固定为：

1. `old_sources_to_new_language`：按旧 family admission order 为所有旧 source 补新目标语言。
2. `new_language_sources_to_existing_languages`：接纳新语言自己的 source，并生成到所有已有语言。
3. `vertical`：恢复普通 round-robin，继续接纳各语言的新 source。

先完成已在途 revision，再应用新配置。每个 phase 可以跨多个 interval revision，final pair 只在
现有尾部追加，不能为了形成语言矩阵而重排历史数据。

## 三阶段发布

一个 revision 有三个逻辑阶段：

```text
source@r -> translation@r -> tts@r
```

- `source`：冻结本 revision 的 family/pair plan，并确定 voice condition。
- `translation`：精确引用 `source@r`，增加目标文本。
- `tts`：精确引用 `translation@r`，增加 source/target waveform，成为训练可见结果。

三者 revision 相同，但发布时间可以不同，例如：

```text
source.latest = 12
translation.latest = 11
tts.latest = 10
```

下游阶段必须读取 manifest 中的 `upstream_snapshot_id` 和 digest，禁止在运行中再次解析
“当前 latest”。`translation@r` 逻辑包含 `source@r`；`tts@r` 逻辑包含 `translation@r`。
三个阶段分别推进自己的 catalog；默认训练入口只读取 `tts`，调试和下游 producer 可以显式读取
`source` 或 `translation`。

同一模型只对应一个工厂族：source 和 target audio 都由一个 `tts` 工厂生成，不配置
`source_tts` / `target_tts` 两套模型或设备。

## Lineage 和 snapshot manifest

translator checkpoint、TTS checkpoint、voice mode 或 seed 变化时必须创建新 lineage；已有
language/source slot 定义变化时显式失败，不能把不同语义静默混入旧数据。

每个阶段 manifest 至少包含：

```text
lineage_id
config_revision
revision
stage
snapshot_id
upstream_snapshot_id
upstream_digest
previous_snapshot_id
source_count
pair_count
coverage
store_path
store_digest
```

每个物理 store 仍遵守 anydataset standalone store 契约：store 内 sample index 必须是稠密的
`0..N-1`。跨 revision 的逻辑全局索引由 stage catalog 显式映射，不能把稀疏 global index 塞进
store manifest，也不引入 runtime base/delta item overlay。

每个阶段发布一个不可变、完整 canonical delta store。阶段 catalog 按 revision 顺序连接这些 store，
形成 append-only 逻辑 dataset：

- 新 catalog 必须完整保留旧 catalog 的所有 entries。
- 已发布 global sample index 永远映射到同一个 store/local index 和 payload digest。
- catalog 通过 staging、完整性校验和原子替换发布。
- invalid/corrupt 更新必须硬失败，不能继续使用旧版本并假装成功。

每个 catalog entry 还保存该 delta pair index 的精确增量 summary：按 source family 记录稳定
source key/sequence/language/speaker、本次新增 target languages，以及新 family 唯一的 first target。
publisher 从 catalog summary 重放全局 source/pair identity，只读取本次待发布的 records；因此增长时
不会重新打开全部历史 pair index，同时仍会硬校验 pair 唯一性、source identity、dense sequence、
first-for-source 和 added/total counts。完整 JSONL pair index 仍是 sample admission order 的事实来源，
summary digest 同时绑定 index SHA256 和 summary 内容；dataset 每次从头装载时还会对每个 segment
重新从 JSONL records 计算 summary 并核对，summary 不能替代 index payload 校验。

发布时对新的 waveform store 计算完整 content SHA256，并把 store tree 与 pair index 的 stat identity
绑定到 catalog entry。identity 覆盖相对路径、文件类型、mode、device/inode、size、mtime 和 ctime；
后续 publish/load 对历史 store 只扫描 metadata，不再反复读取 waveform bytes。任何普通的
原地写、替换、增删或权限变化都会改变 identity 并硬失败。这个优化依赖已发布 store 的本地
append-only/immutable 文件系统边界；如果部署需要抵抗能够伪造 inode/ctime 的特权协调篡改，应在
存储层增加可信 CAS 或签名，而不能通过跳过完整性校验获得性能。

需要恢复历史 source waveform 的上层 producer 应先调用
`catalog_source_locations(catalog, source_keys)`。summary 直接返回 source 首次发布的 entry 和
store-local index，因此 language backfill 不需要逐 entry 读取历史 pair index。只对返回的 entry 调用
`validate_catalog_entry(root, entry)`，再打开 store 并按 local index 取 sample；不要重新调用
`store_digest(root / entry.store_path)`。entry validation 会检查 index/store 发布时绑定的 immutable
stat identity，但不读取 pair-index 或 waveform payload bytes。真正消费 pair index 的路径仍必须使用
`read_pair_index(root, entry)` 完整核对 SHA256 和 summary。

这个布局避免每次增长复制完整历史 store，同时不违反 standalone store 约束。

每个阶段的 catalog 和锁彼此隔离：

```text
root/catalogs/source.json       # source@r
root/catalogs/translation.json  # translation@r
root/catalog.json               # tts@r（默认训练入口）
```

`source@r` 发布后即可独立读取，即使 `translation@r` 或 `tts@r` 尚未发布；
缺少某个阶段的首个 snapshot 只影响该阶段的 `load(stage=...)`，不会被其他阶段的 catalog
“补齐”。每次 `load()` 都从所选阶段的第一个 snapshot 开始遍历到该阶段当前 latest，
返回可在显式规划边界扩张的普通 map-style dataset。catalog 本身就是逻辑数据集格式，不需要 S2ST 专用
compact/merge；如果下游确实要求单一物理 store，应显式使用 anydataset 的通用物化 workflow。

## Dataset views

首版提供两种 dataset projection：

- `pairs`：每个 `PairKey` 返回 canonical SOURCE/TARGET text+audio sample。
- `sources`：每个 source family 返回唯一 source text+audio sample。

`S2STView` 可按 source language、target language、source slot 和 speaker 过滤；过滤只建立 side
index，不改变 admission order。text-only、waveform、codec codes 等表示继续使用现有 `Schema` 和
`TextView` / `AudioView` 选择，不新增 `texts`、`audio` 之类平行字段系统。

如果以后需要一个 family 内多目标语言的 grouped batch，应新增明确 morphology；不能改变
`Role.TARGET` 的单 item 契约。

## 增量 snapshot 和全量加载

`S2STDataset(root, lineage_id, stage=..., view=...)` 是普通 map-style dataset。构造时必须读取并校验
所选阶段的首个 snapshot；没有首个 snapshot 时立即抛错，不等待后台生产。随后按 revision 从第一个
增量 snapshot 到当前 latest 逐段加载，并组装构造时可见的完整逻辑数据集：

- 每次 `load()` 都从所选阶段 catalog 的第一个 snapshot 开始遍历，不保存 cursor，也不从上次位置续读。
- 已打开对象可通过 `refresh()` 原子吸收 append-only successor；anydataset DataLoader 在每次新的
  batch 规划周期自动调用它。本轮已经生成的 index 保持固定，下一轮才纳入新增 snapshot。
- 多 worker 的副本在收到超出旧视图的非负 index 时本地 refresh；DDP 规划使用所有 rank 当前可见
  逻辑长度的最短前缀。consumer 不等待、不轮询，也不维护 snapshot cursor。
- 上述生命周期由通用 `AppendOnlyMapStyleABC` 持有；S2ST 只实现 catalog successor 的读取和原子
  替换。callable costs 保留旧 global index 的缓存，materialize 时只追加新 suffix。
- 标量、任意顺序和重复 index 的 batch 读取都路由到对应 segment-local index。
- catalog、pair index 或 immutable store identity 损坏时直接失败。
- 没有首个 snapshot 时 `load()` 抛出缺失错误；`status(stage=...)` 可做只读预检。

snapshot 分段属于数据集本身，filter 只按相同分段生成一一对应的 decision，不拥有或创建 S2ST
snapshot 概念；materializer 也复用同一套 snapshot 拼接抽象。

## 统一入口

workspace 入口统一为：

```python
from zhuyin.datasets.s2st import source

dataset_source = source(...)
dataset_source.status()
dataset_source.load()
dataset_source.load(stage="source")
dataset_source.load(stage="translation")
dataset_source.load(stage="tts")
dataset_source.generate()
dataset_source.toy()
```

- `status(stage=...)`：验证并汇总所选阶段 catalog 的全部增量 snapshots，不加载模型；默认阶段为 `tts`。
- `load(stage=...)`：要求所选阶段的首个 snapshot 已发布，从第一个 snapshot 加载到该阶段 latest，
  返回可通过 batch 规划周期自动 refresh 的普通数据集；默认阶段为 `tts`。
- `generate()`：返回 factory 名称、producer command 和 lineage identity，不启动进程、不选择
  设备。
- `toy()`：不加载 translator/TTS，确定性构造与 final `pairs` view 相同 schema 的短 waveform；
  family 内仍使用一致 speaker/reference identity。

正式训练不做 source route 或生产设备切分；所有可见设备由 Lightning 管理。

## 日志

生成侧使用结构化事件：

- `s2st.plan`：`initial`、`vertical`、`language_backfill` 或 `language_sources`，以及 backlog。
- `s2st.stage.waiting|started|finished`：stage、revision、upstream wait、queue wait、compute seconds。
- `s2st.snapshot.published`：stage、revision、parent、added/total source/pair、path 和 digest。
- `s2st.speakers.updated`：old/new count、pool revision、effective source sequence、
  `future_only=true`。
- 每个 source slot 的 cursor 和 exhausted 状态。

训练日志保持普通训练指标、吞吐和 checkpoint。生产任务的 stage 日志由 anydataset/workspace
producer 记录；训练只在 Dataset 构造或读取校验失败时暴露错误。

## 恢复和一致性

- 任一 factory 只能提交 workspace 分配的 staging 路径。
- workspace 发布前校验 snapshot manifest、精确 parent、store integrity、sample coverage 和 digest。
- 重启时从每个 stage 的已发布 latest 和未完成 staging 恢复；下游不重新选择 parent。
- 同一个 revision 的重试必须幂等：相同输入和 provider identity 产生相同 sample ids；不一致结果拒绝发布。
- producer 的通知只用于协调显式调用；阶段 catalog 始终是事实来源，consumer 不运行后台 polling。

## 首版代码范围

首版按以下顺序落地：

1. anytrain 增加纯字符串 translator protocol/backend 和 TTS capability 声明。
2. anydataset 增加 S2ST identity、growth scheduler、views、toy、stage manifest 和 append-only stage catalog。
3. workspace 增加 `zhuyin.datasets.s2st`，把当前 WMT19/Qwen/MOSS 实现改为一个可选 source 配置，
   不再让 producer import speech-to-speech。
4. speech-to-speech 通过 `DatasetConfig.factory` 构造普通 Dataset；batch 周期扩张、生产与 snapshot
   生命周期留在 anydataset。
5. 旧 WMT19 streaming 入口保留一个兼容周期并明确弃用；稳定 waveform/codec store views 不受影响。
