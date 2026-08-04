# Performance Notes

本文记录 `anydataset` 当前性能优化的讨论范围、实验顺序和阶段一基准。这里先记录
局部 benchmark 结果；真实数据集和目标机器上的最终结论仍应沉淀到实验结果文档。

## 当前边界

- `ViewMaterializer` 和 `FilterRule` 都依赖稳定的全局 `sample_index`，用于分片写入、
  resume fragment、filter partition 和后续 standalone store 的覆盖校验。
- 外层 device/provider worker 继续保持 spawn-friendly。provider 可能加载 CUDA 模型，
  不应为了 DataLoader 读取性能把外层进程模型改成 fork。
- 外层扫描 worker、server 和 reader 的 start method 分开配置。
  reader/writer 的 `auto` 始终使用 spawn；即使 provider 由独立 server 持有，也只有调用方
  验证目标平台和设备边界后才显式选择 fork。显式 reader/writer 配置始终覆盖 `auto`。
  后台 writer 默认使用 thread backend；只有显式使用 process writer backend 时才读取
  `writer_start_method`。
- 公开入口默认值优先保证跨 provider、跨平台的可用性和可恢复性，而不是把某个
  workspace 的生产吞吐配置固化为库默认值。高吞吐任务应在调用方 workflow/job wrapper
  中显式设置 `batch_size`、`num_workers`、`prefetch_factor`、
  `write_workers`、`write_prefetch`、`commit_samples` 或 `num_shards`，并把
  依赖具体数据集、存储和 GPU 的 benchmark 结论留在对应项目文档中。
- 默认用户数据集以 map-style 为主。`StoreDataset`、`FilteredDataset` 这类默认
  map-style shard 语义的 materializer/filter 热路径会使用 map-style indexed
  loader；`AnyDataset` 仍优先保留 source-aware `iter_shard` 路径，避免把顺序
  source 退化成随机访问。`tsv` 和 `sharded_csv` prepare 后使用按源文件生成的
  Parquet cache，因此也走 map-style indexed loader。
- iterable source 只有显式实现全局 `iter_shard` 契约时才会走 source-native
  路径；契约要求返回原始全局 `sample_index`，并由入口校验已产出值的稠密 modulo
  序列。Hugging Face `streaming=True` 已拒绝；不要依赖 streaming 做训练吞吐。
- store 格式保持稳定；reader 侧可以只读 parquet metadata 和轻量 index 列，按 row group
  懒加载 sample/view manifest 的完整行。`preload=True` 仍表示显式加载并校验所有 view
  manifest。

## 运行时验证结论

- 真实 LongCat 请求中，一次 Unix socket `PING` 的中位数是 0.135 ms；persistent
  connection 能省掉的只会更少，不足以抵消新增的 owner、fork、重连和 shutdown 状态。
  `ProviderServer` 保持一请求一连接。
- server-owned provider 配合显式 fork reader 的 materializer 和 filter 路径已在目标
  Linux Python 3.9 环境通过；这项结果不改变跨平台默认值。
- 分布式动态 batch 验证已确认：存在稳定 index 时可使用 metadata-only final flush，
  无 index 时才需要回退到 object gather；anydataset 的当前 dataloader 直接沿用全局
  index 契约，不再新增独立集成层。

环境、命令和完整测量见
[`001_runtime_followups.md`](experiments/results/001_runtime_followups.md)。真实 provider
入口是 `scripts/benchmark_provider_server.py`。

## 已落地

- `anydataset._runtime.parallel.map_style_sample_index_loader` 使用 rank sampler 分发全局 sample index，
  并由 `MapStyleSampleIndexDataset` 返回 `(sample_index, sample)`。
- wrapper 可以在当前进程复用已构造 dataset；spawn 序列化时丢弃该缓存，让 worker 通过
  `dataset_factory` 懒加载重建。
- `ViewMaterializer` 和多设备 `FilterRule` 对默认 map-style shard 语义的数据集使用该
  loader；source 显式提供全局 `iter_shard` 的 iterable dataset 继续走 runtime loader。
- dataset 层 `iter_shard` 是 index-preserving 的 dense global modulo 分区。原生加速
  只通过 source 的 `ShardingSource` opt-in，不调用 raw dataset 的 `shard()` 或局部
  分片方法。内建 `hf-disk`、`hf-files`、`store`、`tsv` 和 `sharded_csv` 通过随机访问
  实现该路径。`Source.HF` 拒绝 `streaming=True`；Hub 文件树使用
  `Source.HF_FILES`。
- reader/writer worker 的 `auto` 始终使用 spawn，避免静默继承 torch/CUDA/provider
  状态。独立 server 拓扑若已验证 fork 安全，可由调用方显式覆盖 start method。
- `StoreDataset` 打开时不再把 `samples.parquet` 全量转成 Python tuple；`samples` 保留
  sequence 接口，并按 parquet row group 懒加载完整 sample manifest 行。
- `AnyDataset.from_store(..., views=...)` 在顶层选择训练所需 view，pickle/spawn 后仍保留
  该选择。物理 `Spec` 不混入训练 schema；filter identity 单独记录 view 子集，避免不同
  payload 输入错误复用缓存。
- sample/view manifest 的 schema、row count 和 row-group layout 由同一个
  `ParquetFile` 读取；sample index 的全量一致性校验按 store path 和文件 fingerprint
  缓存，后续打开不再重复扫描，同时仍会拒绝打开过程中发生变化的 manifest。
- store view manifest 先加载 `sample_index` 轻量列建立查找索引，具体 shard/key 行按
  row group 懒加载；随机读单个样本不需要把整个 view manifest 转成对象。
- store shuffle 按已选择 view 的 payload shard 组合分组，并在每个 group 内跨 rank 切分，
  不再把完整 group 只交给一个 rank。group 扫描结果按 sample/view manifest fingerprint
  缓存在 reader 内；manifest 变化会失效，pickle 后不携带进程内缓存。
- `sharded_csv` 保留 CSV 作为事实来源，prepare 阶段以 spawn process pool 并行生成
  每文件 Parquet part；manifest 原子提交并按源文件 size/mtime 增量复用。读取侧缓存
  Parquet row group，预计算每个文件的 row-group stop，并以 LRU 复用已打开的
  `ParquetFile`；动态 batch shuffle 直接按 row group 生成有界 index group，避免 rank 和
  DataLoader worker 重复解析全部 CSV，也不构造全数据集 Python index list。
- cost-aware planner 接受稳定 cost sequence、callable metadata cost 或单位 cost，并以
  bounded lookahead 流式生成 batch；候选删除和 batch cost 更新不再随全量 pending list
  或当前 batch size 重复放大。callable cost 可显式按全局 index 顺序一次物化，以顺序
  manifest 扫描替代 shuffle 后的随机 row-group 读取。DDP 默认在首个 batch 前完整生成
  rank-local plans，仅通过一次 tensor collective 同步 plan count，并裁剪 rank-local
  最终尾部；首个 batch 后不再进入 planner collective。
- part/fragment commit 不再常驻保存 `item ref -> sample_index array`；提交时先写
  ordered sample manifest，再按 view 流式扫描 sample manifest 做覆盖校验。
- 大量 part/fragment 的 manifest 使用固定 fan-in 的分层归并，打开的 parquet 文件数不再
  随输入数线性增长。最终发布 `.ready` 前还会核对声明的 sample 行数、view shard tar
  文件，以及 manifest 引用的全部 payload key。
- `BackgroundWriteSink` 支持 thread 和 process backend；materializer/filter 默认使用
  thread writer，保留 provider/filter 计算和落盘重叠，同时避免把大 write job 通过
  process pipe pickle 传输。
- 每个 materializer rank 的 writer 在 fragment 阶段结束后归并自己的 rank part；各
  rank 通过屏障确认 fragment 写入完成，再按稳定顺序分配包括续跑产物在内的 fragment。
  主进程最终只对 rank part 做 k-way merge、全局覆盖校验和原子发布。
- 新任务的 missing index 使用 `range`；续跑中 missing 较少时只物化 missing tuple，
  completed 较少时使用保存已完成下标的可 pickle lazy complement，避免按样本总数建立
  大型 Python tuple。
- `PayloadCache` 对已打开的 tar shard 做进程内 LRU 缓存，并优先读取每个 shard 的
  `*.tar.index.json` offset sidecar；sidecar 通过 tar fingerprint 和 header 校验后建立
  `payload key -> TarInfo` 映射，缺失、损坏或旧 store 自动回退 `getmembers()`。archive
  被淘汰或进程退出后随句柄释放；同一路径的 shard fingerprint 变化时再次访问会关闭旧句柄，
  避免 store 重建后复用旧 payload。
- manifest reader 使用按 path、fingerprint 和 pid 隔离的 `ParquetFile` LRU；fork 后不继承
  父进程句柄。`StoreDataset.close()` 和 context manager 会显式释放 manifest、payload archive
  与 file lease，析构函数只作为兜底。
- 最终 store 写入 `payload-groups.json`，按实际 shard 组合保存压缩的等差 sample-index
  分组；多 part 的 modulo 索引不会退化为每样本一条 JSON 记录。shuffle 优先读取该
  sidecar，缺失、损坏或 fingerprint 失效时回退逐样本扫描并生成同样的压缩分组，因此旧
  store 无需迁移即可读取；非 shuffle 读取始终保持全局 sample-index 顺序。
- `sharded_csv` 的 `iter_shard` 按 Parquet file/row group 顺序读取，只对当前 shard
  过滤全局 sample index；CSV cache fingerprint 额外包含 device、inode 和 ctime，旧记录
  缺少这些字段时自动失效重建。
- materializer 的 pending output 使用 deque，commit 阶段一次构建 `ref -> sample indexes`
  覆盖缓存，后台 writer 用完成队列降低 pending 扫描成本；collate 对 schema 计划预编译，
  变长 tensor 使用一次性 batch/mask 分配。
- integrity 校验提供 `fast`、`normal` 和 `full` 三档；`normal` 起会拒绝非法或重复的 tar
  payload member，默认的 `full` 还会读取全部 manifest 引用的 payload body，并拒绝 manifest
  外的额外 payload member。JSON 和目录原子替换在支持的文件系统上额外同步父目录，便于断电恢复场景。
- materializer resume metadata 除自动 factory 标识外，还接受显式 `input_id` 和
  `provider_id` 语义版本。它们共同决定 fragment 是否可复用，避免 mutable input 或模型
  checkpoint 变化后错误续跑。

## 公开默认值与生产调优

公开入口默认值只表达跨项目安全语义；吞吐相关参数由调用方根据任务、硬件和存储显式配置。
不要把 workspace 中某个数据集、provider 或 GPU 的 benchmark 结果反推成
`anydataset` 的库级默认值。

| 入口 | 当前默认值 | 默认值定位 | 生产调优入口 |
| --- | --- | --- | --- |
| `Runtime()` | reader/writer 的 `auto` 始终解析为 spawn；writer 默认 thread backend | 不让平台或 server 拓扑静默改变进程语义，避免继承设备状态 | 验证目标平台和设备边界后可显式覆盖 reader/writer start method；只有明确需要跨进程写入时才改 writer backend |
| `DatasetWriter` | `num_shards=1`、`num_workers=0`、`prefetch_factor=None` | 默认串行写，支持任意 iterable，避免默认要求 dataset factory 可 pickle | 大数据集显式传 `dataset_factory`，按数据源和存储调 `num_shards`、`num_workers`、`prefetch_factor` |
| `FilterRule.apply` | `device="auto"`、`batch_size=1`、`num_workers=0`、`prefetch_factor=None`、`commit_samples=100_000`、`write_workers=1` | 兼容只实现逐样本 `__call__` 的 predicate，并用后台 writer 重叠 partition cache 落盘 | predicate 支持 `call_batch` 时显式增大 `batch_size`；CPU decode 或特征读取重时调 `num_workers`/`prefetch_factor`；落盘慢时调 `write_workers`/`write_prefetch` |
| `ViewMaterializer` / `ModalityMaterializer` | `batch_size=1`、`num_workers=0`、`prefetch_factor=None`、`commit_samples=max(batch_size, 1024)`、`write_workers=1` | 默认单样本 provider 可运行，以有界 checkpoint 内存让 provider 执行与落盘重叠；最终按 `max_shard_samples` 流式 repack | GPU/provider 生产任务在 workflow/job wrapper 中显式调 `batch_size`、`num_workers`、`prefetch_factor`、`write_workers`、`write_prefetch`、`commit_samples`、`max_shard_samples` 和 `devices` |
| `AnyDataset.from_store(...)` | `views=None` 读取完整 store 语义 | 默认保留数据集语义，不猜训练只需要哪些 view | 训练、过滤或物化只需要部分 payload 时显式传 `views=...`，减少 manifest 和 payload 读取 |

如果某个 workflow 已经有稳定 benchmark，例如固定 A100、固定 provider、固定 store 后端，
应在对应 workspace/job wrapper 文档中记录推荐组合，并由 wrapper 显式传参。
库级文档只记录这些参数如何影响吞吐和可恢复性。

## 阶段一基准

阶段一入口是：

```bash
PYTHONPATH=src python scripts/benchmark_hot_paths.py
```

`scripts/benchmark_hot_paths.py` 覆盖九组热路径：

- `store_commit`: 多 part store 提交成本。
- `sharded_csv`: 物理 CSV 分片的 `iter_shard` 读取成本。
- `sharded_csv_lookup`: 同一 Parquet 文件跨多个 row group 随机读取，并报告打开句柄和
  row-group cache 数量。
- `store_reader`: lazy/preload manifest 的 store 打开成本。
- `store_shuffle`: 首次 payload group 扫描与 fingerprint cache 命中的读取计划成本。
- `store_payload_read`: `all_views` 和 `selected_view` 两种模式下逐样本执行 tar 定位、
  payload 读取和 UTF-8 解码的成本，并报告被跳过的未选择 view payload 数量。
- `sample_index_loader`: 当前 runtime iterable loader 和正式 map-style sample-index loader 实现。
- `filter_parallel`: 多 device filter 扫描、partition cache 写入和提交成本。
- `writer_pipeline`: inline、thread、spawn process 和 fork process 后台写入对比。

`sample_index_loader` 默认候选：

- `runtime`: 当前 `anydataset._runtime.parallel.runtime_sample_index_loader` 路径。
- `map_default`: map-style wrapper + global index sampler；当前等价于显式 spawn。
- `map_spawn`: map-style wrapper + global index sampler，DataLoader 显式使用 spawn。
- `map_fork`: map-style wrapper + global index sampler，DataLoader 显式使用 fork；仅在当前
  Python 支持 fork 时运行。

快速 smoke run：

```bash
PYTHONPATH=src python scripts/benchmark_hot_paths.py \
  --repeats 1 \
  --store-samples 32 \
  --store-payload-bytes 256 \
  --csv-rows-per-file 32 \
  --sample-index-samples 128 \
  --sample-index-num-workers 0
```

对 DataLoader worker 进程模型做对比时，把 `--sample-index-num-workers` 设为正数：

```bash
PYTHONPATH=src python scripts/benchmark_hot_paths.py \
  --repeats 3 \
  --sample-index-samples 20000 \
  --sample-index-batch-size 32 \
  --sample-index-num-workers 2 \
  --sample-index-variants runtime,map_default,map_spawn,map_fork
```

## 判断标准

- 每个候选必须输出相同的 selected sample count 和 index checksum。
- `store_payload_read` 的 `payload_reads` 必须等于样本数乘已选择 view 数；
  `selected_view` 还必须报告非零 `skipped_payload_reads`。该项单独衡量真实 payload
  读取，不能用只执行 `read_store_dataset()` 的 `store_reader` 打开时间替代。
- `sharded_csv_lookup` 必须覆盖至少两个 row group，同时保持一个打开的 Parquet 文件句柄。
- `map_spawn` 必须能通过 spawn worker 重建 dataset，证明 wrapper serialization 不携带
  已构造 dataset cache。
- 如果 `map_default` 或 `map_fork` 只在特定平台快，默认实现仍要保留显式可控的 start
  method，不能把平台差异藏进静默兼容逻辑。
- provider/filter 是否隔离到 server 不改变 reader/writer 的 `auto`：默认始终使用 spawn；
  经过目标平台验证后可显式选择 fork。
