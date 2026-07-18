# Performance Notes

本文记录 `anydataset` 当前性能优化的讨论范围、实验顺序和阶段一基准。这里先记录
局部 benchmark 结果；真实数据集和目标机器上的最终结论仍应沉淀到实验结果文档。

## 当前边界

- `ViewMaterializer` 和 `FilterRule` 都依赖稳定的全局 `sample_index`，用于分片写入、
  resume fragment、filter partition 和后续 store/delta 对齐。
- 外层 device/provider worker 继续保持 spawn-friendly。provider 可能加载 CUDA 模型，
  不应为了 DataLoader 读取性能把外层进程模型改成 fork。
- 外层扫描 worker、server 和 reader 的 start method 分开配置。
  `Runtime(reader_start_method="auto")` 在没有 server 时使用 spawn，在
  `server_start_method` 非空时使用 fork。后台 writer 默认使用 thread backend；只有显式
  使用 process writer backend 时才读取 `writer_start_method`。
- 默认用户数据集以 map-style 为主；streaming/iterable 数据集需要保留支持。当前
  `StoreDataset`、`FilteredDataset` 和 `MergedDataset` 这类默认 map-style shard 语义的
  materializer/filter 热路径会使用 map-style indexed loader；`AnyDataset` 仍优先保留
  source-aware indexed shard 路径，避免把顺序 source 退化成随机访问。`sharded_csv`
  prepare 后使用按源文件生成的 Parquet cache，因此也走 map-style indexed loader。
- store 格式保持稳定；reader 侧可以只读 parquet metadata 和轻量 index 列，按 row group
  懒加载 sample/view manifest 的完整行。`preload=True` 仍表示显式加载并校验所有 view
  manifest。

## 待验证事项

1. 在目标 Linux 机器上用真实 store/materializer 输入验证 server 模式的 fork
   reader/writer 行为；直接持有 torch/CUDA/provider 状态的本地路径继续使用 spawn。
2. 基于 map-style indexed loader 已提供的稳定全局 sample index，验证 distributed LBA
   tail flush 能否从 object gather 改成 metadata-only flush。

## 已落地

- `anydataset._parallel.map_style_indexed_loader` 使用 rank sampler 分发全局 sample index，
  并由 `MapIndexedDataset` 返回 `(sample_index, sample)`。
- wrapper 可以在当前进程复用已构造 dataset；spawn 序列化时丢弃该缓存，让 worker 通过
  `dataset_factory` 懒加载重建。
- `ViewMaterializer` 和多设备 `FilterRule` 对默认 map-style shard 语义的数据集使用该
  loader；有自定义 `iter_indexed_shard` 的数据集继续走 runtime indexed loader。
- server 模式下 reader/writer worker 默认用 fork；无 server 的本地路径保持 spawn，避免
  本地 torch/CUDA/provider 状态被 worker 继承。
- `StoreDataset` 打开时不再把 `samples.parquet` 全量转成 Python tuple；`samples` 保留
  sequence 接口，并按 parquet row group 懒加载完整 sample manifest 行。
- store view manifest 先加载 `sample_index` 轻量列建立查找索引，具体 shard/key 行按
  row group 懒加载；随机读单个样本不需要把整个 view manifest 转成对象。
- `sharded_csv` 保留 CSV 作为事实来源，prepare 阶段以 spawn process pool 并行生成
  每文件 Parquet part；manifest 原子提交并按源文件 size/mtime 增量复用。读取侧缓存
  Parquet row group，避免 rank 和 DataLoader worker 重复解析全部 CSV。
- part/fragment commit 不再常驻保存 `item ref -> sample_index array`；提交时先写
  ordered sample manifest，再按 view 流式扫描 sample manifest 做覆盖校验。
- `BackgroundWriteSink` 支持 thread 和 process backend；materializer/filter 默认使用
  thread writer，保留 provider/filter 计算和落盘重叠，同时避免把大 write job 通过
  process pipe pickle 传输。
- 每个 materializer rank 的 writer 在 fragment 阶段结束后归并自己的 rank part；各
  rank 通过屏障确认 fragment 写入完成，再按稳定顺序分配包括续跑产物在内的 fragment。
  主进程最终只对 rank part 做 k-way merge、全局覆盖校验和原子发布。
- 新任务的 missing index 使用 `range`；续跑中 missing 较少时只物化 missing tuple，
  completed 较少时使用保存已完成下标的可 pickle lazy complement，避免按样本总数建立
  大型 Python tuple。
- `PayloadCache` 对已打开的 tar shard 做进程内 LRU 缓存，并在每个打开的 archive 上缓存
  `payload key -> TarInfo` 映射；连续随机读取不再反复打开 tar 或线性扫描 member。该索引
  不持久化，archive 被淘汰或进程退出后随句柄释放。
- materializer resume metadata 除自动 factory 标识外，还接受显式 `input_id` 和
  `provider_id` 语义版本。它们共同决定 fragment 是否可复用，避免 mutable input 或模型
  checkpoint 变化后错误续跑。

## 阶段一基准

阶段一入口是：

```bash
PYTHONPATH=src python scripts/benchmark_hot_paths.py
```

`scripts/benchmark_hot_paths.py` 覆盖七组热路径：

- `store_commit`: 多 part store 提交成本。
- `sharded_csv`: 物理 CSV 分片的 indexed shard 读取成本。
- `store_reader`: lazy/preload manifest 的 store 打开成本。
- `store_payload_read`: 打开 store 后逐样本执行 tar 定位、payload 读取和 UTF-8 解码的成本。
- `indexed_loader`: 当前 runtime iterable loader 和正式 map-style indexed loader 实现。
- `filter_parallel`: 多 device filter 扫描、partition cache 写入和提交成本。
- `writer_pipeline`: inline、thread、spawn process 和 fork process 后台写入对比。

`indexed_loader` 默认候选：

- `runtime`: 当前 `anydataset._parallel.indexed_loader` 路径。
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
  --indexed-samples 128 \
  --indexed-num-workers 0
```

对 DataLoader worker 进程模型做对比时，把 `--indexed-num-workers` 设为正数：

```bash
PYTHONPATH=src python scripts/benchmark_hot_paths.py \
  --repeats 3 \
  --indexed-samples 20000 \
  --indexed-batch-size 32 \
  --indexed-num-workers 2 \
  --indexed-variants runtime,map_default,map_spawn,map_fork
```

## 判断标准

- 每个候选必须输出相同的 selected sample count 和 index checksum。
- `store_payload_read` 必须输出与样本数相同的 `payload_reads`；该项单独衡量真实 payload
  读取，不能用只执行 `read_store_dataset()` 的 `store_reader` 打开时间替代。
- `map_spawn` 必须能通过 spawn worker 重建 dataset，证明 wrapper serialization 不携带
  已构造 dataset cache。
- 如果 `map_default` 或 `map_fork` 只在特定平台快，默认实现仍要保留显式可控的 start
  method，不能把平台差异藏进静默兼容逻辑。
- remote fork 默认只适用于 provider/filter 已经隔离到 server 的 reader/writer worker 路径；
  local provider 路径继续使用 spawn。

## 暂不处理

- 不在阶段一接入 LBA 或修改 distributed tail flush。
- 不在阶段一重写 filter partition cache 生命周期；lazy index loading 继续和 cache
  snapshot 设计一起讨论。
