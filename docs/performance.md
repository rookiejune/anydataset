# Performance Notes

本文记录 `anydataset` 当前性能优化的讨论范围、实验顺序和阶段一基准。这里不放
已经验证的最终结论；结论需要等 benchmark 在目标机器和真实数据集上跑完以后再沉淀。

## 当前边界

- `ViewMaterializer` 和 `FilterRule` 都依赖稳定的全局 `sample_index`，用于分片写入、
  resume fragment、filter partition 和后续 store/delta 对齐。
- 外层 device/provider worker 继续保持 spawn-friendly。provider 可能加载 CUDA 模型，
  不应为了 DataLoader 读取性能把外层进程模型改成 fork。
- 内层 PyTorch DataLoader worker 主要负责读样本。它的 start method 可以和外层
  provider worker 分开讨论。
- 默认用户数据集以 map-style 为主；streaming/iterable 数据集需要保留支持，但不应让
  iterable runtime sharding 成为所有 map-style 热路径的唯一实现。

## 待验证假设

1. 对 map-style dataset，DataLoader 使用 sampler 分发全局 index，再由 wrapper 返回
   `(sample_index, sample)`，可能比当前 `RuntimeIndexedDataset(IterableDataset)` 路径
   更快。
2. 对内层 DataLoader worker，PyTorch 默认 context 或 Linux fork 可能比固定 spawn 更快；
   但 spawn 仍需要作为可验证的安全路径存在。
3. map-style indexed wrapper 如果缓存 dataset 实例，必须在 spawn serialization 时丢弃
   缓存，让 worker 通过 `dataset_factory` 懒加载重建。
4. 只有 indexed loader 决策稳定后，才适合继续讨论 LBA tail flush 是否能从 object gather
   改成 metadata-only flush。

## 阶段一基准

阶段一只增加实验入口，不修改 `src/` 默认实现。入口是：

```bash
PYTHONPATH=src python scripts/benchmark_hot_paths.py
```

`scripts/benchmark_hot_paths.py` 覆盖三组热路径：

- `store_commit`: 多 part store 提交成本。
- `sharded_csv`: 物理 CSV 分片的 indexed shard 读取成本。
- `indexed_loader`: 当前 runtime iterable loader 和 map-style indexed wrapper 候选实现。

`indexed_loader` 默认候选：

- `runtime`: 当前 `anydataset._parallel.indexed_loader` 路径。
- `map_default`: map-style wrapper + global index sampler，DataLoader 使用 PyTorch 默认
  multiprocessing context。
- `map_spawn`: map-style wrapper + global index sampler，DataLoader 显式使用 spawn。
- `map_fork`: map-style wrapper + global index sampler，DataLoader 显式使用 fork；仅在当前
  Python 支持 fork 时运行。

快速 smoke run：

```bash
PYTHONPATH=src python scripts/benchmark_hot_paths.py \
  --repeats 1 \
  --store-samples 32 \
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
- `map_spawn` 必须能通过 spawn worker 重建 dataset，证明 wrapper serialization 不携带
  已构造 dataset cache。
- 如果 `map_default` 或 `map_fork` 只在特定平台快，默认实现仍要保留显式可控的 start
  method，不能把平台差异藏进静默兼容逻辑。
- 只有当 map-style wrapper 在 store-like 和真实 materializer 输入上都稳定更优，才考虑
  把它设为 `ViewMaterializer` 默认 indexed path。

## 暂不处理

- 不在阶段一改 `anydataset._parallel.indexed_loader` 的公开行为。
- 不在阶段一接入 LBA 或修改 distributed tail flush。
- 不在阶段一重写 filter partition cache 生命周期；lazy index loading 继续和 cache
  snapshot 设计一起讨论。
