# TODO

## Dataset iteration

- [ ] 为 unified store 增加 epoch-aware 的物理分片规划：按 epoch/seed 稳定分配或打乱 tar shard，让多 worker / 多卡训练减少重复 IO，并为 resume 和审计保留可复现的 shard 计划。
