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

`Task` 只适合非常稳定、跨数据集一致的默认 schema，例如图像分类、基础 audio codec 或机器翻译文本对。组合型、研究型或仍在快速变化的任务不应急着进入 `Task` 枚举，用户显式写 schema 更清楚。

## 派生 View

派生表示应通过 provider 和 `ViewMaterializer` 生成。典型流程是：

```text
base store -> provider -> delta store -> AnyDataset(store).merge(delta) -> schema selects derived view
```

例如 LongCat codes 是 `AudioView.LONGCAT`。Preset 不负责加载 codec，也不应该把 LongCat 逻辑塞进 raw row parse。Preset 只需要产出可被 provider 消费的音频 view，例如 `AudioView.WAVEFORM` 或 `AudioView.FILE`。

## 派生 Modality

同一 role 下缺失的模态应通过 provider 和 `ModalityMaterializer` 生成。provider 只声明输出 view，materializer 用输出 view 推出输出 modality，并在同一 role 中寻找唯一的非输出 modality 作为输入。

如果输出 modality 已经存在，materializer 必须报错；这条路径只负责补缺失模态，不负责覆盖或刷新已有数据。如果同一 role 去掉输出 modality 后还剩多个输入 modality，materializer 也必须报错，调用方应先用 schema 或 transform 明确输入。

`ModalityMaterializer` 生成的新 item 默认不复制 meta。label、language 等跨模态语义继承必须由调用方显式完成，避免库替用户猜测业务规则。

## 过滤分区

过滤规则直接作用在 dataset 产出的完整 canonical `Sample` 上。predicate 返回 bool、字符串、枚举值或带 metrics 的 `FilterDecision`，库统一归一化为字符串 label，并缓存每个 label 对应的原始样本下标。

`FilterRule` 的缓存契约只包含 `name`。predicate、parse function 和 transforms 的语义版本不由库检查，调用方应把这些约定写进 `name`。这样 filter 只负责可验证的数据结构和缓存机制，不把用户业务规则伪装成库能自动理解的东西。
