# Python 兼容性

`anydataset` 的运行下限是 Python 3.9。源码应避免使用会阻塞 3.9 解析或运行的
语法和标准库特性。

## 约定

- 不使用 PEP 695 语法，包括 `type Alias = ...`、`def f[T](...)` 和
  `class C[T]`。
- 不使用 `match` / `case`。
- 需要 `StrEnum`、`Self`、`NotRequired` 或严格长度 `zip` 时，通过
  `anydataset._compat` 引入。
- 类型别名如果需要运行时求值，使用 `typing.Union[...]`，不要依赖
  Python 3.10 的 `A | B` 运行时类型合并。

## 验证

```bash
PYTHONPYCACHEPREFIX=/private/tmp/anydataset-pycache python3 -m compileall -q src tests
PYTHONPYCACHEPREFIX=/private/tmp/anydataset-pycache PYTHONPATH=src python -m pytest -q
```
