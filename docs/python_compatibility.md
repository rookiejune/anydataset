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

最低版本验证必须用同一个明确的 Python 3.9 interpreter 完成编译和测试，避免
`python3` 与 `python` 指向不同环境。`PYTHON` 按本机环境替换：

```bash
PYTHON=/path/to/python3.9
PYTHONPYCACHEPREFIX=/tmp/anydataset-pycache "$PYTHON" -m compileall -q src tests examples
PYTHONPYCACHEPREFIX=/tmp/anydataset-pycache PYTHONPATH=src "$PYTHON" -m pytest -q
```

`pyproject.toml` 的 `requires-python >= 3.9` 是安装下限；classifiers 当前列出
Python 3.9-3.12。发布前至少验证 3.9，并在声称某个版本已验证时使用对应 interpreter
重复以上两条命令。
