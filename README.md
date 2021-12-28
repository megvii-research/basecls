# BaseCls

[![Documentation Status](https://readthedocs.org/projects/basecls/badge/?version=latest)](https://basecls.readthedocs.io/zh_CN/latest/?badge=latest) [![CI](https://github.com/megvii-research/basecls/actions/workflows/ci.yml/badge.svg)](https://github.com/megvii-research/basecls/actions/workflows/ci.yml) [![codecov](https://codecov.io/gh/megvii-research/basecls/branch/main/graph/badge.svg?token=EOB6AISNJ0)](https://codecov.io/gh/megvii-research/basecls)

BaseCls 是一个基于 [MegEngine](https://megengine.org.cn/) 的预训练模型库，帮助大家挑选或训练出更适合自己科研或者业务的模型结构。

文档地址：<https://basecls.readthedocs.io>

## 安装

### 安装环境

BaseCls 需要 Python >= 3.6。

BaseCls 依赖 MegEngine >= 1.6.0。

### 通过包管理器安装

通过 `pip` 包管理器安装 BaseCls 的命令如下：

```shell
pip3 install basecls --user
```

默认不会安装包括 MegEngine 在内的部分依赖，可以通过以下命令进行完整安装：

```shell
pip3 install basecls[all] --user
```

对于 `conda` 用户, 可以选择通过在环境中先安装 `pip`，再按照上述方式进行 BaseCls 的安装。

### 通过源代码安装

为保证模型性能的可追溯性，避免实验碎片化，建议通过包管理器安装。如果包管理器安装的方式无法满足你的需求，则可以尝试自行通过源码安装。

#### 安装依赖

```shell
pip3 install -r requirements.txt --user
```

#### 安装 BaseCls

```shell
python3 setup.py develop --user
```

### 验证安装

在 Python 中导入 BaseCls 验证安装成功并查看安装版本：

```python
import basecls
print(basecls.__version__)
```

## 开发者须知

### 开发环境

```shell
# 安装依赖
pip3 install -r requirements-dev.txt --user

# 配置 pre-commit
pre-commit install
```

### 开发流程

提交者需补充相应修改的单元测试。

```shell
# （外部开发者）fork repo，或（内部开发者）建立 new-feature 分支
git checkout -b new-feature

# 进行修改

# 代码风格检查与格式化
make lint
make format

# 单元测试与覆盖率检查
make unittest

# 提交修改
git commit

# 提交MR/PR
```
