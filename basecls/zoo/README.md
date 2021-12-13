# 评测所有模型

假设所有模型所在目录为 `/path/to/models` ，目录里每个子目录下面有同名的模型配置文件和模型参数文件，如：

```
/path/to/models
    |-> resnet
    |    |resnet18.py
    |    |resnet18.pkl
    |    |resnet50.py
    |    |resnet50.pkl
    |-> regnet
    |    |regnetx_002.py
    |    |regnetx_002.pkl
```

可执行如下命令测试

```shell
python3 -m basecls.zoo.testing_all -d /path/to/models
```

最终结果会将 Top-1 Accuracy 和 Top-5 Accuracy 输出到 `result.json` 文件中
