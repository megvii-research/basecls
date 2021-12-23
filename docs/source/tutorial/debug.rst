.. _debug:

====
调试
====

本节例子中假设你已经处于你的实验目录中，你的模型配置文件名为 ``resnet50.py`` 。

在终端中调试
------------

如果只希望构造出模型在终端中调试，或在其他地方引用，BaseCls 提供了简单的工具函数，帮助用户直接从模型配置文件中构造网络：

.. code-block:: python
   :linenos:

   # 创建配置
   from resnet50 import Cfg
   cfg = Cfg()

   # 创建模型
   from basecls.models import build_model
   model = build_model(cfg)

   # 如果不需要分类头，可以直接删除
   del model.head

   # 载入权重
   from basecls.models import load_model
   if getattr(cfg, "weights", None) is not None:
       load_model(model, cfg.weights, strict=False)

   # 开始调试
   from megengine.utils.module_stats import module_stats
   module_stats(model, input_shapes=(1, 3, 224, 224))

BaseCls 同样支持在 :py:mod:`~megengine.hub` 中调用官方已有的模型并载入初始化权重（初次载入需要联网）：

.. code-block:: python
   :linenos:

   # 创建模型
   import megengine.hub as hub
   model = hub.load(
       "megvii-research/basecls:main",
       "resnet50",  # 模型名称
       use_cache=False,
       pretrained=True,  # 载入初始化权重
   )

   # 开始调试
   from megengine.utils.module_stats import module_stats
   module_stats(model, input_shapes=(1, 3, 224, 224))
