.. _model:

==========
自定义网络
==========

BaseCls 支持接入用户自定义的网络。

实现范式
--------

* 网络必须继承自 :py:class:`~megengine.module.Module` 。
* 自定义参数通过模型配置文件 ``model`` 字段传入。
* 以下字段为保留字段不可使用：

  * ``model.name`` ， BaseCls 用此字段构造网络。
  * ``model.head.name`` ， BaseCls 用此字段构造分类头。

具体步骤
--------

实现网络并注册
~~~~~~~~~~~~~~

.. code-block:: python
   :linenos:
   :emphasize-lines: 30, 33, 50

   from typing import Any, Mapping

   import megengine as mge
   import megengine.functional as F
   import megengine.module as M
   from basecls.layers import build_head
   from basecls.utils import registers

   class AlexNetHead(M.Module):

       def __init__(self, w_in: int, w_out: int, width: int = 4096):
           super().__init__()
           self.avg_pool = M.AdaptiveAvgPool2d((6, 6))
           self.classifier = M.Sequential(
               M.Dropout(),
               M.Linear(w_in * 6 * 6, width),
               M.ReLU(),
               M.Dropout(),
               M.Linear(width, width),
               M.ReLU(),
               M.Linear(width, w_out),
           )

       def forward(self, x: mge.Tensor) -> mge.Tensor:
           x = self.avg_pool(x)
           x = F.flatten(x, 1)
           x = self.classifier(x)
           return x

   @registers.models.register()
   class AlexNet(M.Module):

       def __init__(self, head: Mapping[str, Any] = None):
           super().__init__()
           self.features = M.Sequential(
               M.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
               M.ReLU(),
               M.MaxPool2d(kernel_size=3, stride=2),
               M.Conv2d(64, 192, kernel_size=5, padding=2),
               M.ReLU(),
               M.MaxPool2d(kernel_size=3, stride=2),
               M.Conv2d(192, 384, kernel_size=3, padding=1),
               M.ReLU(),
               M.Conv2d(384, 256, kernel_size=3, padding=1),
               M.ReLU(),
               M.Conv2d(256, 256, kernel_size=3, padding=1),
               M.ReLU(),
               M.MaxPool2d(kernel_size=3, stride=2),
           )
           self.head = build_head(256, head)

       def forward(self, x: mge.Tensor) -> mge.Tensor:
           x = self.features(x)
           if getattr(self, "head", None) is not None:
               x = self.head(x)
           return x

修改模型配置文件
~~~~~~~~~~~~~~~~

.. code-block:: python
   :linenos:
   :emphasize-lines: 3, 5, 7-10

   _cfg = dict(
       ...
       num_classes=1000,
       model=dict(
           name="AlexNet",
           ...  # 你想传入的自定义参数
           head=dict(
               name=AlexNetHead,  # 也可以直接传入一个类
               # w_out=1000,  # 若该字段未定义，会自动传入cfg.num_classes
           ),
       )
       ...
   )
