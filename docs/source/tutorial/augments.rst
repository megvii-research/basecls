.. _augments:

==============
自定义数据增强
==============

BaseCls 支持自定义数据增强。

实现范式
--------

* 数据增强类必须实现 ``build`` 类方法，返回一个 :py:class:`~megengine.data.transform.VisionTransform` 对象。
* 自定义参数通过模型配置文件 ``augments`` 字段传入。
* 以下字段为保留字段不可使用：

  * ``augments.name`` ，BaseCls 用此字段构造数据增强类。

具体步骤
--------

实现网络并注册
~~~~~~~~~~~~~~

.. code-block:: python
   :linenos:
   :emphasize-lines: 6

   import megengine.data.transform as T
   import numpy as np
   from basecls.utils import registers
   from basecore.config import ConfigDict

   @registers.augments.register()
   class YourAugmentBuilder:

        @classmethod
        def build(cls, cfg: ConfigDict) -> T.Transform:
            return YourAugment(cfg)

   class YourAugment(T.VisionTransform):

        def __init__(self, cfg: ConfigDict):
            pass

        def _apply_image(self, image: np.ndarray) -> np.ndarray:
            pass

修改模型配置文件
~~~~~~~~~~~~~~~~

.. code-block:: python
   :linenos:
   :emphasize-lines: 4

   _cfg = dict(
       ...
       argments=dict(
           name="YourAugmentBuilder",
           ...  # 你想传入的自定义参数
       )
       ...
   )
