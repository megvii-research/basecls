.. _dataloader:

============
自定义数据源
============

BaseCls 支持接入第三方数据源。

实现范式
--------

* 数据源类必须实现 ``build`` 类方法，返回一个 :py:class:`~collections.abc.Iterable` 对象（实现了 ``__iter__`` 方法）。
* 自定义参数通过模型配置文件 ``data`` 字段传入。
* 以下字段为保留字段不可使用：

  * ``data.name`` ，BaseCls 用此字段构造数据源。

具体步骤
--------

实现数据源并注册
~~~~~~~~~~~~~~~~

.. code-block:: python
   :linenos:
   :emphasize-lines: 4

   from basecls.utils import registers
   from basecore.config import ConfigDict

   @registers.dataloaders.register()
   class YourDataSourceBuilder:

       @classmethod
       def build(cls, cfg: ConfigDict, augments):
           return YourDataSource(cfg, augments)

   class YourDataSource:

       def __init__(self, cfg: ConfigDict, augments):
           pass

       def __iter__(self):
           pass

修改模型配置文件
~~~~~~~~~~~~~~~~

.. code-block:: python
   :linenos:
   :emphasize-lines: 4

   _cfg = dict(
       ...
       data=dict(
           name="YourDataSourceBuilder",
           ...  # 你想传入的自定义参数
       )
       ...
   )
