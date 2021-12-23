.. _config:

============
配置实验环境
============

实验目录
--------

用户可以任意新建一个目录作为实验目录，BaseCls 鼓励的实验范式是 library 与 playground 分离，每个人（组）自行维护一个 playground 目录。

模型配置文件
------------

一个模型通常对应一个 ``.py`` 模型配置文件，一个标准的模型配置文件如下所示：

.. code-block:: python
   :linenos:

   from basecls.configs import ResNetConfig

   _cfg = dict(
       batch_size=64,
       model=dict(
           name="resnet50",
       ),
       solver=dict(
           basic_lr=0.05,
       ),
   )


   class ResNet50Config(ResNetConfig):
       def __init__(self, values_or_file=None, **kwargs):
           super().__init__(_cfg)
           self.merge(values_or_file, **kwargs)


   Cfg = ResNet50Config

BaseCls 要求模型配置文件必须包含一个名为 ``Cfg`` 的类，该类必须为 :py:class:`~basecore.config.ConfigDict` 的子类。用户可以继承已有的模型配置文件并修改。

BaseCls 基于注册机制，支持接入用户自定义的网络。详情见 :ref:`model` 。

数据源
------

* 倘若数据集在本地，请使用 :py:class:`~basecls.data.FolderLoader` 并填写 ``data.train_path`` 和 ``data.val_path`` 字段。
* 倘若只需要随机数据（如性能评测等），请使用 :py:class:`~basecls.data.FakeData` 。

BaseCls 支持接入第三方数据源。详情见 :ref:`dataloader` 。

数据增强
--------

BaseCls 标准数据处理流程为 :py:class:`~megengine.data.transform.RandomResizedCrop` -> :py:class:`~megengine.data.transform.RandomHorizontalFlip` -> ``[augments]`` -> :py:class:`~basecls.data.rand_erase.RandomErasing` -> :py:class:`~basecls.data.mixup.MixupCutmixCollator` 。其中 ``[augments]`` 为可修改的数据增强部分，默认为 :py:class:`~basecls.data.ColorAugment` 。

BaseCls 支持自定义数据增强。详情见 :ref:`augments` 。
