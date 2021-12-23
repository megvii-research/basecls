.. _advanced:

========
进阶内容
========

本节例子中假设你已经处于你的实验目录中，你的模型配置文件名为 ``resnet50.py`` 。

DTR
---

BaseCls 支持利用 `DTR`_ 技术进行训练。相关知识内容请参考文档。

通过修改模型配置文件中的 ``dtr`` 相关字段开启：

.. code-block:: python

   cfg.dtr = True

也可以在终端中追加参数覆盖模型配置文件中的字段：

.. code-block:: shell

   cls_train -f resnet50.py dtr True

AMP
---

BaseCls 支持利用 `AMP`_ 自动混合精度进行训练，可以提升训练吞吐。相关知识内容请参考文档。

通过修改模型配置文件中的 ``amp`` 相关字段开启：

.. code-block:: python

   cfg.amp.enabled = True

也可以在终端中追加参数覆盖模型配置文件中的字段：

.. code-block:: shell

   cls_train -f resnet50.py amp.enabled True

混合精度训练中使用 fp16 代替 fp32 进行梯度计算，因此会涉及浮点数表达能力的问题。为了防止小量级的梯度在 fp16 表示中损失精度，实践中往往将损失函数扩大适当的倍率，称为 ``loss_scale`` ，在 :py:class:`~megengine.amp.GradScaler` 中实现了相关逻辑。

混合精度的默认行为是固定 ``loss_scale = 128.0`` ，另外一种动态调整 ``loss_scale`` 的方法可以更好地利用fp16的动态范围，可以使用 ``dynamic_scale`` 字段开启。开启时初始 ``loss_scale = 65536.0`` ，并每 ``2000`` 轮迭代进行一次翻倍，当遇到梯度为 ``inf`` 时会立即将其减半：

.. code-block:: python

   cfg.amp.dynamic_scale = True


Model EMA
---------

Model EMA 是一种加快模型收敛的方法，常用于较新的卷积模型和 Transformer 类模型。通过统计模型参数的滑动平均，能够获得更好的泛化能力。

通过修改模型配置文件中的 ``model_ema`` 相关字段开启，默认每步迭代都会更新 Model EMA：

.. code-block:: python

   cfg.model_ema.enabled = True
   cfg.model_ema.momentum = 0.9999
   cfg.model_ema.update_period = 1

也可以设置 ``update_period`` 隔一定步数进行更新，一个建议是当提升更新间隔时，要相应减小 ``momentum`` 的值：

.. code-block:: python

   cfg.model_ema.enabled = True
   cfg.model_ema.momentum = 0.9992
   cfg.model_ema.update_period = 8

另外如果对于设置正确的 ``momentum`` 没有经验，也可以设置 ``alpha`` 的默认配置，``momentum`` 会根据更新间隔和更新量 ``alpha`` 自动适配：

.. code-block:: python

   cfg.model_ema.enabled = True
   cfg.model_ema.alpha = 1e-5
   cfg.model_ema.update_period = 32

相关更新公式为：

.. math::

   \mathrm{momentum} = 1 - \alpha \times \frac{\mathrm{update\_period} \times \mathrm{total\_batch\_size}}{\mathrm{max\_epochs}}

.. _DTR: https://megengine.org.cn/doc/stable/zh/user-guide/model-development/dtr/index.html
.. _AMP: https://docs.nvidia.com/deeplearning/performance/mixed-precision-training/index.html
