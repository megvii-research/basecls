.. _train:

====
训练
====

本节例子中假设你已经处于你的实验目录中，你的模型配置文件名为 ``resnet50.py`` 。

单机训练
--------

目前单机训练会占用全部 GPU ：

.. code-block:: shell

   cls_train -f resnet50.py

若中途训练中断，可追加 ``--resume`` 参数，BaseCls 会从上一次保存的 checkpoint 开始继续训练。

训练结束后，模型将自动使用最后一个 checkpoint ，测试其在 ImageNet 1k validation set 上的 Top-1 Accuracy 和 Top-5 Accuracy。

可以在终端中追加参数覆盖模型配置文件中的字段，例如你需要在显存更大的机器上训练模型：

.. code-block:: shell

   cls_train -f resnet50.py batch_size 128 solver.basic_lr 0.1

模型配置文件中的 ``batch_size`` 和 ``solver.basic_lr`` 字段将被覆盖。

.. warning::

   并不推荐通过传参修改配置项，因为这可能会影响“模型配置文件—测试结果”的可复现性。直接修改模型配置文件是更推荐的做法。
