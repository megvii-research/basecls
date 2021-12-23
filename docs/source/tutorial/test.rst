.. _test:

====
测试
====

本节例子中假设你已经处于你的实验目录中，你的模型配置文件名为 ``resnet50.py`` 。

单机测试
--------

测试脚本默认在模型配置文件的 ``output_dir`` 目录中寻找最后一个 checkpoint ，并测试在 ImageNet 1k validation set 上的 Top-1 Accuracy 和 Top-5 Accuracy 。

.. code-block:: shell

   cls_test -f resnet50.py

如果你需要测试指定的 checkpoint 或模型权重，请加 ``-w`` 参数，既可以是本地路径，也可以是 OSS 路径。
