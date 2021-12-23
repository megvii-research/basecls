.. _install:

============
安装 BaseCls
============

安装环境
--------

BaseCls |version| 需要 Python >= 3.6。

BaseCls |version| 依赖 MegEngine >= 1.6.0。

通过包管理器安装
----------------

通过 ``pip`` 包管理器安装 BaseCls 的命令如下：

.. code-block:: shell

   pip3 install basecls --user

默认不会安装包括 MegEngine 在内的部分依赖，可以通过以下命令进行完整安装：

.. code-block:: shell

   pip3 install basecls[all] --user

.. note::

   对于 ``conda`` 用户, 可以选择通过在环境中先安装 ``pip`` ，再按照上述方式进行 BaseCls 的安装。

通过源代码安装
--------------

为保证模型性能的可追溯性，避免实验碎片化，建议通过包管理器安装。如果包管理器安装的方式无法满足你的需求，则可以尝试自行通过源码安装。

请先将 `源码 <https://github.com/megvii-research/basecls>`_ clone 到本地。

安装依赖
~~~~~~~~

.. code-block:: shell

   pip3 install -r requirements.txt --user

安装 BaseCls
~~~~~~~~~~~~

.. code-block:: shell

   python3 setup.py develop --user

验证安装
--------

在 Python 中导入 BaseCls 验证安装成功并查看安装版本：

.. code-block:: python

   import basecls
   print(basecls.__version__)
