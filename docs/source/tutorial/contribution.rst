.. _contribution:

========
贡献指南
========

开发环境
--------

.. code-block:: shell

   # 安装依赖
   pip3 install -r requirements-dev.txt --user

   # 配置 pre-commit
   pre-commit install

开发流程
--------

提交者需补充相应修改的单元测试。

.. code-block:: shell

   # （外部开发者）fork repo，或（内部开发者）建立 new-feature 分支
   git checkout -b new-feature

   # 进行修改

   # 代码风格检查与格式化
   make lint
   make format

   # 单元测试与覆盖率检查
   make unittest

   # 提交修改
   git commit

   # 提交 MR/PR
