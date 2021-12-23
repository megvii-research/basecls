#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright (c) 2014-2021 Megvii Inc. All rights reserved.
import re

import setuptools

with open("README.md", "r") as f:
    long_description = f.read()


with open("basecls/__init__.py", "r") as f:
    version = re.search(r'^__version__\s*=\s*[\'"]([^\'"]*)[\'"]', f.read(), re.MULTILINE).group(1)

with open("requirements.txt", "r") as f:
    reqs = [x.strip() for x in f.readlines()]


setuptools.setup(
    name="basecls",
    version=version,
    author="basedet team",
    author_email="base-detection@megvii.com",
    description="A codebase & model zoo for pretrained backbone based on MegEngine.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://basecls.readthedocs.io/zh_CN/latest/index.html",
    packages=setuptools.find_packages(),
    classifiers=["Programming Language :: Python :: 3", "Operating System :: OS Independent"],
    python_requires=">=3.6",
    install_requires=reqs,
    extras_require={"all": ["megengine>=1.6.0"]},
    entry_points={
        "console_scripts": [
            "cls_train=basecls.tools.cls_train:main",
            "cls_test=basecls.tools.cls_test:main",
        ]
    },
)
