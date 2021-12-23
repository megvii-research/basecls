#!/usr/bin/env python3
# Copyright (c) 2014-2021 Megvii Inc. All rights reserved.
"""
Create a single page for all models, and append toctree to all index pages
"""
import pathlib
import re
import sys

from basecore.utils.import_utils import import_content_from_path
from loguru import logger

from basecls.zoo.utils import get_series_and_name_and_id

model_rst = r"""
.. _zoo-{series}-{name}:

{overline}
{name}
{overline}

Model info
----------

.. raw:: html

   <div id="modelInfoContainer" data-uid="{uid}"></div>


Python source
-------------

.. literalinclude:: {name}.py
   :language: python

"""

index_md = r"""
# {series}

"""

index_toctree = r"""
```{toctree}
:maxdepth: 1
:glob:

*
```
"""


def main():
    series = {}
    # restrict two-level folder structure
    for scope in ("public", "internal"):
        for f in pathlib.Path(f"source/zoo/{scope}").glob("**/*.py"):
            logger.info(f"Found {f}")
            sys.path.insert(0, str(f.absolute().parent))  # extend path to correctly import cfg
            try:
                import_content_from_path("Cfg", f)
            except AttributeError:
                logger.warning(f"{f} not a config file, skip")
                continue
            meta = get_series_and_name_and_id(f)
            meta["origin_path"] = str(f.absolute().resolve())
            logger.debug(meta)
            if meta["series"] not in series:
                series[meta["series"]] = meta["origin_path"]
            # Generate individual pages for each model
            # if f.with_suffix(".rst").exists():
            #     logger.warning("rst file exists, skip it")
            #     continue
            with open(f.with_suffix(".rst"), "w") as rstf:
                rstf.write(model_rst.format(**meta, overline="=" * len(meta["name"])))
                logger.info(f"Create {rstf.name}")
            sys.path.pop(0)

    # generate index pages if not exists
    # also build toctree if required
    for series_name, origin_path in series.items():
        logger.info(f"Process {series_name} located at {origin_path}")
        origin_path = pathlib.Path(origin_path).parent
        has_index = False
        for possible_index in ["index.md", "readme.md", "README.md"]:
            if (origin_path / possible_index).exists():
                has_index = True
                break
        if has_index:
            logger.warning(f"{origin_path / possible_index} exists")
            content = open(origin_path / possible_index, "r").read()
            if re.findall(r"\{toctree\}", content):
                logger.warning("already has toctre")
            else:
                with open(origin_path / possible_index, "a") as indexf:
                    indexf.write(index_toctree)
                    logger.debug(f"append toctre to {indexf.name}")
        else:
            with open(origin_path / "index.md", "w") as indexf:
                indexf.write(index_md.format(series=series_name))
                indexf.write(index_toctree)
                logger.info(f"Create {indexf.name}")


if __name__ == "__main__":
    main()
