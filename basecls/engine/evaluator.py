#!/usr/bin/env python3
# Copyright (c) 2014-2021 Megvii Inc. All rights reserved.
from typing import Iterable, Sequence, Tuple

import megengine as mge
import megengine.functional as F
import numpy as np
from basecore.engine import BaseEvaluator
from loguru import logger

__all__ = ["AccEvaluator"]


class AccEvaluator(BaseEvaluator):
    """Classification evaluator with top-1 and top-5 accuracy."""

    ResultType = Tuple[int, float, float]

    def preprocess(self, input_data: Sequence[np.ndarray]) -> mge.Tensor:
        """Preprocess input data per batch.

        Args:
            input_data: input data.

        Returns:
            Preprocessed input data.
        """
        return mge.Tensor(input_data[0])

    def postprocess(
        self, model_outputs: mge.Tensor, input_data: Sequence[np.ndarray]
    ) -> ResultType:
        """Postprocess model outputs with input data per batch.

        Args:
            model_outputs: model outputs.
            input_data: input data.

        Returns:
            A tuple that (batch size, top-1 accuracy per batch, top-5 accuracy per batch).
        """
        targets = mge.Tensor(input_data[1])
        accs = F.metric.topk_accuracy(model_outputs, targets, (1, 5))
        cnt = targets.shape[0]
        acc1 = accs[0].item() * 100 * targets.shape[0]
        acc5 = accs[1].item() * 100 * targets.shape[0]
        return cnt, acc1, acc5

    def evaluate(self, results: Iterable[ResultType]):
        """Evaluation function.

        Args:
            results: all results.
        """
        cnt, acc1, acc5 = map(sum, zip(*results))
        logger.info(f"Test Acc@1: {acc1 / cnt:.3f}, Acc@5: {acc5 / cnt:.3f}")
