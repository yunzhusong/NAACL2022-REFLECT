""" Hierachical Extractor. """
import math
from typing import Optional
import torch
from torch import Tensor
from torch.nn import CrossEntropyLoss
import pdb

class SoftCrossEntropyLoss(CrossEntropyLoss):

    def __init__(self,
                 weight: Optional[Tensor] = None,
                 size_average=None,
                 ignore_index: int = -100):
        CrossEntropyLoss.__init__(self, weight, size_average, ignore_index)

    def forward(self, inputs, targets, scores=None):
        loss_mask = (targets != self.ignore_index).int()
        logprobs = torch.nn.functional.log_softmax(inputs, dim=-1)
        logprobs = torch.gather(logprobs, 1,
                                (targets * loss_mask).unsqueeze(-1)).squeeze(-1)

        if scores is not None:
            # Weight for non-selected setence
            non_select_mask = torch.logical_and(
                targets != 1, targets != self.ignore_index).int()
            kpi = (1 - scores)**10 * non_select_mask  # inverse proportional
            kpi += (1 - kpi.max()) * non_select_mask  # shift
            non_select_weights = kpi

            # Weight for selected setence
            select_weights = (targets == 1).int()

            loss_weights = select_weights + non_select_weights
            loss = -(logprobs * loss_mask *
                     loss_weights).sum() / loss_mask.sum()
        else:
            loss = -(logprobs * loss_mask).sum() / loss_mask.sum()

        return loss
