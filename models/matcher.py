import torch
from scipy.optimize import linear_sum_assignment
from torch import nn
import torch.nn.functional as F
import numpy as np
from einops import rearrange
import random


class HungarianMatcher(nn.Module):
    def __init__(
        self,
        cost: float = 1,
        cost_power: float = 1,
        norm: float = 1,
        normalize: bool = False,
    ):
        """Creates the matcher
        Params:
            cost_class: This is the relative weight of the classification error in the matching cost
            cost_giou: This is the relative weight of the giou loss of the bounding box in the matching cost
        """
        super().__init__()
        self.cost = cost
        self.cost_power = cost_power
        self.normalize = normalize


    def forward(
        self,
        predictions,
        targets
    ):
        """ Performs the matching
        Params:
            predictions: Tensor of dim [batch_size, num_predicted_counts] 
            targets: Tensor of dim [batch_size, num_gt_counts] 
        Returns:
            A list of size batch_size, containing tuples of (index_i, index_j) where:
                - index_i is the indices of the selected predictions (in order)
                - index_j is the indices of the corresponding selected targets (in order)
        """

        if len(predictions.shape) == 1 or len(predictions.shape) == 2:
            predictions = predictions.unsqueeze(0)
        if len(targets.shape) == 1 or len(targets.shape) == 2:
            targets = targets.unsqueeze(0)

        num_gt_counts = targets.shape[1]

        inds = []

        for y, gt in             zip(predictions, targets)        :


            gt_nonz = torch.nonzero(torch.sum(gt, dim=1)).squeeze()

            if not torch.equal(gt_nonz,torch.tensor(range(gt_nonz.numel())).cuda().squeeze()):
                print(f"Error: the nonzeros should be first {gt_nonz=} {torch.tensor(range(gt_nonz.numel())).cuda().squeeze()}")
            gt = gt[gt_nonz, :]

            if len(gt.shape) == 1:
                gt = gt.unsqueeze(0)
            if len(y.shape) == 1:
                y = y.unsqueeze(0)

            if self.normalize:
                y = F.normalize(y.clone(), dim=1)
                gt = F.normalize(gt.clone(), dim=1)

            cost = torch.cdist(y, gt, p=self.cost)
            cost = cost ** self.cost_power

            indices = linear_sum_assignment(cost.detach().cpu())
            y_inds = indices[0]
            gt_inds = indices[1]

            used_gt_inds = set(list(gt_inds))
            possible_gt_inds = set(range(num_gt_counts))
            not_used_inds = list(possible_gt_inds - used_gt_inds)
            gt_inds = list(gt_inds)
            gt_inds.extend(not_used_inds)

            inds.append(
                (
                    torch.as_tensor(y_inds, dtype=torch.int64),
                    torch.as_tensor(gt_inds, dtype=torch.int64),
                )
            )

        return inds
