from math import pi

import numpy as np
import torch

from .epoch_runner import EpochRunner


class RelRotRunner(EpochRunner):
    def __init__(self, configs):
        super().__init__(configs)

    def _metrics(self, outputs_model, inputs_data, mode="train") -> dict:
        with torch.no_grad():
            R_pred: torch.Tensor = outputs_model["rotation"]
            R_gt: torch.Tensor = inputs_data["rotation"]
            rel_R = torch.bmm(R_pred, R_gt.transpose(-1, -2))
            angles = torch.stack(
                [
                    torch.acos(torch.clamp((torch.trace(R) - 1) / 2, -1, 1)).abs()
                    for R in rel_R
                ]
            )
            angles = angles / pi * 180
            metrics = {
                "mean": torch.mean(angles),
                "acc_3": torch.mean((angles <= 3).float()),
                "acc_5": torch.mean((angles <= 5).float()),
                "acc_10": torch.mean((angles <= 10).float()),
            }
        return metrics

