
import numpy as np

import torch

from torch import nn


class DiceBCELoss(nn.Module):

    def __init__(self):

        super().__init__()


    def dice_jaccard_score(self, y_pred, y_truth):

        assert y_pred.size(0) == y_truth.size(0)

        smoothing = 1e-9

        
        # batch_size = y_pred.size(0)
        
        numerator = torch.sum(y_pred.view(-1) * y_truth.view(-1))

        denominator = torch.sum(y_pred.view(-1)) + torch.sum(y_truth.view(-1))

        jaccard = ( numerator / ( denominator - numerator + smoothing ) ) #/ batch_size

        dice = 2 * ( numerator / ( denominator + smoothing) ) #/ batch_size

        return dice, jaccard

        

    def dice_jaccard_loss(self, y_pred, y_truth):

        assert y_pred.size(0) == y_truth.size(0)

        dice, jaccard = self.dice_jaccard_score(y_pred, y_truth)


        jaccard_loss = 1 - jaccard

        dice_loss = 1 - dice

        return dice_loss, jaccard_loss


    def forward(self, y_pred, y_truth):

        bce_criterion = nn.BCELoss()
        loss_bce = bce_criterion(y_pred, y_truth)

        dice_loss, jaccard_loss = self.dice_jaccard_loss(y_pred , y_truth)

        return loss_bce + dice_loss + jaccard_loss



        



