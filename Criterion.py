import torch.nn as nn
from config import *


def lossL2(x,heatmap):
    loss = nn.MSELoss()
    return loss(x, heatmap)

def strictFuckUpLoss(x, plate):
    # Dot product to check if we miss the car plate
    if torch.equal(x * plate, torch.zeros(x.shape).to(device=DEVICE, dtype=DTYPE)):
        loss = nn.MSELoss()
        return loss(x, plate)
    else:
        return 0

def areaLoss(x, heatmap):
    a = torch.abs(heatmap - 1)
    return torch.sum(x * a)


def tvLoss(x):
    return torch.sum(torch.abs(x[:, :-1] - x[:, 1:])) + torch.sum(torch.abs(x[:-1, :] - x[1:, :]))

def totalLoss(x, heatmap, plate):
    return lossL2Multiplicator * lossL2(x,heatmap) + strictFuckUpLossMultiplicator * strictFuckUpLoss(x,plate) + areaLossMultiplicator * areaLoss(x, heatmap) + tvLossMultiplicator * tvLoss(x)

