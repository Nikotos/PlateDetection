import torch.nn as nn
from config import *


def lossL2(x,heatmap):
    loss = nn.MSELoss()
    return loss(x, heatmap)

def strictFuckUpLoss(x, plate):
    # Dot product to check if we miss the car plate
    if torch.equal(x * plate, torch.zeros(x.shape)):
        loss = nn.MSELoss()
        return loss(x ,plate)
    else:
        return 0

def totalLoss(x, heatmap, plate):
    return lossL2Multiplicator * lossL2(x,heatmap) + strictFuckUpLossMultiplicator * strictFuckUpLoss(x,plate)
