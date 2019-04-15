import os as os
from os import listdir
from os.path import isfile, join
import json
import torch

AMOUNT_OF_DATA = 3353


lossL2Multiplicator = 1
strictFuckUpLossMultiplicator = 10
areaLossMultiplicator = 0
tvLossMultiplicator = 0


EPOCHS = 200
SAMPLES = 100
ALABATCH = 32

#DEVICE_ID = 0
#DEVICE = torch.device('cuda:%d' % DEVICE_ID)
#torch.cuda.set_device(DEVICE_ID)

DEVICE = "cpu"

DTYPE = torch.float32


OUTPUT_H = 11
OUTPUT_W = 18

