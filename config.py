import cv2
import os as os
from os import listdir
from os.path import isfile, join
import json
import torch

AMOUNT_OF_DATA = 3353


lossL2Multiplicator = 1
strictFuckUpLossMultiplicator = 10


EPOCHS = 100
SAMPLES = 1000
ALABATCH = 32

DEVICE = 'cpu'


OUTPUT_H = 11
OUTPUT_W = 18
