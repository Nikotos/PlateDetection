import cv2
import os as os
from os import listdir
from os.path import isfile, join
import json
import pickle
import torch
import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image
import numpy as np
import torch.nn as nn
import random


from DopeTech import *
from Dataset import *
from model import *

import torchvision.transforms.functional as F
import torchvision.transforms as transforms


MAX_HEIGHT = 42
MAX_WIDTH = 70
def prepare(img):
    height, width = img.shape[:2]
    
    scalingFactor = MAX_HEIGHT / float(height)
    img = cv2.resize(img, None, fx=scalingFactor, fy=scalingFactor, interpolation=cv2.INTER_AREA)
    
    _, newWidth = img.shape[:2]
    padding = MAX_WIDTH - newWidth
    if padding > 0:
        if padding % 2 == 0:
            img = cv2.copyMakeBorder(img, 0, 0, padding // 2, padding // 2, cv2.BORDER_CONSTANT)
        else:
            img = cv2.copyMakeBorder(img, 0, 0, padding // 2, (padding // 2) + 1, cv2.BORDER_CONSTANT)
    img = torch.tensor(img).type(torch.FloatTensor)
    r, g, p = img[:,:,0], img[:,:,1], img[:,:,2]
    img = 0.2989 * r + 0.5870 * g + 0.1140 * p
    return img



def findPlates(image):
    img = prepare(image)
    img = img.unsqueeze(0).unsqueeze(0)
    mapp = model(img).squeeze(0).squeeze(0)
    mapup = nn.functional.interpolate(mapp.unsqueeze(0).unsqueeze(0), size = image.shape[:2], mode = 'bilinear')
    image = torch.tensor(image).type(torch.FloatTensor)
    
    r, g, p = image[:,:,0], image[:,:,1], image[:,:,2]
    slic = 0.2989 * r + 0.5870 * g + 0.1140 * p
    image[:,:,0] = slic + mapup.squeeze(0).squeeze(0) * 400
    image[:,:,1] = slic
    image[:,:,2] = slic
    return image.detach().numpy().astype(np.int8)

# for use with downsampled dataset
def findPlatesTensored(image):
    img = image.unsqueeze(0).unsqueeze(0)
    mapup = nn.functional.interpolate(model(img), size = image.shape, mode = 'bilinear')
    return (image + mapup.squeeze(0).squeeze(0) * 400).detach().numpy().astype(np.int)


d = loadFromFile("Dataset.pkl")
loaderTrain = DataLoader(d, 0, 3000)
loaderVal = DataLoader(d, 3101, 3352)
model = TroubleShooter()
model.load_state_dict(torch.load("zoo/modelNext", map_location='cpu'))  #modelAfter, modelNext is good enough, TSNew might be better (50/50)
model.eval()


cap = cv2.VideoCapture('try.mov')

frame_width = int(cap.get(3))
frame_height = int(cap.get(4))
out = cv2.VideoWriter('out.avi',cv2.VideoWriter_fourcc('M','J','P','G'), 10, (frame_width,frame_height))

if (cap.isOpened()== False):
    print("Error opening video stream or file")

while(cap.isOpened()):
    ret, frame = cap.read()
    if ret == True:
        frame = findPlates(frame)
        print(frame.shape)
        out.write(frame)
    else:
        break
cap.release()
out.release()

cv2.destroyAllWindows()
