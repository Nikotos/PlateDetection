from Dataset import *
from config import *
from DopeTech import *
import json
import torch as torch
import torch.nn as nn


HEIGHT = 42
WIDTH = 70


originalPathData = "./dataset/data/car"
originalPathMarkup = "./dataset/markup/carMarkup"

sizeW = []
sizeH = []

def fillRectangleWithOnes(tensor, coords):
    x0,y0 = coords[0]
    x1,y1 = coords[1]
    x2,y2 = coords[2]
    x3,y3 = coords[3]
    
    xAddition = 1
    yAddition = 1
    x0,y0 = round(mulX(x0)) - xAddition, round(mulY(y0)) - 2 * yAddition
    x1,y1 = round(mulX(x1)) + xAddition, round(mulY(y1)) - 2 * yAddition
    x2,y2 = round(mulX(x2)) + xAddition, round(mulY(y2)) + yAddition
    x3,y3 = round(mulX(x3)) - xAddition, round(mulY(y3)) + yAddition

    for i in range(0, OUTPUT_H):
        for j in range(0, OUTPUT_W):
            if ((i >= y0) and (i <= y2)) and ((j >= x0) and (j <= x2)):
                tensor[i][j] = 1
            if ((i >= y1) and (i <= y3)) and ((j >= x3) and (j <= x1)):
                tensor[i][j] = 1
            if ((i >= y0) and (i <= y3)) and ((j >= x0) and (j <= x1)):
                tensor[i][j] = 1
            if ((i >= y1) and (i <= y2)) and ((j >= x3) and (j <= x2)):
                tensor[i][j] = 1



def fillPlateWithClearOnes(tensor, coords):
    x0,y0 = coords[0]
    x1,y1 = coords[1]
    x2,y2 = coords[2]
    x3,y3 = coords[3]

    xAddition = 0.2
    yAddition = 0.1
    x0,y0 = round(mulX(x0 - xAddition)), round(mulY(y0 - yAddition))
    x1,y1 = round(mulX(x1 + xAddition)), round(mulY(y1 - yAddition))
    x2,y2 = round(mulX(x2 + xAddition)), round(mulY(y2 + yAddition))
    x3,y3 = round(mulX(x3 - xAddition)), round(mulY(y3 + yAddition))
    
    for i in range(0, OUTPUT_H):
        for j in range(0, OUTPUT_W):
            if ((i >= y0) and (i <= y2)) and ((j >= x0) and (j <= x2)):
                tensor[i][j] = 1
            if ((i >= y1) and (i <= y3)) and ((j >= x3) and (j <= x1)):
                tensor[i][j] = 1
            if ((i >= y0) and (i <= y3)) and ((j >= x0) and (j <= x1)):
                tensor[i][j] = 1
            if ((i >= y1) and (i <= y2)) and ((j >= x3) and (j <= x2)):
                tensor[i][j] = 1




def mulX(x):
    return x * (18 / 70)

def mulY(y):
    return y * (11 / 42)

#   RECTANGLE
#
#   0 # # # # # # # 1
#   #               #
#   #               #
#   3 # # # # # # # 2
#
#



gaussian_weights = torch.tensor([[0.109634,    0.111842,    0.109634],
                                 [0.111842,    0.114094,    0.111842],
                                 [0.109634,    0.111842,    0.109634]]).unsqueeze(0).unsqueeze(0)


def heatmapFromMarkup(markup):
    tensor = torch.zeros([OUTPUT_H, OUTPUT_W])
    
    coordinatesArray = markup["plates"]
    for struct in coordinatesArray:
        fillRectangleWithOnes(tensor, struct["frame"])
    
    gaussianBlur = nn.Conv2d(1, 1,  stride=1,
                         kernel_size=3, padding=1, bias=False)
    with torch.no_grad():
        gaussianBlur.weight = nn.Parameter(gaussian_weights)

    return gaussianBlur(tensor.unsqueeze(0).unsqueeze(0)).squeeze(0).squeeze(0)



def plateMapFromMarkup(markup):
    tensor = torch.zeros([OUTPUT_H, OUTPUT_W])
    
    coordinatesArray = markup["plates"]
    for struct in coordinatesArray:
        fillPlateWithClearOnes(tensor, struct["frame"])
    
    return tensor


def prepareImage(image):
    image = torch.tensor(image).type(torch.FloatTensor)
    r, g, p = image[:,:,0], image[:,:,1], image[:,:,2]
    gray = 0.2989 * r + 0.5870 * g + 0.1140 * p
    return gray




dataset = Dataset()

for index in range(1, AMOUNT_OF_DATA  + 1):
    openDataName = originalPathData + str(index) + ".png"
    openMarkupName = originalPathMarkup + str(index) + ".json"
    
    img = cv2.imread(openDataName)
    
    with open(openMarkupName, 'r') as f:
        markup = json.load(f)
    

    dataset.add(prepareImage(img), heatmapFromMarkup(markup), plateMapFromMarkup(markup))
    print("done -", index)


dataset.shuffle()
saveToFile(dataset, "Dataset.pkl")
