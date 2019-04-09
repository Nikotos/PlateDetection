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
    
    x0,y0 = round(x0) - 6, round(y0) - 3
    x1,y1 = round(x1) + 6, round(y1) - 3
    x2,y2 = round(x2) + 6, round(y2) + 1
    x3,y3 = round(x3) - 6, round(y3) + 1

    for i in range(0, HEIGHT):
        for j in range(0, WIDTH):
            if ((i >= y0) and (i <= y2)) and ((j >= x0) and (j <= x2)):
                tensor[i][j] = 1
            if ((i >= y1) and (i <= y3)) and ((j >= x3) and (j <= x1)):
                tensor[i][j] = 1
            if ((i >= y0) and (i <= y3)) and ((j >= x0) and (j <= x1)):
                tensor[i][j] = 1
            if ((i >= y1) and (i <= y2)) and ((j >= x3) and (j <= x2)):
                tensor[i][j] = 1



def fillRectangleWithClearOnes(tensor, coords):
    x0,y0 = coords[0]
    x1,y1 = coords[1]
    x2,y2 = coords[2]
    x3,y3 = coords[3]

    x0,y0 = round(x0 - 0.3), round(y0 - 0.3)
    x1,y1 = round(x1 + 0.3), round(y1 - 0.3)
    x2,y2 = round(x2 + 0.3), round(y2 + 0.3)
    x3,y3 = round(x3 - 0.3), round(y3 + 0.3)
    
    for i in range(0, HEIGHT):
        for j in range(0, WIDTH):
            if ((i >= y0) and (i <= y2)) and ((j >= x0) and (j <= x2)):
                tensor[i][j] = 1
            if ((i >= y1) and (i <= y3)) and ((j >= x3) and (j <= x1)):
                tensor[i][j] = 1
            if ((i >= y0) and (i <= y3)) and ((j >= x0) and (j <= x1)):
                tensor[i][j] = 1
            if ((i >= y1) and (i <= y2)) and ((j >= x3) and (j <= x2)):
                tensor[i][j] = 1


#   RECTANGLE
#
#   0 # # # # # # # 1
#   #               #
#   #               #
#   3 # # # # # # # 2
#
#




#[[0.109634,    0.111842,    0.109634],
#[0.111842,    0.114094,    0.111842],
#[0.109634,    0.111842,    0.109634]]



#[[0.111096,    0.111119,    0.111096],
#[0.111119,    0.111141,    0.111119],
#[0.111096,    0.111119,    0.111096]]


#[[0.039968,    0.039992,    0.04,    0.039992,    0.039968],
#[0.039992,    0.040016,    0.040024,    0.040016,    0.039992],
#[0.04,    0.040024,    0.040032,    0.040024,    0.04],
#[0.039992,    0.040016,    0.040024,    0.040016,    0.039992],
#[0.039968,    0.039992,    0.04,    0.039992,    0.039968]]


#[[0.011237,    0.011637,    0.011931,    0.012111,    0.012172,    0.012111,    0.011931,    0.011637,    0.011237],
#[0.011637,    0.012051,    0.012356,    0.012542,    0.012605,    0.012542,    0.012356,    0.012051,    0.011637],
#[0.011931,    0.012356,    0.012668,    0.01286,    0.012924,    0.01286,    0.012668,    0.012356,    0.011931],
#[0.012111,    0.012542,    0.01286,    0.013054,    0.013119,    0.013054,    0.01286,    0.012542,    0.012111],
#[0.012172,   0.012605,    0.012924,    0.013119,    0.013185,    0.013119,    0.012924,    0.012605,    0.012172],
#[0.012111,    0.012542,    0.01286,    0.013054,    0.013119,    0.013054,    0.01286,    0.012542,    0.012111],
#[0.011931,    0.012356,    0.012668,    0.01286,    0.012924,    0.01286,    0.012668,    0.012356,    0.011931],
#[0.011637,    0.012051,    0.012356,    0.012542,   0.012605,    0.012542,    0.012356,    0.012051,    0.011637],
#[0.011237,    0.011637,    0.011931,    0.012111,    0.012172,    0.012111,    0.011931,    0.011637,    0.011237]]


gaussian_weights = torch.tensor([[0.011237,    0.011637,    0.011931,    0.012111,    0.012172,    0.012111,    0.011931,    0.011637,    0.011237],
                                 [0.011637,    0.012051,    0.012356,    0.012542,    0.012605,    0.012542,    0.012356,    0.012051,    0.011637],
                                 [0.011931,    0.012356,    0.012668,    0.01286,    0.012924,    0.01286,    0.012668,    0.012356,    0.011931],
                                 [0.012111,    0.012542,    0.01286,    0.013054,    0.013119,    0.013054,    0.01286,    0.012542,    0.012111],
                                 [0.012172,   0.012605,    0.012924,    0.013119,    0.013185,    0.013119,    0.012924,    0.012605,    0.012172],
                                 [0.012111,    0.012542,    0.01286,    0.013054,    0.013119,    0.013054,    0.01286,    0.012542,    0.012111],
                                 [0.011931,    0.012356,    0.012668,    0.01286,    0.012924,    0.01286,    0.012668,    0.012356,    0.011931],
                                 [0.011637,    0.012051,    0.012356,    0.012542,   0.012605,    0.012542,    0.012356,    0.012051,    0.011637],
                                 [0.011237,    0.011637,    0.011931,    0.012111,    0.012172,    0.012111,    0.011931,    0.011637,    0.011237]]).unsqueeze(0).unsqueeze(0)


def heatmapFromMarkup(markup):
    tensor = torch.zeros([HEIGHT, WIDTH])
    
    coordinatesArray = markup["plates"]
    for struct in coordinatesArray:
        fillRectangleWithOnes(tensor, struct["frame"])
    
    gaussianBlur = nn.Conv2d(1, 1,  stride=1,
                         kernel_size=9, padding=4, bias=False)
    with torch.no_grad():
        gaussianBlur.weight = nn.Parameter(gaussian_weights)

    return gaussianBlur(tensor.unsqueeze(0).unsqueeze(0)).squeeze(0).squeeze(0)



def plateMapFromMarkup(markup):
    tensor = torch.zeros([HEIGHT, WIDTH])
    
    coordinatesArray = markup["plates"]
    for struct in coordinatesArray:
        fillRectangleWithClearOnes(tensor, struct["frame"])
    
    return tensor

	

dataset = Dataset()

for index in range(1, AMOUNT_OF_DATA  + 1):
    openDataName = originalPathData + str(index) + ".png"
    openMarkupName = originalPathMarkup + str(index) + ".json"
    
    img = cv2.imread(openDataName)
    
    with open(openMarkupName, 'r') as f:
        markup = json.load(f)
    

    dataset.add(torch.tensor(img).type(torch.FloatTensor), heatmapFromMarkup(markup), plateMapFromMarkup(markup))
    print("done -", index)


dataset.shuffle()
saveToFile(dataset, "Dataset.pkl")
