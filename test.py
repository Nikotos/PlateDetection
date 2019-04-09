import cv2
import os as os
from os import listdir
from os.path import isfile, join
import json
import pickle

originalPathData = "./dataset/data/train"
originalPathMarkup = "./dataset/markup/train"

outputPathData = "./set48/data/train"
outputPathMarkup = "./set48/markup/train"

MAX_HEIGHT = 48
MAX_WIDTH = 48
dataX = []
dataY = []

def saveToFile(object, filename):
    with open(filename, "wb") as file:
        pickle.dump(object, file)

def loadFromFile(filename):
    with open(filename, "rb") as file:
        return pickle.load(file)



for index in range(1, 68):
    filesData = [f for f in listdir(originalPathData + str(index))]
    filesMarkup = [f for f in listdir(originalPathMarkup + str(index))]
    container = dict(zip(filesData, filesMarkup))
    
    for e, (fileName, markupName) in enumerate(container.items()):
        openMarkupName = originalPathMarkup + str(index) + "/" + markupName
        with open(openMarkupName, 'r') as f:
            markup = json.load(f)
        
        coordinatesArray = markup["plates"]
        for struct in coordinatesArray:
            for coords in struct["frame"]:
                dataX.append(coords[0])
                dataY.append(coords[1])
        print("done", e)


saveToFile(dataX, "dataX.pkl")
saveToFile(dataY, "dataY.pkl")

