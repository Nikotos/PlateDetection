import torch
import random
from random import shuffle
from config import *

class Dataset:
    def __init__(self):
        self.data = []
        self.heatmaps = []
        self.plates = []
    
    
    def getOne(self, index):
        if (index < len(self.data)):
            x = self.data[index].to(device=DEVICE, dtype=DTYPE)
            h = self.heatmaps[index].to(device=DEVICE, dtype=DTYPE)
            p = self.plates[index].to(device=DEVICE, dtype=DTYPE)
            return x,h,p
        else:
            return None

    def add(self, x, heatmap, plate):
        self.data.append(x)
        self.heatmaps.append(heatmap)
        self.plates.append(plate)
    
    
    def __len__(self):
        return len(self.data)
    
    
    def shuffle(self):
        indicies = [i for i in range(0, len(self.data))]
        shuffle(indicies)
        
        dataNew = []
        heatmapsNew = []
        platesNew = []
        
        for i in indicies:
            dataNew.append(self.data[i])
            heatmapsNew.append(self.heatmaps[i])
            platesNew.append(self.plates[i])
        
        self.data = dataNew
        self.heatmaps = heatmapsNew
        self.plates = platesNew





class DataLoader:
    def __init__(self, dataset, indexMin, indexMax):
        assert(indexMax > indexMin)
        self.dataset = dataset
        self.indexMin = indexMin
        self.IndexMax = indexMax
    
    
    def getSample(self):
        index = random.randint(self.indexMin, self.IndexMax)
        return self.dataset.getOne(index)
    
    def __len__(self):
        return self.IndexMax - self.indexMin

