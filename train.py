from Dataset import *
from model import *
from DopeTech import *
from Criterion import *
from config import *
import torch


model = TroubleShooter()
model = model.to(device=DEVICE)

dataset = loadFromFile("Dataset.pkl")

loaderTrain = DataLoader(dataset, 0, 3100)
loaderVal = DataLoader(dataset, 3101, 3352)


def checkAccuracy(model, loader):
    index = len(loader)
    model.eval()
    with torch.no_grad():
        loss = []
        for i in range(index):
            x,h,p = loader.getSample()
            x = x.view(3,42,70).unsqueeze(0)
            x = model(x).squeeze(0).squeeze(0)
            loss.append(totalLoss(x,h,p))
    return loss



for e in range(EPOCHS):
    optimizer = torch.optim.SGD(model.parameters(), lr=0.0001, momentum=0.90)
    model.train()
    for s in range(SAMPLES):
        loss = 0.0
        # instead of passing minibatches through model
        for s in range(ALABATCH):
            x,h,p = loaderTrain.getSample()
            x = x.view(3,42,70).unsqueeze(0)
            map = model(x).squeeze(0).squeeze(0)
            loss += totalLoss(map,h,p)
          
        loss = loss / ALABATCH
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        print('epoch %d, loss = %.4f' % (e, loss))
        #checkAccuracy(model, loaderVal)
        print()
