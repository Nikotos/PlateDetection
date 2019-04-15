from Dataset import *
from model import *
from DopeTech import *
from Criterion import *
from config import *
import torch


model = TroubleShooter()
#model = loadFromFile("model1.pkl")
model = model.to(device=DEVICE)

dataset = loadFromFile("Dataset.pkl")
dataset.shuffle()

loaderTrain = DataLoader(dataset, 0, 3000)
loaderVal = DataLoader(dataset, 3101, 3352)


def checkAccuracy(model):
    index = len(loaderVal)
    model.eval()
    model = model.to(device=DEVICE)
    print(index)
    with torch.no_grad():
        loss = 0.0
        for i in range(index):
            x,h,p = loaderVal.getSample()
            x = x.unsqueeze(0).unsqueeze(0)
            res = model(x).squeeze(0).squeeze(0)
            lossSample = totalLoss(res,h,p)
            loss += lossSample
        return loss / index


optimizer = torch.optim.Adam(model.parameters(), lr=0.01, betas=(0.9, 0.999))
for e in range(300):
    model.train()
    avgLoss = 0.0
    for s in range(SAMPLES):
        loss = 0.0
        # instead of passing minibatches through model
        for s in range(ALABATCH):
            x,h,p = loaderTrain.getSample()
            x = x.unsqueeze(0).unsqueeze(0)
            mapp = model(x).squeeze(0).squeeze(0)
            loss += totalLoss(mapp,h,p)
        

        loss = loss / ALABATCH
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        avgLoss += loss
        print("TrainLoss -", loss)
        print("ValLoss -", checkAccuracy(model))
    
    avgLoss /= SAMPLES
    print('epoch %d, Validation loss = %.4f' % (e, checkAccuracy(model)))
    print('avgLoss = %.4f' % (avgLoss))
    print()
    if e % 10 == 0:
        torch.save(model.state_dict(), "modelAfter")

