import pickle


def saveToFile(object, filename):
    with open(filename, "wb") as file:
        pickle.dump(object, file)

def loadFromFile(filename):
    with open(filename, "rb") as file:
        return pickle.load(file)
