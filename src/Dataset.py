import os

def makeDict(categories):
    '''

    :param categories: list of possible categories in a categorical variable
    :return: returns a dictionary which maps a string into a unique integer
    '''
    d={}
    index=0
    for cat in categories:
        d[cat]=index
        index+=1
    return  d

class Dataset():
    def __init__(self):
        self.root="/mnt/Backup/Dataset/Extract/notMNIST_small"
        self.labels=os.listdir(self.root)
        self.labelsDict=makeDict(self.labels)
