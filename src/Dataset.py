import os
import numpy as np
from collections import deque
import math
import random


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

def readFromPath(path):
    '''

    :param path: a string denoting the path of the image
    :return: a numpy array for the image
    '''
    data=np.fromfile(path)
    return  data

def makeQueue(items):
    '''

    :param items: list of some filepaths
    :return:  queue with filepaths
    '''
    Q=deque(items)
    return Q

def splitData(items,ratio=0.8):
    '''

    :param items: list of filePAths
    :param ratio: ratio of train set to total data set
    :return: returns filepaths in two split array first for
    '''
    index=math.floor(ratio*len(items))
    return items[:index],items[:index:]

class Dataset():
    def __init__(self):
        self.root="/mnt/Backup/Dataset/Extract/notMNIST_small"
        self.labels=os.listdir(self.root)
        self.labelsDict=makeDict(self.labels)
        self.Data=self.makeData()
        self.trainData,self.testData=splitData(self.Data)

    def makeData(self,shuf=True):
        '''

        :return: list of tuples (path,label)
            path is that of the file and label is a category for the coreesonding image represented by path
        '''

        data=[]

        for cat in os.listdir(self.root): # these are same as labels
            dirpath=os.path.join(self.root,cat)
            for file in os.listdir(dirpath):
                filepath=os.path.join(dirpath,file)
                data.append((filepath,cat))

        if(shuf):
            random.shuffle(data)
        return data
