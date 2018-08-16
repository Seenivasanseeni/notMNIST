import os
import numpy as np
from collections import deque
import math
import random
import matplotlib.pyplot as plt

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
    data=plt.imread(path)
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
        self.trainDataQueue=makeQueue(self.trainData)
        self.testDataQueue=makeQueue(self.testData)
        self.batchSize=100

    def getOnehotLabel(self,label):
        '''

        :param label: label which is one of the categories
        :return: a one hot vector
        '''
        onehot=[0]*len(self.labels)
        onehot[self.labelsDict[label]]=1
        return onehot

    def getLabel(self,label):
        '''
        :param label: category in string
        :return: a numerical value from 0 to len(self.labels)-1 . It uses self.labelsDict for this mapping
        '''
        return self.labelsDict[label]

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
                data.append((filepath,self.getOnehotLabel(cat)))

        if(shuf):
            random.shuffle(data)
        return data

    def getBatch(self,test=False):
        '''

        :return: returns array of images and  labels
        '''
        images=[]
        labels=[]
        dataQueue=self.trainDataQueue
        if(test):
            dataQueue=self.testDataQueue

        for _ in range(self.batchSize):
            path,label=dataQueue.popleft()
            try:
                image=readFromPath(path)
                images.append(image)
                labels.append(label)
                dataQueue.append((path, label))
            except:
                print("Error Occured while reading a file")
        return images,labels
