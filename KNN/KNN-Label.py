#coding=utf_8
##VILLAIN
##2016.7.24
from numpy import *
import numpy as np
import operator
import time
from os import listdir



def createDataSet(filename):
    # group = array([[1.0,1.1],[1.0,1.0],[0,0],[0,0.1]])
    # labels = ['A','A','B','B']
    f = open(filename)
    arrayLines = f.readlines()
    print arrayLines
    # 为了方便特征数据的归一化，我们将特征数据和标签分开存放
    dataArray = zeros([len(arrayLines), 2])  # 特征数据的数据矩阵
    index = 0
    classList = []  # 标签的存放

    # 解析每行字符串
    for line in arrayLines:
        line = line.strip()
        datalist = line.split('\t')
        dataArray[index, :] = datalist[0:2]
        classList.append(datalist[-1])
        index += 1

    return dataArray, classList
    # return group,labels


def kNNClassify(inX, dataSet, labels, k):
    dataSetSize = dataSet.shape[0]
    diffMat = tile(inX, (dataSetSize,1)) - dataSet
    sqDiffMat = diffMat**2
    sqDistances = sqDiffMat.sum(axis=1)
    distances = sqDistances**0.5
    sortedDistIndicies = distances.argsort()
    classCount={}
    for i in range(k):
        voteIlabel = labels[sortedDistIndicies[i]]
        classCount[voteIlabel] = classCount.get(voteIlabel,0) + 1
    sortedClassCount = sorted(classCount.iteritems(), key=operator.itemgetter(1), reverse=True)
    return sortedClassCount[0][0]

def test():
    dataSet, labels = createDataSet('C:/Code/PycharmProjects/K-NN/KNN-master/KNN-master/211.txt')
    k = 3
    testX = array([0.2, 0.])
    # print testX


    ff=[]
    g = input("input测试元组：")
    print g

    outputLabel = kNNClassify(g, dataSet, labels, k)
    print ("Your input is:", g, "and classified to class: ", outputLabel)
    # testX = array([1.8, 0.3])
    # outputLabel = kNNClassify(testX, dataSet, labels, k)
    # print ("Your input is:", testX, "and classified to class: ", outputLabel)




if __name__ == '__main__':
    test()
