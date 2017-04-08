#coding:utf-8
'''
Created on 2016/8/25

@author: villain
'''

from numpy import *

#加载数据集
def loadDataSet():
    dataMat = []
    labelMat = []
    fp = open("dataSet.txt")
    for line in fp.readlines():
        lineArr = line.strip().split()
        dataMat.append([1.0,float(lineArr[0]),float(lineArr[1])])
        labelMat.append(int(lineArr[2]))

    return dataMat,labelMat

#定义Sigmoid函数
def sigmoid(inX):
    return 1.0/(1+exp(-inX))

#梯度上升算法求最佳回归系数
def gradAscent(dataMatIn,classLabels):
    dataMatrix = mat(dataMatIn)
    labelMat = mat(classLabels).transpose()
    m,n = shape(dataMatrix)
    alpha = 0.001
    maxCycles = 500
    weights = ones((n,1))
    for i in range(maxCycles):
        h = sigmoid(dataMatrix*weights)
        error = labelMat-h
        weights = weights+alpha*dataMatrix.transpose()*error
    return weights
#随机梯度上升算法求回归系数
def stoGradAscent0(dataMatrix,labelMat):
    dataMatrix = array(dataMatrix)
    m,n = shape(dataMatrix)
    alpha = 0.01
    weights = ones(n)
    for i in range(m):
        h = sigmoid(sum(dataMatrix[i] * weights))
        error = labelMat[i] - h
        weights = weights + alpha * error * dataMatrix[i]
    return weights
#改进版的随机梯度上升算法
def stocGradAscent1(dataMatrix,labelMat,numIter=150):
    m,n = shape(dataMatrix)
    weights = ones(n)
    for i in range(numIter):
        dataIndex = range(m)
        for j in range(m):
            alpha = 4/(1.0+j+i)+0.01
            randIndex = int(random.uniform(0,len(dataIndex)))
            h = sigmoid(sum(dataMatrix[randIndex]*weights))
            error = labelMat[randIndex]-h
            weights = weights+alpha*error*dataMatrix[randIndex]
            del(dataIndex[randIndex])
    return weights


#分析数据，画出决策边界
def plotBestFit(weights,data):
    dd=0

