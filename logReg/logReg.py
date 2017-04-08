#coding=utf-8
from numpy import *
import numpy as np

def loadDataSet():
    dataMat = []; labelMat = []
    fr = open('testSet.txt')
    for line in fr.readlines():
        lineArr = line.strip().split()
        dataMat.append([1.0, float(lineArr[0]), float(lineArr[1])])
        # print dataMat
        # print shape(dataMat)
        # print type(dataMat)
        labelMat.append(int(lineArr[2]))
    return dataMat,labelMat

def sigmoid(inX):
    return 1.0/(1+exp(-inX))

#梯度上升
def gradAscent(dataMatIn, classLabels):
    # print shape(dataMatIn)
    dataMatrix = mat(dataMatIn)             #convert to NumPy matrix
    # print type(dataMatrix)
    # print dataMatrix
    labelMat = mat(classLabels).transpose() #convert to NumPy matrix   100*1
    # print labelMat
    m,n = shape(dataMatrix)                 #100*3
    alpha = 0.001
    maxCycles = 500
    weights = ones((n,1))                   #3*1
    # print weights
    for k in range(maxCycles):              #heavy on matrix operations
        h = sigmoid(dataMatrix*weights)     #matrix mult   100*1   #预测类别
        error = (labelMat - h)              #vector subtraction  100*1    #差值
        weights = weights + alpha * dataMatrix.transpose()* error #matrix mult  3*100    *100*1  #在差值方向调整回归系数
    return weights

# 随机梯度上升
def stocGradAscent0(dataMatrix, classLabels):
    # print dataMatrix

    m,n = shape(dataMatrix)
    alpha = 0.01
    weights = ones(n)   #initialize to all ones
    for i in range(m):
        # print dataMatrix[i]
        # print sum(dataMatrix[i]*weights)
        h = sigmoid(sum(dataMatrix[i] * weights))
        # print h
        # print "LLLLLLLLLLLLLLLLLLLLLLLAAAAAAAAAAAAAAAAAAAAAAAAABBBBBBBBBBBBBBBBBBBBEEEEEEEEEEEEEEEEEELLLLLLLLLLLLLLL"
        # print classLabels[i]
        error = classLabels[i] - h
        # print error
        weights = weights + alpha * error * dataMatrix[i]
        return weights

#改进的随机梯度上升
def stocGradAscent1(dataMatrix, classLabels, numIter=150):
    m,n = shape(dataMatrix)
    weights = ones(n)   #initialize to all ones
    for j in range(numIter):
        dataIndex = range(m)
        for i in range(m):
            alpha = 4/(1.0+j+i)+0.0001    #apha decreases with iteration, does not
            randIndex = int(random.uniform(0,len(dataIndex)))#go to 0 because of the constant
            h = sigmoid(sum(dataMatrix[randIndex]*weights))
            error = classLabels[randIndex] - h
            weights = weights + alpha * error * dataMatrix[randIndex]
            del(dataIndex[randIndex])
    return weights

def plotBestFit(weights):
    print "huituhuituhuitu"
    print weights
    import matplotlib.pyplot as plt
    dataMat,labelMat=loadDataSet()
    dataArr = array(dataMat)
    n = shape(dataArr)[0]
    xcord1 = []; ycord1 = []
    xcord2 = []; ycord2 = []
    for i in range(n):
        if int(labelMat[i])== 1:
            xcord1.append(dataArr[i,1]); ycord1.append(dataArr[i,2])
        else:
            xcord2.append(dataArr[i,1]); ycord2.append(dataArr[i,2])
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(xcord1, ycord1, s=30, c='red', marker='s')
    ax.scatter(xcord2, ycord2, s=30, c='green')
    x = arange(-3.0, 3.0, 0.1)
    # print x
    # print weights[0]
    # print weights[1]*x
    # print weights[0]-weights[1]*x
    # print weights[2]
    y = (-weights[0]-weights[1]*x)/weights[2]
    # print y
    ax.plot(x, y)
    plt.xlabel('X1'); plt.ylabel('X2');
    plt.show()

# 分类  通过分类器预测病马的生死问题
def classifyVector(inX, weights):
    prob = sigmoid(sum(inX * weights))
    if prob > 0.5:
        return 1.0
    else:
        return 0.0


def colicTest():
    frTrain = open('horseColicTraining.txt');
    frTest = open('horseColicTest.txt')
    trainingSet = [];
    trainingLabels = []
    for line in frTrain.readlines():
        currLine = line.strip().split('\t')
        lineArr = []
        for i in range(21):
            lineArr.append(float(currLine[i]))
        trainingSet.append(lineArr)
        trainingLabels.append(float(currLine[21]))
    trainWeights = stocGradAscent1(array(trainingSet), trainingLabels, 1000)
    errorCount = 0;
    numTestVec = 0.0
    for line in frTest.readlines():
        numTestVec += 1.0
        currLine = line.strip().split('\t')
        lineArr = []
        for i in range(21):
            lineArr.append(float(currLine[i]))
        if int(classifyVector(array(lineArr), trainWeights)) != int(currLine[21]):
            errorCount += 1
    errorRate = (float(errorCount) / numTestVec)
    print "the error rate of this test is: %f" % errorRate
    return errorRate


def multiTest():
    numTests = 10;
    errorSum = 0.0
    for k in range(numTests):
        errorSum += colicTest()
    print "after %d iterations the average error rate is: %f" % (numTests, errorSum / float(numTests))


def test():
    #dataArr为List类型
    dataArr,labelMat=loadDataSet()
    # print dataArr
    # print type(dataArr)
    # print mat(dataArr)
    # print array(dataArr)
    # weights =  gradAscent(dataArr,labelMat)                      #梯度上升
    # weights =  stocGradAscent0(array(dataArr),labelMat)         #随机梯度上升
    weights = stocGradAscent1(array(dataArr), labelMat)             #改进的随机梯度上升
    # print weights
    # ddd = weights.getA()
    # print ddd
    # print type(ddd)
    # plotBestFit(weights.getA())                     #梯度上升
    plotBestFit(weights)                                 #随机梯度上升/改进的随机梯度上升


def main():
    multiTest()

if __name__ == '__main__':
    # test()   #为前面的单独输出系数和画图的函数     梯度上升和改进的随机的
    main()     #运行即可得到    通过分类器预测病马的生死问题



#     运行之后弹出的警告（不影响结果）
# E:/PYTHON DOC/PycharmProjects/BestNotLast/logReg/logReg.py:14: RuntimeWarning: overflow encountered in exp
#   return 1.0/(1+exp(-inX))
#
#