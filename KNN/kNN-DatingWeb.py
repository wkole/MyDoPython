# -*- coding: utf-8 -*-
#测试组成已经可用
#Villian
from numpy import *
import operator

#从文件中读取数据，前三列数据为特征数据，最后一列为标签
def parseData(filename):

    f = open(filename)
    arrayLines = f.readlines()
    #为了方便特征数据的归一化，我们将特征数据和标签分开存放
    dataArray = zeros([len(arrayLines),3]) #特征数据的数据矩阵
    index = 0
    classList = []  #标签的存放

    #解析每行字符串
    for line in arrayLines:

        line = line.strip()
        datalist = line.split('\t')
        dataArray[index,:] = datalist[0:3]
        classList.append(int(datalist[-1]))
        index +=1


    return dataArray,classList


#归一化特征数据
def normalize(dataArray):


    minValue = dataArray.min(0) #获得每列特征数据的最小值组成的数组
    maxValue = dataArray.max(0) #获得每列特征数据的最大值组成的数组
    ranges = maxValue - minValue

    m = dataArray.shape[0]  #获取特征数据矩阵的行数

    minArray = tile(minValue,(m, 1)) #扩大最小值组成的数组，扩大其维度，变为m行数组，每行数据都一样
    maxArray = tile(maxValue,(m, 1)) #扩大最大值组成的数组，扩大其维度，变为m行数组，每行数据都一样

    normArray = (dataArray - minArray) / (maxArray - minArray) #归一化特征数据

    return normArray,ranges,minValue


#k邻近算法
def classify(inArray,dataSet,labels,k):

    dataSetSize = dataSet.shape[0] #获取数据样本的行数

    inSet = tile(inArray,[dataSetSize,1]) #扩大测试数据的数组，扩大其行数为dataSetSize行，每行数据一样

    #以下是求距离d的过程

    diffMat = inSet - dataSet

    sqdiffMat = diffMat**2 # 不是矩阵的平方，而是矩阵元素的平方

    sqSum = sqdiffMat.sum(axis=1) #axis为1时，每行元素 相加，为0是每列元素相加，空时元素全相加

    #sqSum为数组，只有一列元素。

    distances = sqSum**0.5

    sortedIndexes = distances.argsort()# argsort函数返回的是元素从小到大的索引值的一维数组

    classcount = {} #建立一个字典来存放标签结果

    for i in range(k):

        voterLabel = labels[sortedIndexes[i]] #根据索引值获取相对应的标签

        classcount[voterLabel] = classcount.get(voterLabel,0)+1 #对应的标签出现次数增1

    #按照值来排序
    sortedClassCount = sorted(classcount.iteritems(),key=operator.itemgetter(1),reverse=True)

    return sortedClassCount[0][0] #返回出现次数最多的标签，这里没有返回次数


#测试算法(如果分类器的正确率满足要求，既可用来处理约会网站的名单，接受输入，用分类器给出结果)
def datingTest():

    testRatio = 0.10

    dataArray ,labelList = parseData("datingTestSet2.txt")

    normArray ,ranges,minVals = normalize(dataArray)

    m = normArray.shape[0]

    numTest = int(m * testRatio)

    errorCount = 0.0

#用数据样本中的后m-numTest个数据测试前numTest个数据
    for i in range(numTest):

        result = classify(normArray[i,:],normArray[numTest:m,:],labelList[numTest:m],4)

        print "the classifier came back with: %d , the real answer is %d " % (result,labelList[i])

        if(result != labelList[i]): errorCount+=1.0

#输出概率
    print "error rate is %f" %(errorCount/float(numTest))
#
# datingTest()

#预测函数
def classifyPerson():
    resultList = ['not at all', 'in small doses', 'in large doses']
    percentTats = float(raw_input("percentage of time spent playing video games?"))
    ffMiles = float(raw_input("frequent flier miles earned per year?"))
    iceCream = float(raw_input("liters of ice cream consumed per year?"))
    datingDataMat, datingLabels = parseData('datingTestSet2.txt')
    normMat, ranges, minVals = normalize(datingDataMat)
    inArr = array([ffMiles, percentTats, iceCream])
    classifierResult = classify((inArr-minVals)/ranges, normMat, datingLabels, 3)
    print "You will probably like this person:", resultList[classifierResult-1]


if __name__ == '__main__':
    datingTest()
    classifyPerson()


