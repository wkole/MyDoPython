#coding=utf-8

#villan
#datetime:2016.7.24
from numpy import *
import operator
import time
from os import listdir

#文本向量化 32x32 -> 1x1024
def img2vector(filename):
    returnVect = []
    fr = open(filename)
    for i in range(32):
        lineStr = fr.readline()
        for j in range(32):
            returnVect.append(int(lineStr[j]))
    return returnVect
#从文件名中解析分类数字
def classnumCut(fileName):
    fileStr = fileName.split('.')[0]     #take off .txt
    classNumStr = int(fileStr.split('_')[0])
    return classNumStr
#构建训练集数据向量，及对应分类标签向量
def trainingDataSet():
    hwLabels = []
    trainingFileList = listdir('digits/trainingDigits')           #获取目录内容
    m = len(trainingFileList)
    trainingMat = zeros((m,1024))                          #m维向量的训练集
    for i in range(m):
        fileNameStr = trainingFileList[i]
        hwLabels.append(classnumCut(fileNameStr))
        trainingMat[i,:] = img2vector('digits/trainingDigits/%s' % fileNameStr)
    return hwLabels,trainingMat
# 分类器
def classify(inputPoint, dataSet, labels, k):
    dataSetSize = dataSet.shape[0]  # 已知分类的数据集（训练集）的行数
    # 先tile函数将输入点拓展成与训练集相同维数的矩阵，再计算欧氏距离
    diffMat = tile(inputPoint, (dataSetSize, 1)) - dataSet  # 样本与训练集的差值矩阵
    sqDiffMat = diffMat ** 2  # 差值矩阵平方
    sqDistances = sqDiffMat.sum(axis=1)  # 计算每一行上元素的和
    distances = sqDistances ** 0.5  # 开方得到欧拉距离矩阵
    sortedDistIndicies = distances.argsort()  # 按distances中元素进行升序排序后得到的对应下标的列表
    # 选择距离最小的k个点
    classCount = {}
    for i in range(k):
        voteIlabel = labels[sortedDistIndicies[i]]
        classCount[voteIlabel] = classCount.get(voteIlabel, 0) + 1
    # 按classCount字典的第2个元素（即类别出现的次数）从大到小排序
    sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)
    return sortedClassCount[0][0]


#测试函数
def handwritingTest():
    hwLabels,trainingMat = trainingDataSet()    #构建训练集
    testFileList = listdir('digits/testDigits')        #获取测试集
    errorCount = 0.0                            #错误数
    mTest = len(testFileList)                   #测试集总样本数
    t1 = time.time()
    for i in range(mTest):
        fileNameStr = testFileList[i]
        classNumStr = classnumCut(fileNameStr)
        vectorUnderTest = img2vector('digits/testDigits/%s' % fileNameStr)
        #调用knn算法进行测试
        classifierResult = classify(vectorUnderTest, trainingMat, hwLabels, 3)
        print "the classifier came back with: %d, the real answer is: %d" % (classifierResult, classNumStr)
        if (classifierResult != classNumStr): errorCount += 1.0
    print "\nthe total number of tests is: %d" % mTest               #输出测试总样本数
    print "the total number of errors is: %d" % errorCount           #输出测试错误样本数
    print "the total error rate is: %f" % (errorCount/float(mTest))  #输出错误率
    t2 = time.time()
    print "Cost time: %.2fmin, %.4fs."%((t2-t1)//60,(t2-t1)%60)      #测试耗时

if __name__ == "__main__":
    handwritingTest()