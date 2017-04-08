#-*- coding: utf-8 -*-  #添加中文注释

from numpy import *

def loadSimpData():
    datMat = matrix([[ 1. ,  2.1],
        [ 2. ,  1.1],
        [ 1.3,  1. ],
        [ 1. ,  1. ],
        [ 2. ,  1. ]])
    classLabels = [1.0, 1.0, -1.0, -1.0, 1.0]
    return datMat,classLabels

#*****加载数据
def loadDataSet(fileName):
    #特征数（包含最后一列的类别标签）
    numFeat = len(open(fileName).readline().split('\t'))
    dataMat = []; labelMat = []
    fr = open(fileName)
    #按行读取
    for line in fr.readlines():
        lineArr =[]
        curLine = line.strip().split('\t')
        for i in range(numFeat-1):
            lineArr.append(float(curLine[i]))
        #把每个样本特征存入 dataMat中
        dataMat.append(lineArr)
        #存每个样本的标签
        labelMat.append(float(curLine[-1]))
    return dataMat,labelMat


#*****通过取某列特征进行阈值比较，进而对数据分类，阈值的一边为-1类，另一边为+1类
#dataMatrix：样本数据
#dimen：特征编号，将该列特征进行阈值比较对数据分类
#threshVal：阈值
#threshIneq：两种阈值选择：‘lt’：小于阈值的预测类别值为-1，‘gt’：大于阈值的预测类别取-1
def stumpClassify(dataMatrix,dimen,threshVal,threshIneq):#just classify the data
    #单层决策树，根据与阈值比较，返回类别
    retArray = ones((shape(dataMatrix)[0],1))
    #小于阈值，取类别-1
    if threshIneq == 'lt':
        retArray[dataMatrix[:,dimen] <= threshVal] = -1.0
    #大于阈值，取类别-1
    else:
        retArray[dataMatrix[:,dimen] > threshVal] = -1.0
    return retArray

# *****构建单层决策分类树，adaboost每次迭代过程中的基分类器
# dataArr：训练样本数据
# classLabels：训练数据的真实类别标签
# D：样本权值
def buildStump(dataArr,classLabels,D):
    dataMatrix = mat(dataArr);
    labelMat = mat(classLabels).T
    #训练数据个数m,特征维数n
    m,n = shape(dataMatrix)
    #某个特征值范围内，递增选阈值，numSteps为所选阈值总数
    numSteps = 10.0
    #字典，存储在给定样本权值D下的决策树
    bestStump = {}
    bestClasEst = mat(zeros((m,1)))#最佳预测类别
    minError = inf #初始化误差率为无穷
    #遍历所有特征维度
    for i in range(n):
        #求当前特征的范围值，从小到大按一定步长递增选择阈值
        rangeMin = dataMatrix[:,i].min(); rangeMax = dataMatrix[:,i].max();
        stepSize = (rangeMax-rangeMin)/numSteps #步长

        #遍历整个特征值范围，递增取步长
        #（多取了两个值j=-1,j=int(numSteps)+1）
        for j in range(-1,int(numSteps)+1):
        #for j in range(0,int(numSteps)):
            #两种阈值划分：'lt'表示小于阈值取-1，大于阈值取+1；‘gt’与‘lt’相反
            for inequal in ['lt', 'gt']:
                #按步长单调递增取阈值
                threshVal = (rangeMin + float(j) * stepSize)
                #利用单层决策树预测类别
                predictedVals = stumpClassify(dataMatrix,i,threshVal,inequal)
                #统计预测错误情况：预测正确的样本记0，预测错误的记1
                errArr = mat(ones((m,1)))
                errArr[predictedVals == labelMat] = 0
                #分类误差率：错分样本的权值之和
                weightedError = D.T*errArr
                # print "split: dim %d, thresh %.2f, thresh ineqal: %s, the weighted error is %.3f" % (i, threshVal, inequal, weightedError)
                #求最小的分类误差率下的决策树分类器
                if weightedError < minError:
                    minError = weightedError
                    #基分类器的预测结果
                    bestClasEst = predictedVals.copy() #备份预测值
                    #最佳单层决策树的信息
                    bestStump['dim'] = i #划分维度
                    bestStump['thresh'] = threshVal#划分阈值
                    bestStump['ineq'] = inequal #划分方式,取'lt'or'gt'
    return bestStump,minError,bestClasEst


#*****完整的adaboost算法实现
#dataArr：训练样本数据
#classLabels：训练样本的类别
#numIt：迭代次数
def adaBoostTrainDS(dataArr,classLabels,numIt=40):
    #存每次迭代中生成的弱分类器
    weakClassArr = []
    #训练样本数
    m = shape(dataArr)[0]

    #step1:初始化权值分布
    D = mat(ones((m,1))/m)
    #S
    aggClassEst = mat(zeros((m,1)))

    #step2：迭代
    for i in range(numIt):
        #step2.1-2.2:基本弱分类器及分类误差率
        bestStump,error,classEst = buildStump(dataArr,classLabels,D)
        # print "D:",D.T

        #step2.3:计算alpha
        #max(error,1e-16)为了防止error=0时除以0溢出
        alpha = float(0.5*log((1.0-error)/max(error,1e-16)))
        # print "alpha:",alpha
        #存最佳分类器的权重alpha
        bestStump['alpha'] = alpha
        #将各个基分类器的相关信息存在数组中
        weakClassArr.append(bestStump)
        #打印预测的类别
        # print "classEst: ",classEst.T

        #step2.4:更新样本权值D
        #计算更新权值的指数部分
        expon = multiply(-1*alpha*mat(classLabels).T,classEst)
        #更新样本权值D
        D = multiply(D,exp(expon))
        D = D/D.sum() #归一化

        #step3：构建本次迭代后的各个基分类器的线性组合f(x)
        #本次迭代的累计估计值之和
        aggClassEst += alpha*classEst
        # print "aggClassEst: ",aggClassEst.T

        #step4：本次迭代后的分类器G(x)=sign(f(x))
        #本次迭代后训练样本的分类情况：错误标1，正确标0,#计算分类错误率：错误数/总数
        # print  "sign(aggClassEst):",sign(aggClassEst)
        aggErrors = multiply(sign(aggClassEst) != mat(classLabels).T,ones((m,1)))
        # print "sign(aggClassEst) != mat(classLabels).T:",sign(aggClassEst) != mat(classLabels).T
        # print "aggErrors:",aggErrors
        #本轮迭代后集成的分类错误率
        errorRate = aggErrors.sum()/m
        # 这是训练分类器的错误率，以最后一次为准（villian）
        print "total error: ",errorRate
        #当分类错误率为0，停止迭代
        if errorRate == 0.0: break
    return weakClassArr,aggClassEst

'''classifierArr[0][i]['dim'] 和书上classifierArr[i]['dim']略有区别   因为classifierArr返回的是一个含有([{'dim': 0, 'ineq': 'lt', 'thresh': 1.3, 'alpha': 0.6931471805599453}, {'dim': 1, 'ineq': 'lt', 'thresh': 1.0, 'alpha': 0.9729550745276565}, {'dim': 0, 'ineq': 'lt', 'thresh': 0.90000000000000002, 'alpha': 0.8958797346140273}], matrix([[ 1.17568763],
        [ 2.56198199],
        [-0.77022252],
        [-0.77022252],
        [ 0.61607184]]))的tuple
'''
def adaClassify(datToClass,classifierArr):
    dataMatrix = mat(datToClass)#do stuff similar to last aggClassEst in adaBoostTrainDS
    m = shape(dataMatrix)[0]
    aggClassEst = mat(zeros((m,1)))
    for i in range(len(classifierArr)):
        classEst = stumpClassify(dataMatrix,classifierArr[i]['dim'],\
                                 classifierArr[i]['thresh'],\
                                 classifierArr[i]['ineq'])#call stump classify
        aggClassEst += classifierArr[i]['alpha']*classEst
        print aggClassEst
    # print sign(aggClassEst)
    return sign(aggClassEst)


def plotROC(predStrengths, classLabels):
    import matplotlib.pyplot as plt
    # 光标位置
    cur = (1.0,1.0) #cursor
    ySum = 0.0 #variable to calculate AUC
    #正类样本数
    numPosClas = sum(array(classLabels)==1.0)
    #y轴修正步长
    yStep = 1/float(numPosClas)
    #x轴修正步长
    print "yStep:",yStep

    xStep = 1/float(len(classLabels)-numPosClas)
    print "xStep:",xStep
    #预测估计值从小到大排序
    sortedIndicies = predStrengths.argsort()#get sorted index, it's reverse
    fig = plt.figure()
    fig.clf()
    ax = plt.subplot(111)
    #loop through all the values, drawing a line segment at each point
    for index in sortedIndicies.tolist()[0]:
        if classLabels[index] == 1.0:
            delX = 0; delY = yStep;  #y轴方向下降一个步长
        else:
            delX = xStep; delY = 0;  #x轴方向倒退一个步长
            ySum += cur[1]
        #draw line from cur to (cur[0]-delX,cur[1]-delY)
        ax.plot([cur[0],cur[0]-delX],[cur[1],cur[1]-delY], c='b')
        cur = (cur[0]-delX,cur[1]-delY)
    ax.plot([0,1],[0,1],'b--')
    plt.xlabel('False positive rate'); plt.ylabel('True positive rate')
    plt.title('ROC curve for AdaBoost horse colic detection system')
    ax.axis([0,1,0,1])
    plt.show()
    print "the Area Under the Curve is: ",ySum*xStep

def main0():
    D = mat(ones((5,1))/5)
    # print D
    datMat , classLabels = loadSimpData()
    bestStump, minError, bestClasEst =  buildStump(datMat,classLabels,D)
    # print bestStump,minError,bestClasEst
    print "bestStump:",bestStump
    print "minError:",minError
    print "bestClasEst:",bestClasEst

def main1():
    datMat, classLabels = loadSimpData()
    classifierArray = adaBoostTrainDS(datMat,classLabels,9)
    print classifierArray
    # 下面几行是我做测试时候用的
    # print classifierArray[0]
    # print classifierArray[1]
    # print classifierArray[0][0]['dim']
    # print classifierArray[0][1]
    # print classifierArray[0][2]


def main2():
    datMat, classLabels = loadSimpData()
    classifierArray = adaBoostTrainDS(datMat,classLabels,30)
    print classifierArray

    # siaggCla = adaClassify([0,0],classifierArray)
    # print siaggCla
    siaggCla1 = adaClassify([[5,5],[0,0]],classifierArray)
    print siaggCla1

def main3():
    datArr,labelArr = loadDataSet('horseColicTraining2.txt')
    # 此处会输出训练错误率
    classifierArray,aggClassEst = adaBoostTrainDS(datArr,labelArr,10)

    testArr,testLabelArr = loadDataSet('horseColicTest2.txt')
    prediction10 = adaClassify(testArr,classifierArray)
    errArr = mat(ones((67,1)))
    errArroutput = errArr[prediction10!=mat(testLabelArr).T].sum()
    # 要得到错误率，只需将上述错分样例的个数除以67而已
    # 此处会输出测试错误率
    print "MY**************ERROR:", float(errArroutput)/67

def main4():
    datArr, labelArr = loadDataSet('horseColicTraining2.txt')
    classifierArray, aggClassEst = adaBoostTrainDS(datArr, labelArr, 10)
    plotROC(aggClassEst.T,labelArr)

if __name__ == '__main__':
    # main0()
    # main1()
    # main2()
    main3()
    main4()