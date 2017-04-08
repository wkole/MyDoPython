import SMOP

dataArr, labelArr = SMOP.loadDataSet('testSet.txt')
b, alphas = SMOP.smoP(dataArr, labelArr, 0.6, 0.001, 40)
SMOP.show(dataArr, labelArr, alphas, b)