# -*- coding: utf-8 -*-：
'''
Created on Nov 22, 2010

@author: Peter
'''
from numpy import *
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

xcord0 = []
ycord0 = []
xcord1 = []
ycord1 = []
markers =[]
colors =[]
fr = open('testSet.txt')#this file was generated by 2normalGen.py
for line in fr.readlines():  # 按行读取文件
    lineSplit = line.strip().split('\t')  # 划分，使用字符串中的split()和strip()函数，它们经常在一起出现，注意：划分之后是list形式
    xPt = float(lineSplit[0])  # the first feature
    yPt = float(lineSplit[1])  # the second feature
    label = int(lineSplit[2])  # the target label
    if (label == -1):  # the first class
        xcord0.append(xPt)  # 注意list类型有append()函数，数据的第一维特征
        ycord0.append(yPt)  # 数据的第二维特征
    else:
        xcord1.append(xPt)
        ycord1.append(yPt)

fr.close()  # 关闭文件，其中的数据都已经copy到list中了
fig = plt.figure()  # 句柄取出来
ax = fig.add_subplot(111)  # 定义只有一个视图
ax.scatter(xcord0,ycord0, marker='s', s=90)  # 散点图，负类用正方形
ax.scatter(xcord1,ycord1, marker='o', s=50, c='red')  # 正类用圆形
plt.title('Support Vectors Circled')  # 名称
#下面的circle圈出了SVM中的支持向量
circle = Circle((4.6581910000000004, 3.507396), 0.5, facecolor='none', edgecolor=(0,0.8,0.8), linewidth=3, alpha=0.5)
ax.add_patch(circle)
circle = Circle((3.4570959999999999, -0.082215999999999997), 0.5, facecolor='none', edgecolor=(0,0.8,0.8), linewidth=3, alpha=0.5)
ax.add_patch(circle)
circle = Circle((6.0805730000000002, 0.41888599999999998), 0.5, facecolor='none', edgecolor=(0,0.8,0.8), linewidth=3, alpha=0.5)
ax.add_patch(circle)
#plt.plot([2.3,8.5], [-6,6]) #seperating hyperplane
b = -3.75567; w0=0.8065; w1=-0.2761
x = arange(-2.0, 12.0, 0.1)  # 从-2到12，中间间隔0.1，现在一般用range
y = (-w0*x - b)/w1
ax.plot(x,y)  # plot函数的输入也是两个list：x and y
ax.axis([-2,12,-8,6])  # 定义界面大小
plt.show()  # 最后展示效果，而图像的内容要在ax上面操作，最后才会plt.show()，这一点和MATLAB有区别