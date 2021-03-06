# coding=utf-8
#################################################
# kmeans: k-means cluster  
# Author :wojiushimogui 
# Date   :2015年7月28日11:12:44
# HomePage :http://write.blog.csdn.net/postlist
# Email  :   
#################################################  
from numpy import *  
import time  
import matplotlib.pyplot as plt  
import biKMeans
from biKMeans import *
## step 1: load data  
print ("step 1: load data..." ) 
dataSet = []  
fileIn = open('C:/Code/PycharmProjects/K-means/1/BiKmeans-master/testSet.txt')
for line in fileIn.readlines():  
	lineArr = line.strip().split('\t')  
	dataSet.append([float(lineArr[0]), float(lineArr[1])])  
  
## step 2: clustering...  
print ("step 2: clustering...")  
dataSet = mat(dataSet)  
k = 4

centroids, clusterAssment = biKmeans(dataSet, k)
# centroids, clusterAssment = kmeans(dataSet, k)

## step 3: show the result  
print ("step 3: show the result..."  )
showCluster(dataSet, k, centroids, clusterAssment) 