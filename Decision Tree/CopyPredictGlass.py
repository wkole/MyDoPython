# coding=utf-8
'''''
Created on 2016年7月28日


@author: Villian
'''
import CopyTree2
import CopyTreePlotter
# import saveTree

fr = open('lenses.txt')
lensesData = [data.strip().split('\t') for data in fr.readlines()]
lensesLabel = ['age', 'prescript', 'astigmatic', 'tearRate']
lensesTree = CopyTree2.createTree(lensesData, lensesLabel)
print lensesData
print lensesTree
CopyTreePlotter.createPlot(lensesTree)