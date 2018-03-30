# -*- coding:utf-8 -*-
from sklearn.cluster import KMeans
import numpy as np

class Kcluster(object):
    def __init__(self,theta,phi):
        self.M_K = theta
        self.K_N = phi
        [self.M,self.K] = self.M_K.shape
        [self.K, self.N] = self.K_N.shape
        self.newK = 5
        self.result = np.zeros(self.M,dtype=np.int)

    def start(self):
        # 聚类->将已有的矩阵数据视为特征集合，给每个矩阵元素分配一个类别（0~5）
        # 当前矩阵有K维特征，但是在kmeans这个包中怎么表示的则不清楚
        km = KMeans(n_clusters=self.newK, random_state=9, max_iter=50)
        km.fit(self.M_K)
        k_y = km.predict(self.M_K)
        # 根据实际情况设置映射
        k2y = np.arange(0,self.K,dtype=np.int)
        self.result = k2y[k_y]
        print("聚类结果：")
        print(self.result)


