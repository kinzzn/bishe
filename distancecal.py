# -*- coding:utf-8 -*-
import numpy as np

class DistanceCal(object):
    def __init__(self,M_K,cluster_result,clusternum):
        self.mk = M_K
        [self.m,self.k] = self.mk.shape
        self.resultdict = {}    # {元组(i,j):sim}
        self.clres = cluster_result # 经过lda和聚类之后的结果，每个m对应一个k（聚类中的k）
        self.clnum = clusternum # 聚类中的类别数量
        self.no_record = {}     # 记录每个聚类含有的文档编号
        self.calprepare = {}    # 按聚类结果将每个文档的theta值准备好，{0:[],1:[],...}

    # kl距离，输入两个向量（一维矩阵）
    # 每两个文章计算距离的函数
    def kl(self,x,y):
        klxy = 0.0
        for i in range(0,self.k):
            pd = x[i]
            klxy += pd*np.log(pd/y[i])
        return klxy

    # 每两个文章计算距离的函数
    # js距离调用kl距离
    def js(self,x,y):
        return np.double((self.kl(x,y)+self.kl(y,x))/2)

    # 矩阵之间所有文章计算散度的函数
    def js_all(self,m_total,m_data,m_no):
        resultbuf = {}
        # 共计算(m-1)!次
        for i in range(0,m_total):
            for j in range(i+1,m_total):
                m = (m_data[i]+m_data[j])/2
                simres = (self.js(m_data[i],m)+self.js(m_data[j],m))/2
                # print("当前计算结果：",str(simres))
                resultbuf[(m_no[i],m_no[j])] = str(simres)
        print("js散度计算完毕")
        return resultbuf

    def calbycluster(self):
        for i in range(0,self.clnum):  # 为每个字典项建立一个列表
            self.calprepare[i] = []
            self.no_record[i] = []
        for i in range(0,len(self.clres)):
            clusternum = self.clres[i]
            self.calprepare[clusternum].append(self.mk[i,:])
            self.no_record[clusternum].append(i)
        # print("分开的结果：")
        # print(self.calprepare)

        for i in range(0,self.clnum):
            totalnum = len(self.calprepare[i])
            totaldata = self.calprepare[i]
            totalno = self.no_record[i]
            result = self.js_all(totalnum,totaldata,totalno)
            self.resultdict[i] = result
        print("全部计算完毕")
        print(self.resultdict)







