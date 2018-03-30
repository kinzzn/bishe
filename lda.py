# -*- coding:utf-8 -*-
import random
import re
import numpy as np


from settings import Configs

class LDAModel(object):

    def __init__(self, marklist, all_dict, all_text_words, all_text):
        # 为K寻优准备
        self.perp = 0.0

        # 从para.conf中读取参数
        pc = Configs()
        pc.confgetparas()
        self.K = pc.K
        self.alpha = pc.alpha
        self.beta = pc.beta
        self.iter_times = pc.iters
        self.top_words_num = 5 # 每个类特征词个数，也写入conf
        self.marklistlen = len(marklist)
        print('paras: K='+str(self.K)+' alpha='+str(self.alpha)+' beta='+str(self.beta)+' iter='+str(self.iter_times))

        self.all_text = []  # 存储每个文档的所有词语列表[ ['a','b','c'],[...],...]
        for i in range(0, len(all_text)):
            strs = all_text[i]
            wordlist = strs.split(' ')
            self.all_text.append(wordlist)
        self.all_text_words = all_text_words
        self.dicts = all_dict

        self.V = len(self.dicts)
        self.M = len(self.all_text)
        print("输入字典长度：",str(self.V),"输入文档篇数：",str(self.M))

        self.p = np.zeros(self.K)   # 每次gibbs sampling的临时变量
        self.nw = np.zeros((self.V,self.K),dtype=np.int)
        self.nwsum = np.zeros(self.K, dtype=np.int)
        self.nd = np.zeros((self.M,self.K),dtype=np.int)
        self.ndsum = np.zeros(self.M,dtype=np.int)
        self.Z = []
        for i in range(0,self.M):
            self.Z.append(np.zeros(int(self.all_text_words[i]),dtype=np.int))
        # self.Z = np.zeros(self.textlen,每篇词数)

        self.theta = np.zeros((self.M,self.K),dtype=np.float)
        self.phi = np.zeros((self.K,self.V),dtype=np.float)

        # 随机分配类型
        for m in range(0, self.M):
            self.ndsum[m] = int(self.all_text_words[m])
            for y in range(0,int(self.all_text_words[m])):
                topic = random.randint(0,self.K-1)  # topic编号应该从1到K
                self.Z[m][y] = topic
                # nw：获取每篇文章，每篇文章的词，获得词在字典找index，在nw的对应位置的topic操作
                # nw：词index（V）,topic（）
                # print(self.all_text[m][y])
                self.nw[self.dicts[self.all_text[m][y]]][topic] += 1   # 词典，顺序必须和nw的V*K维对应，中间不能有变化
                self.nd[m][topic] += 1
                self.nwsum[topic] += 1
        # print('随机分配结果：')
        # print(self.Z)

    # 核心：sampling函数
    def sampling(self,i,j):
        topic = self.Z[i][j]
        word = self.all_text[i][j]
        self.nw[self.dicts[word]][topic] -= 1
        self.nd[i][topic] -= 1
        self.nwsum[topic] -= 1
        self.ndsum[i] -= 1
        Kalpha = self.K * self.alpha
        Vbeta = self.V * self.beta
        self.p = (self.nw[self.dicts[word]] + self.beta) / (self.nwsum + Vbeta) * \
                 (self.nd[i] + self.alpha) / (self.ndsum[i] + Kalpha)

        for k in range(0,self.K):
            self.p[k] += self.p[k-1]
        u = random.uniform(0,self.p[self.K-1])
        for topic in range(0,self.K):
            if self.p[topic]>u:
                break

        self.nw[self.dicts[word]][topic] += 1
        self.nwsum[topic] += 1
        self.nd[i][topic] += 1
        self.ndsum[i] += 1

        return topic


    # 循环每个词调用sampling函数
    def start(self):
        for itr in range(0,self.iter_times):
            for m in range(0,self.M):
                for w in range(0,int(self.all_text_words[m])):
                    topic = self.sampling(m,w)
                    self.Z[m][w] = topic
        # logging
        print("计算M-K: theta")
        for m in range(0,self.M):
            self.theta[m] = (self.nd[m]+self.alpha)/(self.ndsum[m]+self.K*self.alpha)
        print(self.theta)
        print("计算N-K: phi")
        for k in range(0,self.K):
            self.phi[k] = (self.nw.T[k]+self.beta)/(self.nwsum[k]+self.V*self.beta)
            # ndarray.T:转制 -> phi: K * V（词典序）
        print(self.phi)

        print("打印每类主题词")
        dictstr = str(self.dicts) # 为从值获得键做准备
        self.top_words_num = min(self.top_words_num, self.V)
        for k in range(0,self.K):
            twords = []
            for v in range(0,self.V):
                twords.append((v,self.phi[k][v]))  # twords存储 词典编号，KV对应phi值的元组
            # print('排序前的twords：')
            # print(twords)
            twords.sort(key=lambda i:i[1],reverse=True) # 按第二项排序
            # print('排序后的twords：')
            # print(twords)
            # 精彩的从值获得键的解决，感谢正则表达式测试网站https://regexr.com/
            for y in range(0,self.top_words_num):
                topwordid = twords[y][0]
                m = re.findall('\'\S+?\': ' + str(topwordid) + ',',dictstr)
                wstr = m[0]
                printstr = re.findall('\'(.+?)\'',wstr)
                print(printstr) # 打印每一类的关键词
            print("------")

        # 保存nw,nwsum,nd,ndsum,Z
        # 保存K,beta,alpha,itertimes,top_words_num

        # print("计算后的Z")
        # print(self.Z)

    def word_for_perplexity(self,w,mid):
        wordid = self.dicts[w]
        result = 0
        for i in range(0,self.K):
            pwk = self.phi[i][wordid]
            pkm = self.theta[mid][i]
            mult_result = pwk*pkm
            result += mult_result
        return result


    def perplexity(self):
        totalp = 0.0
        for mm in range(0,self.M):
            mlist = self.all_text[mm]
            for word in mlist:
                if word == '':
                    continue
                totalp += np.log(self.word_for_perplexity(word,mm))
        allN = sum(self.all_text_words)
        perp = np.exp(totalp*(-1)/allN)
        self.perp = perp
        print("类为",str(self.K),"时，计算困惑度为",str(perp))
        return perp

    # 优化算法，使用perplexity算法计算最优的K
    def optimization(self):
        start = 2
        end = 2*self.marklistlen
        perpdict = {}   # 存成字典
        # 设置marklistlen为最小的k，设置最大的k是marklistlen的倍数
        for k in range(start,end):
            self.optimbyk(k)
            perpdict[k] = self.perp
        # 画函数图线，推荐一个合适的K值
        perpvalues = perpdict.values()
        perpkeys = perpdict.keys()

        # 根据斜率差值计算找拐点：计算两点间斜率(由于k值相差1，斜率就是计算perp差值) -> 斜率差值（再做一遍差值，其实就是二阶导） -> 找二阶导数最大的 -> 得到k
        slope1 = []
        slope2 = []
        for k in range(start+1,end):
             slope1.append(perpdict[k]-perpdict[k-1])
        for i in range(1,end-start-1):
            slope2.append(abs(slope1[i]-slope1[i-1]))
        bufferslope2 = slope2
        slope2.sort(reverse=True)
        print("推荐K值为",str(start + bufferslope2.index(slope2[0]) + 1),",",str(start + bufferslope2.index(slope2[1]) + 1))

        # 画图
        import matplotlib.pyplot as plt
        plt.figure()
        plt.plot(perpkeys,perpvalues)
        plt.xlabel("k")
        plt.ylabel("perplexity")
        plt.title("k ~ perplexity")
        plt.show()
        return


    def optimbyk(self,k):
        print("---now k =",str(k),"---")
        self.K = k
        self.start()
        self.perplexity()
        print("---k =",str(k),"finish! ---")