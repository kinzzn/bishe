# -*- coding:utf-8 -*-
from cluster import Kcluster
from distancecal import DistanceCal
from docproc import DocProc
from lda import LDAModel

def evaluation(marklist,effectdict):
    typenum  = len(marklist)
    marklistfortotal = [0,marklist[0]]
    for i in range(1,typenum):
        marklistfortotal.append(marklist[i]+marklistfortotal[i])
    print("编号计算完毕，结果为")
    print(marklistfortotal)
    statisticlist1 = []
    # for i in range(0, typenum):
    #     simplelist.append(0)
    # for i in range(0,typenum):
    #     statisticlist.append(simplelist)
    # print("初始化记录列表为")
    # print(statisticlist)

    for flag1 in range(0,typenum):
        nowlist = effectdict[flag1]
        simplelist = []
        for flag2 in range(0,typenum):
            dicttypelist = [x for x in nowlist if marklistfortotal[flag2]<= x < marklistfortotal[flag2+1]]
            simplelist.append(len(dicttypelist))
        statisticlist1.append(simplelist)
    print("结果恢复完毕")
    print(statisticlist1)

    # 两个statistic二维数组如果使用矩阵转置就方便很多了
    statisticlist2 = []
    for i in range(0,typenum):
        simplelist2 = []
        for j in range(0,typenum):
            simplelist2.append(statisticlist1[j][i])
        statisticlist2.append(simplelist2)
    print("结果恢复完毕2")
    print(statisticlist2)

    maxtypelist = []
    for i in range(0,typenum):
        nowlist = statisticlist2[i]
        if(sum(nowlist) != marklist[i]):
            print("something is calculated wrong!")
            return
        maxtype = nowlist.index(max(nowlist))
        maxtypelist.append(maxtype)
        # 计算误报P漏报R,F值
        # 计算漏报率时，如果分对的过多，说明这其中有的分错了，需要加绝对值
        P = abs(len(effectdict[maxtype])-nowlist[maxtype])/len(effectdict[maxtype])
        R = abs(marklist[i]-nowlist[maxtype])/marklist[i]
        F = 2*(1-P)*(1-R)/(2-P-R)
        # print("第", str(i), "类，误报率 = (", str(len(effectdict[maxtype])),"-",str(nowlist[maxtype]),")/",str(len(effectdict[maxtype])),"=",str(P))
        # print("第", str(i), "类，漏报率 = (", str(marklist[i]), "-", str(nowlist[maxtype]), ")/",str(marklist[i]), "=", str(R))
        print("第",str(i),"类，误报率 =",str(P),"，漏报率 =",str(R),"，F值 = ",str(F))



def run():
    dp = DocProc()
    dp.doc_process()  # 讲道理也可以把所有文件信息通过传参传进来
    lda = LDAModel(dp.marklist, dp.all_dict, dp.all_text_num, dp.all_text)#所有数据从文件读取

    # # 1.--------
    # # 对单独使用lda根据perplexity选择最优的k
    # lda.optimization()

    # 2.------
    # LDA方法，k内置
    lda.start()
    lda.perplexity()
    # 对lda得到的众多主题进行聚类
    k = Kcluster(lda.theta,lda.phi)
    k.start()
    # 处理聚类结果，距离计算，对每个聚类中的每两个文本计算距离
    dc = DistanceCal(lda.theta,k.result,k.newK)
    dc.calbycluster()
    # 根据样本标记类别进行算法评价
    evaluation(dp.marklist,dc.no_record)

if __name__ == '__main__':
    run()
