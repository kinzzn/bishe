# -*- coding:utf-8 -*-
# 处理文件，最后返回文件为分好词、标记好类别的文档/文件对象
# python2.7 和python3的处理有不同

import jieba
import re

# 所有的路径都最终在conf中进行统一处理
# 处理分类文件，应具有文件的原类别

class DocProc(object):

    def __init__(self):
        self.marklist = []  # 存储每一类的列表编号
        self.all_dict = {}   # 词典
        self.all_text = []  # 所有处理后的文本，每篇文档是词语列表
        self.all_text_num = []  # 每篇文本的词数


    def doc_process(self):
        # 按之前Ipython的来
        # 分词
        doc_path = 'F:/ProgramInstall/JetBrains/PyCharm/work/bishe/weibo'
        # 读取分好类的txt
        doc_dir = '/'
        # 文档顺序：ZARA，虾米，雅思，RADWIMPS
        doc_name = ['霍金去世.txt','百世快递陷窘境.txt','2018考研国家线.txt','水形物语.txt','微博315.txt']#,'头条新闻.txt']#, '1718436033.txt', '2010639813.txt', '5539705240.txt']

        all_shorttext = []
        for name in doc_name:
            with open(doc_path + doc_dir + name, 'r', encoding='utf8') as f:  # 只在这里固定编码方式，2.7没有问题
                doc = f.read()
                li = doc.split('\n')
                mark = len(li)
                self.marklist.append(mark)
                for content in li:
                    # 去掉内容里的数字编号
                    m = re.findall('\d+:(.+)', content)
                    if m:
                        content = m[0]
                    # 去掉链接
                    m2 = re.findall('[a-z]+://[0-9A-Za-z./]+', content)
                    for links in m2:
                        content = content.replace(links, ' ')
                    # 去掉表情符号
                    m = re.findall('\[\w+\]', content)
                    if m:
                        for emojis in m:
                            content = content.replace(emojis, ' ')
                    # 去掉除'之外的标点符号
                    m = re.findall('[’•：“”，。！？》《【】!"#$%&()*+,-./:;<=>?@[\\]^_`{|}「」~]+', content)
                    if m:
                        for code in m:
                            content = content.replace(code, ' ')
                    content = content.replace('\u200b', '')
                    content = content.replace('\xa0','')
                    content = content.replace('\u3000','')
                    doc_cut = jieba.cut(content)  # _decode)
                    result = ' '.join(doc_cut)
                    # 去掉多余空格
                    result = re.sub(" +", " ", result)
                    #print(result)
                    all_shorttext.append(result)
                # result = ' '.join(doc_cut)
                print(name + ' : finish')
                # print(result)
                # result = result.encode('utf-8')
                # with open('E:/20171204 BS/coderef/scikit-LDA/r'+note+'.txt','w',encoding='utf8') as f2:#只在这里固定编码方式，2.7没有问题
                #    f2.write(result)
            f.close()
            # f2.close()
            # print (note+' : finish')

        all_text_num = 0    # 所有文档数
        for n in self.marklist:
            all_text_num += n
        print(self.marklist) # marklist是每类总篇数
        # print(all_shorttext)    # 还没去停用词的所有词

        # 去除停用词
        # 导入停用词，读取文件转为list
        stpwpath = 'F:/ProgramInstall/JetBrains/PyCharm/work/mylda/input/stop_words.txt'
        stpw_dic = open(stpwpath, 'r')
        stpw_content = stpw_dic.read()
        stpwlst = stpw_content.splitlines()
        stpwlst.append('图片')
        stpwlst.append('来自')
        stpwlst.append('网络')
        stpwlst.append('全文')
        stpwlst.append('】')
        stpw_dic.close()


        dict_count = 0  # 第几个出现的词
        for text in all_shorttext:
            wordnum = 0 # 本文出现词计数
            newtext=''
            segs = text.split(' ')
            for seg in segs:    # 每个seg是一个词
                if seg not in stpwlst:
                    wordnum += 1
                    newtext = newtext + ' ' + seg
                    # 如果键不存在于字典中，将会添加键并将值设为默认值
                    if not seg in self.all_dict :
                        self.all_dict[seg]=dict_count
                        dict_count += 1
            self.all_text.append(newtext.strip())    # strip去掉最开始的所有空格
            self.all_text_num.append(wordnum)

        # print('所有文本：'+str(len(self.all_text)))
        # print(self.all_text)
        # print('词典：')
        # print(self.all_dict)

        print('数据预处理完毕')
        return(self.marklist, self.all_dict, self.all_text_num, self.all_text)

        # 文档集list，每篇的词数，词典->存到相应的文件
        # M，V返回给调用，不一定用
        # marklist
        # all_text_words
        # all_shorttext
