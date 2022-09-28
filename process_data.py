import json
import random
import os
import re
import string

"""
本文件对CodeSearchNet的源数据进行一定的处理：
1、用来构建为慢速分类器训练的分类数据
2、对一些质量不好的数据进行一定的处理（如过滤掉等）
3、对源数据中的一些本项目用不到的属性进行过滤
"""

class DataFilterer:
    def __init__(self,data_path=None,data=None):
        if data is None:
            self.data_path = data_path
            self.data = self.load_data(self.data_path)
        else:
            self.data = data

    def filter(self,data):
        result = []
        for i in range(len(data)):
            if self.too_short(data[i]):
                continue
            elif self.has_chinese(data[i]):
                continue
            else:
                data[i].nl_tokens = self.too_long(data[i])
                result.append(data[i])
        return result

    def load_data(self,data_path):
        data = []
        with open(data_path,'r') as f:
            for line in f.readlines():
                js = json.loads(line)
                data.append(TrainData(js['code_tokens'],js['docstring_tokens'],js['code'],js['docstring']))
        return data
    
    # 一些过滤规则，True则应该过滤，False则不应该过滤
    # 第一条过滤规则：nl_tokens长度过小的数据需要被过滤.注意，判断nl_tokens的长度是在去除标点符号后判断的
    def too_short(self,example):
        punc = string.punctuation
        nl_tokens = example.nl_tokens
        length = 0
        for token in nl_tokens:
            if token not in punc:
                length += 1
        return True if length<=3 else False
    
    # 第二条过滤规则：将中文的NL过滤出去，因为它们在数据集中是以unicode的形式存储的，这在模型中无法提供有用的语义信息
    def has_chinese(self,example):
        nl = example.docstring
        for str in nl:
            if u'\u4e00' <= str <= u'\u9fff':
                return True
        return False
    
    # 第三条规则：考虑将一些过长的nl_tokens截断(考虑截断tokens长度大于20的，从第一个句号前截断)
    def too_long(self,example):
        if len(example.nl_tokens) > 20:
            nl_tokens = example.nl_tokens
            loc = len(nl_tokens)-1
            for index in range(len(nl_tokens)):
                if nl_tokens[index] == ".":
                    loc = index
                    break
            return nl_tokens[:loc+1]
        else:
            return example.nl_tokens
    

class TrainData:
    def __init__(self,code_tokens,nl_tokens,code,docstring) -> None:
        self.code_tokens = code_tokens
        self.nl_tokens = nl_tokens
        self.code = code
        self.docstring = docstring

class TargetData:
    def __init__(self,label,code_tokens,nl_tokens,code,docstring) -> None:
        self.label = label
        self.code_tokens = code_tokens
        self.nl_tokens = nl_tokens
        self.code = code
        self.docstring = docstring


def process(filter=True,shuffle = False):
    path_prefix = "./CodeSearchNet/origin_data/java_train_"
    data_filter = None

    for i in range(16):
        postfix = str(i)+".jsonl"
        data_path = path_prefix+postfix
        out_path = "./CodeSearchNet/classifier/ctrain_"+postfix
        if (not os.path.exists("./CodeSearchNet/classifier")):
            os.makedirs("./CodeSearchNet/classifier")

        data = []
        with open(data_path,'r') as f:
            for line in f.readlines():
                js = json.loads(line)
                code_tokens = js['code_tokens']
                nl_tokens = js['docstring_tokens']
                code = js['code']
                docstring = js['docstring']
                data.append(TrainData(code_tokens,nl_tokens,code,docstring))
        if filter:
            data_filter = DataFilterer(data=data)
        examples = build_examples(data,data_filter,shuffle)
        with open(out_path,'w') as ft:
            for example in examples:
                js = {}
                js['docstring'] = example.docstring
                js['docstring_tokens'] = example.nl_tokens
                js['code_tokens'] = example.code_tokens
                js['code'] = example.code
                js['label'] = example.label
                ft.write(json.dumps(js)+"\n")


# 构建正样本和负样本
def build_examples(data,data_filter=None,shuffle=False):
    result = []
    for i in range(len(data)):
        # 正样本
        positive = TargetData(1,data[i].code_tokens,data[i].nl_tokens,data[i].code,data[i].docstring)

        # 负样本*2
        r1 = random.randint(0,len(data)-1)
        while(r1 == i):
            r1 = random.randint(0,len(data)-1)
        negative1 = TargetData(0,data[r1].code_tokens,data[i].nl_tokens,data[r1].code,data[i].docstring)

        r2 = random.randint(0,len(data)-1)
        while(r2 == r1 or r2 == i):
            r2 = random.randint(0,len(data)-1)
        negative2 = TargetData(0,data[r2].code_tokens,data[i].nl_tokens,data[r2].code,data[i].docstring)

        result.append(positive)
        result.append(negative1)
        result.append(negative2)
    if shuffle:
        shuffle_data(result)
    if data_filter is not None:
        result = data_filter.filter(result)

    #stat_nltokens(result)
    return result

# 将得到的样本随机打乱
def shuffle_data(data):
    for i in range(len(data)):
        loc = random.randint(0,len(data)-1)
        if loc != i:
            data[i],data[loc] = data[loc],data[i]


# 对样本的nl_tokens的长度进行统计
def stat_nltokens(data):
    with open("./CodeSearchNet/classifier/stat.jsonl",'w') as f:
        stat = {}
        stat['4-6'] = 0
        stat['7-10'] = 0
        stat['11-15'] = 0
        stat['16-20'] = 0
        stat['>20'] = 0
        for example in data:
            if len(example.nl_tokens) in range(4,7):
                stat['4-6'] += 1
            elif len(example.nl_tokens) in range(7,11):
                stat['7-10'] += 1
            elif len(example.nl_tokens) in range(11,16):
                stat['11-15'] += 1
            elif len(example.nl_tokens) in range(16,21):
                stat['16-20'] += 1
            elif len(example.nl_tokens) > 20:
                stat['>20'] += 1
                js = {}
                js['nl'] = example.docstring
                js['docstring_tokens'] = example.nl_tokens
                f.write(json.dumps(js)+"\n")
    print(stat)

def filter_encoder_data():
    prefix = "./CodeSearchNet/origin_data/java_train_"
    if not os.path.exists("CodeSearchNet/filted_data"):
        os.mkdir("CodeSearchNet/filted_data")
    for i in range(16):
        train_path = prefix + str(i)+".jsonl"
        out_path = "./CodeSearchNet/filted_data/java_train_f_"+str(i)+".jsonl"
        filter = DataFilterer(train_path)
        result = filter.filt(filter.data)
        with open(out_path,'w') as f:
            for example in result:
                js = {}
                js['docstring'] = example.docstring
                js['docstring_tokens'] = example.nl_tokens
                js['code'] = example.code
                js['code_tokens'] = example.code_tokens
                f.write(json.dumps(js)+"\n")



process(shuffle=True)
#filter_encoder_data()


