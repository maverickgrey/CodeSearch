import json
import random
import os
import re
import string
from config_class import Config
from transformers import RobertaTokenizer
from utils import convert_examples_to_features
import torch
import numpy as np
from torch.utils.data import DataLoader,Dataset
from dataset import CodeSearchDataset
from model import CasEncoder

"""
本文件对CodeSearchNet的源数据进行一定的处理：
1、用来构建为慢速分类器训练的分类数据
2、对一些质量不好的数据进行一定的处理（如过滤掉等）
3、对源数据中的一些本项目用不到的属性进行过滤
"""

class TrainData:
    def __init__(self,code_tokens,nl_tokens,code,docstring,func_name,repo) -> None:
        self.code_tokens = code_tokens
        self.nl_tokens = nl_tokens
        self.code = code
        self.docstring = docstring
        self.func_name = func_name
        self.repo = repo

class TargetData:
    def __init__(self,label,code_tokens,nl_tokens,code,docstring) -> None:
        self.label = label
        self.code_tokens = code_tokens
        self.nl_tokens = nl_tokens
        self.code = code
        self.docstring = docstring

class DataFilterer:
    def __init__(self,data_path=None,data=None):
        if data is None:
            self.data_path = data_path
            self.data = self.load_data(self.data_path)
        else:
            self.data = data
        self.filter_table = ['()','=','/','{','}','(',')','*','-',',','[',']','#','"','\'','\\']

    def filter(self,data):
        result = []
        for i in range(len(data)):
            data[i].docstring = self.remove_html(data[i])
            data[i].docstring = self.truncate(data[i])
            if self.too_short(data[i]):
                continue
            elif self.has_chinese(data[i]):
                continue
            else:
                data[i].docstring = self.has_java(data[i])
                result.append(data[i])
        return result

    def load_data(self,data_path):
        data = []
        with open(data_path,'r') as f:
            for line in f.readlines():
                js = json.loads(line)
                data.append(TrainData(js['code_tokens'],js['docstring_tokens'],js['code'],js['docstring'],js['func_name'],js['repo']))
        return data
    
    # 一些过滤规则，True则应该过滤，False则不应该过滤
    # 第一条过滤规则：nl_tokens长度过小的数据需要被过滤.注意，判断nl_tokens的长度是在去除标点符号后判断的
    def too_short(self,example):
        punc = string.punctuation
        nl_tokens = example.docstring.split(' ')
        length = 0
        for token in nl_tokens:
            if token not in punc:
                length += 1
        return True if length<=4 else False
    
    # 第二条过滤规则：将中文的NL过滤出去，因为它们在数据集中是以unicode的形式存储的，这在模型中无法提供有用的语义信息
    def has_chinese(self,example):
        nl = example.docstring
        for str in nl:
            if u'\u4e00' <= str <= u'\u9fff':
                return True
        return False
    
    # 第三条规则：只保留docstring的第一句话(以句号或者换行符作为句子结束标志),同时过滤一些黑名单中的符号
    def truncate(self,example):
        nl = ""
        for char in example.docstring:
            if (char != '.') and (char not in self.filter_table):
                if char == '\n':
                    nl += ' '
                else:
                    nl += char
            elif (char != '.') and (char in self.filter_table):
                continue
            else:
                break
        return nl

    
    # 第四条规则：将javadoc符号拿走
    def has_java(self,example):
        nl_tokens = example.docstring.split(' ')
        nl = ""
        for token in nl_tokens:
            if '@' not in token:
                nl += token
                nl += ' '
        return nl.strip()
    
    #第五条规则：将html标签拿走
    def remove_html(self,example):
        nl = example.docstring
        new_nl = re.sub("</?[\w]+ ?[\S]*>",'',nl)
        return new_nl
    
    #将过滤后的数据写入到指定文件
    def write_filtered_data(self,out_put):
        filtered_data = self.filter(self.data)
        write_to_file(filtered_data,out_put)


# 用单文件数据构建classifier数据集
def process_a_file(data_path,out_path,filter=False,shuffle=False):
    data = []
    with open(data_path,'r') as f:
        for line in f.readlines():
            js = json.loads(line)
            code_tokens = js['code_tokens']
            code = js['code']
            docstring = js['docstring']
            docstring_tokens = js['docstring_tokens']
            data.append(TrainData(code_tokens,docstring_tokens,code,docstring))
    filterer = DataFilterer(data=data) if filter else None
    data = build_examples(data,filterer,shuffle)
    with open(out_path,'w') as ft:
        for _data in data:
            js = {}
            js['docstring'] = _data.docstring
            js['docstring_tokens'] = _data.nl_tokens
            js['code'] = _data.code
            js['code_tokens'] = _data.code_tokens
            js['label'] = _data.label
            ft.write(json.dumps(js)+"\n")


# 构建分类的正样本和负样本
def build_examples(data,data_filter=True,shuffle=False):
    result = []
    if len(data)>=2 and len(data)<4:
    #构建两个样本，当当前data长度在2-4之间时
        for i in range(len(data)):
            scripts = [i]
            sample = TargetData(1,data[i].code_tokens,data[i].nl_tokens,data[i].code,data[i].docstring)
            result.append(sample)
            k=0
            while k<1:
                r = random.randint(0,len(data)-1)
                while (r in scripts):
                    r = random.randint(0,len(data)-1)
                scripts.append(r)
                if data[r].func_name == data[i].func_name:
                    sample = TargetData(1,data[r].code_tokens,data[i].nl_tokens,data[r].code,data[i].docstring)
                    result.append(sample)
                    k += 1
                else:
                    sample = TargetData(0,data[r].code_tokens,data[i].nl_tokens,data[r].code,data[i].docstring)
                    result.append(sample)
                    k += 1
    elif len(data)>=4:
    #每条数据构建4个样本：当当前data长度大于等于4时(1p3n)
        for i in range(len(data)):
            scripts = [i]
            sample = TargetData(1,data[i].code_tokens,data[i].nl_tokens,data[i].code,data[i].docstring)
            result.append(sample)
            k=0
            while k<3:
                r = random.randint(0,len(data)-1)
                while (r in scripts):
                    r = random.randint(0,len(data)-1)
                scripts.append(r)
                if data[r].func_name == data[i].func_name:
                    sample = TargetData(1,data[r].code_tokens,data[i].nl_tokens,data[r].code,data[i].docstring)
                    result.append(sample)
                    k += 1
                else:
                    sample = TargetData(0,data[r].code_tokens,data[i].nl_tokens,data[r].code,data[i].docstring)
                    result.append(sample)
                    k += 1
    if shuffle:
        shuffle_data(result)
    if data_filter:
        filterer = DataFilterer(data=result)
        result = filterer.filter(filterer.data)

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

def filter_encoder_data(data_path,out_path):
    with open(data_path,'r') as f:
        data_filter = DataFilterer(data_path)
        result = data_filter.filter(data_filter.data)
        with open(out_path,'w') as f:
            for example in result:
                js = {}
                js['repo'] = example.repo
                js['func_name'] = example.func_name
                js['docstring'] = example.docstring
                js['docstring_tokens'] = example.nl_tokens
                js['code'] = example.code
                js['code_tokens'] = example.code_tokens
                f.write(json.dumps(js)+"\n")

# 把所有的训练数据弄到一个文件里
def converge():
    prefix = "./CodeSearchNet/classifier/classifier_n"
    out_path = "./CodeSearchNet/classifier/java_classifier_new.jsonl"
    f = open(out_path,'a')
    for train_no in range(16):
        postfix = str(train_no)+".jsonl"
        data_path = prefix+postfix
        with open(data_path,'r') as d:
            for line in d.readlines():
                read = {}
                js = json.loads(line)
                # read['repo'] = js['repo']
                # read['func_name'] = js['func_name']
                read['docstring'] = js['docstring']
                read['docstring_tokens'] = js['docstring_tokens']
                read['code'] = js['code']
                read['code_tokens'] = js['code_tokens']
                read['label'] = js['label']
                f.write(json.dumps(read)+'\n')
    f.close()
    
# 将训练集数据按照函数名分类，每个类别之间空一行
def category_by_funcname(datapath,out_path):
    out_file = open(out_path,'w')
    current_name_prefix = None
    with open(datapath,'r') as f:
        line0 = f.readline()
        js0 = json.loads(line0)
        func_name0 = js0['func_name']
        current_name_prefix = func_name0.split('.')[0]
        out_file.write(line0)
        for line in f.readlines():
            js = json.loads(line)
            func_name = js['func_name']
            func_name = func_name.split('.')
            name_prefix = func_name[0]
            if name_prefix == current_name_prefix:
                out_file.write(line)
            else:
                out_file.write("\n"+line)
                current_name_prefix = name_prefix
    out_file.close()

# 用分类好的数据构建分类数据集
def build_by_categorized(data_path,out_path):
    out = open(out_path,'w')
    result = []
    with open(data_path,'r') as f:
        data = []
        unique_prefix_data = []
        for line in f.readlines():
            if line != "\n":
                js = json.loads(line)
                data.append(TrainData(js['code_tokens'],js['docstring_tokens'],js['code'],js['docstring'],js['func_name'],js['repo']))
            else:
                if len(data) == 1:
                    unique_prefix_data.extend(data)
                    data = []
                else:
                    result.extend(build_examples(data,shuffle=True))
                    data=[]
        result.extend(build_examples(unique_prefix_data,shuffle=True))
    for r in result:
        js = {}
        js['docstring'] = r.docstring
        js['docstring_tokens'] = r.nl_tokens
        js['code'] = r.code
        js['code_tokens'] = r.code_tokens
        js['label'] = r.label
        out.write(json.dumps(js)+"\n")
    out.close()

# 用func_name和repo来增强查询
def strengthen_query(data):
    for example in data:
        nl_tokens = example.repo.split('/'[-1])
        nl_tokens.extend(example.func_name.split('.'))
        nl_tokens.extend(example.nl_tokens)
        example.nl_tokens = nl_tokens

def write_to_file(data,file,type='encoder'):
    with open(file,'w') as f:
        for example in data:
            js = {}
            if type == 'encoder':
                js['repo'] = example.repo
                js['func_name'] = example.func_name
            js['docstring'] = example.docstring
            js['docstring_tokens'] = example.nl_tokens
            js['code'] = example.code
            js['code_tokens'] = example.code_tokens
            if type == 'classifier':
                js['label'] = example.label
            f.write(json.dumps(js)+"\n")

# 处理数据的pipeline:读入一个原始数据文件，分别输出一个encoder的文件以及一个classifier的文件
# 处理流程包括了查询增强、过滤数据、按照func_name分类、构建classifier数据、将数据写入到输出文件等
def pipeline(data_path,encoder_out_path,classifier_out_path,filter = True):
    # 读入数据
    data = read_data(data_path)
    #查询增强
    #strengthen_query(data)

    #过滤数据并写入文件
    if filter == True:
        df = DataFilterer(data_path=None,data=data)
        data = df.filter(df.data)
    write_to_file(data,encoder_out_path,'encoder')

    #按照func_name分类
    temp_file = "./CodeSearchNet/classifier/temp.jsonl"
    category_by_funcname(encoder_out_path,temp_file)
    #构建classifier数据
    build_by_categorized(temp_file,classifier_out_path)
    os.remove(temp_file)

def read_data(data_path):
    res = []
    with open(data_path,'r') as f:
        for line in f.readlines():
            js = json.loads(line)
            example = TrainData(js['code_tokens'],js['docstring_tokens'],js['code'],js['docstring'],js['func_name'],js['repo'])
            res.append(example)
    return res

# 对于每个数据的查询，输出相似度top5的序号
def get_encoder_top_data(encoder,topK,dataloader,config,out_path):
    # code_vecs = []
    # nl_vecs = []
    if os.path.exists(config.saved_path+"/encoder3.pt"):
        encoder.load_state_dict(torch.load(config.saved_path+"/encoder3.pt"))
    if config.use_cuda:
        encoder = encoder.cuda()
    example_no = 0
    offset = 0
    for step,example in enumerate(dataloader):
        code_vecs = []
        nl_vecs = []
        pl_ids = example[0]
        nl_ids = example[1]
        if config.use_cuda:
            pl_ids = pl_ids.cuda()
            nl_ids = nl_ids.cuda()
        with torch.no_grad():
            code_vec,nl_vec = encoder(pl_ids,nl_ids)
            code_vecs.append(code_vec.cpu().numpy())
            nl_vecs.append(nl_vec.cpu().numpy())
        if step %500 == 0:
            print("step:{}".format(step))
        code_vecs = np.concatenate(code_vecs,0)
        nl_vecs = np.concatenate(nl_vecs,0)
        scores = np.matmul(nl_vecs,code_vecs.T)
        for score in scores:
            script = np.argsort(-score,axis=-1,kind='quicksort')
            temp = []
            res = {}
            with open(out_path,'a') as f:
                for i in range(min(len(scores),topK)):
                    index = int(script[i])+offset
                    temp.append(index)
                res['example_no'] = example_no
                res['topK'] = temp
                f.write(json.dumps(res)+"\n")
            example_no+=1
        offset += config.eval_batch_size

#   根据encoder的top数据生成分类器数据
def generate_cls_data(encoder_data,source_data,output_path):
    data = []
    with open(source_data,'r') as fd:
        data = fd.readlines()
    with open(encoder_data,'r') as fe:
        for line in fe.readlines():
            e_data = json.loads(line)
            number = e_data['example_no']
            res_list = e_data['topK']
            #构建正样本：
            positive = {}
            p_data = json.loads(data[number])
            positive['code'] = p_data['code']
            positive['code_tokens'] = p_data['code_tokens']
            positive['docstring'] = p_data['docstring']
            positive['docstring_tokens'] = p_data['docstring_tokens']
            positive['label'] = 1
            #构建负样本：
            negative = {}
            if res_list[0]!=number:
                n_data = json.loads(data[res_list[0]])
                negative['code'] = n_data['code']
                negative['code_tokens'] = n_data['code_tokens']
                negative['docstring'] = p_data['docstring']
                negative['docstring_tokens'] = p_data['docstring_tokens']
                negative['label'] = 0
            else:
                rand = random.randint(0,4)
                while res_list[rand]==number:
                    rand = random.randint(0,4)
                n_data = json.loads(data[res_list[rand]])
                negative['code'] = n_data['code']
                negative['code_tokens'] = n_data['code_tokens']
                negative['docstring'] = p_data['docstring']
                negative['docstring_tokens'] = p_data['docstring_tokens']
                negative['label'] = 0
            with open(output_path,'a') as out:
                out.write(json.dumps(positive)+"\n")
                out.write(json.dumps(negative)+"\n")




if __name__ == "__main__":
    # train_path = "./CodeSearchNet/classifier/java_train_cl.jsonl"
    # out_path = "./CodeSearchNet/classifier/java_train_c.jsonl"
    # process_a_file(train_path,out_path,True,True)
    config = Config()
    # data_path = config.data_path + "/origin_data/java_valid_0.jsonl"
    # eop = config.data_path+"/filtered_data/java_valid_new.jsonl"
    # cop = config.data_path +"/classifier/java_cvalid_new.jsonl"
    # pipeline(data_path,eop,cop,filter=True)
    # converge()
    # dataset = CodeSearchDataset(config,'train')
    # dataloader = DataLoader(dataset,config.eval_batch_size,collate_fn=dataset.collate_fn)
    # encoder = CasEncoder()
    # get_encoder_top_data(encoder,5,dataloader,config,config.data_path+"/classifier/encoder_top.jsonl")
    generate_cls_data(config.data_path+"/classifier/encoder_top.jsonl",config.data_path+"/filtered_data/java_train_new.jsonl",config.data_path+"/classifier/java_train_classifier_p1n1.jsonl")

