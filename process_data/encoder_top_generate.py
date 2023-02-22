import os
import torch
import numpy as np
import json
from config_class import Config
import random

"""
为生成分类器的训练数据而使用。生成规则如下，对于filtered_data中的每条样本的查询，用encoder先选取出与其相似度最高的K个样本，
这K个样本中若有真实答案，那么就让该查询与真实答案组成正样本，并从其它的样本中挑一个作为和查询组成负样本；
这K个样本中若无真实答案，那么就让该查询与真实答案组成正样本，从其它样本挑选多个组成负样本
"""

config = Config

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

#   根据encoder的top数据生成分类器数据,同时需要的话还能得到triplet数据
def generate_cls_data(encoder_data,source_data,output_path,get_triplet=False):
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
            if get_triplet:
                triplet_data = {}
                triplet_data['anchor'] = positive['docstring']
                triplet_data['positive'] = positive['code']
                triplet_data['negative'] = negative['code']
                with open(config.data_path+"/filtered_data/java_encoder_triplet.jsonl",'a') as ft:
                    ft.write(json.dumps(triplet_data)+"\n")
            with open(output_path,'a') as out:
                out.write(json.dumps(positive)+"\n")
                out.write(json.dumps(negative)+"\n")