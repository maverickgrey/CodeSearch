import code
from dataset import CodeSearchDataset
from torch.utils.data import DataLoader
from config_class import Config
from model import CasEncoder,CasClassifier,SimpleCasClassifier
from utils import cos_similarity,get_priliminary,rerank,CodeStruct,CodeBase
from eval_encoder import eval_encoder
import numpy as np
import torch
import json


def run():
    config = Config()
    dataset = CodeSearchDataset(config.test_path)
    dataloader = DataLoader(dataset,batch_size=config.eval_batch_size)  
    encoder_nl = CasEncoder(encode='nl')
    encoder_pl = CasEncoder(encode='pl')
    classifier = SimpleCasClassifier()

    if config.use_cuda == True:
        encoder_nl = encoder_nl.cuda()
        encoder_pl = encoder_pl.cuda()
        classifier = classifier.cuda()

    nl_vecs = []
    code_base = []

    # 先填充代码库
    codebase = load_codebase(config.test_path,config,encoder_pl)

    # 再逐个拿出query进行测试
    with open(config.test_path,'r') as f:
        for line in f.readlines():
            js = json.loads(line)
            query_tokens = js['docstring_tokens']
            query = ' '.join(query_tokens)
            query_vec = query_to_vec(query,config,encoder_nl)
            score = cos_similarity(query_vec,codebase.code_vecs).numpy()
            # 得到相似度分数后先初步获取K个candidates，之后让classifier对这K个candidates重排序
            prilim = get_priliminary(score,codebase,config.filter_K)
            result = rerank(prilim,codebase,classifier,config)

    print(result)

# 将代码转换成向量并存入数据库中
def load_codebase(data_path,config,encoder):
    code_base = []
    with open(data_path,'r') as d:
        for line in d.readlines():
            js = json.loads(line)
            pl = js['code_tokens'].join(' ')
            pl_tokens = config.tokenizer.tokenize(pl)
            pl_tokens = pl_tokens[:config.max_seq_length-2]
            pl_tokens = [config.tokenizer.cls_token] + pl_tokens +[config.tokenizer.sep_token]
            padding_length = config.max_seq_length-len(pl_tokens)
            pl_tokens += padding_length*[config.tokenizer.pad_token]
            pl_ids = torch.tensor(config.tokenizer.convert_tokens_to_ids(pl_tokens))
            if config.use_cuda:
                pl_ids = pl_ids.cuda()
            pl_vec = encoder(pl_ids,None)
            code_struct = CodeStruct(pl_vec,pl_tokens,pl)
            code_base.append(code_struct)
    return CodeBase(code_base)

# 将自然语言查询转换为向量
def query_to_vec(query,config,encoder):
    query_tokens = config.tokenizer.tokenize(query)
    query_tokens = query_tokens[:config.max_seq_length-2]
    query_tokens = [config.tokenizer.cls_token]+query_tokens+[config.tokenizer.sep_token]
    padding_length = config.max_seq_length - len(query_tokens)
    query_tokens += padding_length*[config.tokenizer.pad_token]
    query_ids = torch.tensor(config.tokenizer.convert_tokens_to_ids(query_ids))
    if config.use_cuda:
        query_ids = query_ids.cuda()
    query_vec = encoder(None,query_ids)
    return [query_vec]


def test2_func():
    config = Config()
    nl_text = "<cls> This is a test <sep>"
    pl_text = "def func ( x , y ) : return x + y <sep>"
    inputs = nl_text+pl_text
    inputs_tokens = config.tokenizer.tokenize(inputs)
    print(inputs_tokens)
    print(config.tokenizer.cls_token)
    print(config.tokenizer.sep_token_id)
    print(config.tokenizer.eos_token_id)

if __name__ == "__main__":
    test2_func()