from config_class import Config
from model import CasEncoder,CasClassifier,SimpleCasClassifier
from utils import cos_similarity,get_priliminary,get_info,rerank,CodeStruct,CodeBase
from eval_encoder import eval_encoder
import numpy as np
import torch
import json
import os


def run():
    config = Config()
    encoder_nl = CasEncoder(encode='nl')
    encoder_pl = CasEncoder(encode='code')
    classifier = SimpleCasClassifier()

    print("正在加载模型")
    if os.path.exists(config.saved_path+"/encoder.pt"):
        encoder_nl.load_state_dict(torch.load(config.saved_path+"/encoder.pt"))
        encoder_pl.load_state_dict(torch.load(config.saved_path+"/encoder.pt"))

    if os.path.exists(config.saved_path+"/classifier.pt"):
        classifier.load_state_dict(torch.load(config.saved_path+"/classifier.pt"))

    if config.use_cuda == True:
        encoder_nl = encoder_nl.cuda()
        encoder_pl = encoder_pl.cuda()
        classifier = classifier.cuda()
    # 先填充代码库
    print("正在加载代码库")
    codebase = load_codebase(config.test_path,config,encoder_pl)

    if config.run_way == 'test':
        ans_k = {1:0,5:0,10:0,15:0}
        mrr = 0
        # 再逐个拿出query进行测试
        with open(config.test_path,'r') as f:
            line_no = 0
            for line in f.readlines():
                js = json.loads(line)
                query_tokens = js['docstring_tokens']
                query = ' '.join(query_tokens)
                query_vec = query_to_vec(query,config,encoder_nl).cpu()
                score = cos_similarity(query_vec,codebase.code_vecs).detach().numpy()
                # 得到相似度分数后先初步获取K个candidates，之后让classifier对这K个candidates重排序
                prilim = get_priliminary(score,codebase,config)
                for pre in prilim:
                    result = rerank(query_tokens,pre,classifier,config)
                    for i in range(len(result)):
                        if result[i].no == line_no:
                            mrr += 1/(i+1)
                            if i==0:
                                ans_k[1] += 1
                            if i<=4:
                                ans_k[5] += 1
                            if i<=9:
                                ans_k[10] += 1
                            if i<=14:
                                ans_k[15] += 1
                line_no += 1
        mrr /= codebase.base_size
        print("ans_k:{},mrr:{}".format(ans_k,mrr))
                    
    
    elif config.run_way == 'truth':
        while(True):
            query = input("你想查询什么？(退出输入c)")
            if query == 'c':
                break
            query_tokens = query.split(' ')
            query_vec = query_to_vec(query,config,encoder_nl).cpu()
            scores = cos_similarity(query_vec,codebase.code_vecs)
            scores = scores.detach().numpy()
            print("余弦相似度:",scores)
            pre = get_priliminary(scores,codebase,config)
            for _pre in pre:
                final = rerank(query_tokens,_pre,classifier,config)
                get_info(final)

# 将代码转换成向量并存入数据库中
def load_codebase(data_path,config,encoder):
    code_base = []
    with open(data_path,'r') as d:
        code_no = 0 
        for line in d.readlines():
            js = json.loads(line)
            pl = ' '.join(js['code_tokens'])
            origin_pl = js['code']
            pl_tokens = config.tokenizer.tokenize(pl)
            origin_pl_tokens = pl_tokens
            pl_tokens = pl_tokens[:config.max_seq_length-2]
            pl_tokens = [config.tokenizer.cls_token] + pl_tokens +[config.tokenizer.sep_token]
            padding_length = config.max_seq_length-len(pl_tokens)
            pl_tokens += padding_length*[config.tokenizer.pad_token]
            pl_ids = torch.tensor([config.tokenizer.convert_tokens_to_ids(pl_tokens)])
            if config.use_cuda:
                pl_ids = pl_ids.cuda()
            pl_vec = torch.reshape(encoder(pl_ids,None),(768,)).cpu().tolist()
            code_struct = CodeStruct(pl_vec,origin_pl_tokens,origin_pl,code_no)
            code_base.append(code_struct)
            code_no += 1
    return CodeBase(code_base)

# 将一条自然语言查询转换为向量，这个向量的维数为(1,768)，注意是二维的
def query_to_vec(query,config,encoder):
    query_tokens = config.tokenizer.tokenize(query)
    query_tokens = query_tokens[:config.max_seq_length-2]
    query_tokens = [config.tokenizer.cls_token]+query_tokens+[config.tokenizer.sep_token]
    padding_length = config.max_seq_length - len(query_tokens)
    query_tokens += padding_length*[config.tokenizer.pad_token]
    query_ids = torch.tensor([config.tokenizer.convert_tokens_to_ids(query_tokens)])
    if config.use_cuda:
        query_ids = query_ids.cuda()
    query_vec = encoder(None,query_ids)
    return query_vec


def test2_func():
    config = Config()
    classifier = SimpleCasClassifier()
    encoder_pl = CasEncoder('code')
    encoder_nl = CasEncoder('nl')
    if config.use_cuda:
        encoder_pl=encoder_pl.cuda()
        encoder_nl=encoder_nl.cuda()
        classifier = classifier.cuda()
    code_base = load_codebase(config.test_path,config,encoder_pl)
    query = "Concatenates a variable number of ObservableSource sources"
    query_tokens = query.split(' ')
    query_vec = query_to_vec(query,config,encoder_nl).cpu()
    scores = cos_similarity(query_vec,code_base.code_vecs)
    scores = scores.detach().numpy()
    print(scores)
    pre = get_priliminary(scores,code_base,10)
    for _pre in pre:
        final = rerank(query_tokens,_pre,classifier,config)
    get_info(final)
    
if __name__ == "__main__":
    run()