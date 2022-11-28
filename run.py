from config_class import Config
from model import CasEncoder,CasClassifier,SimpleCasClassifier
from utils import cos_similarity,get_priliminary,get_info,rerank,load_codebase,query_to_vec
from eval_encoder import eval_encoder
import numpy as np
import torch
import json
import os
import logging
import datetime

logging.getLogger().setLevel(logging.INFO)
# logging.basicConfig(filename="./model_saved/run_log.log",level=logging.INFO)
def run():
    config = Config()
    encoder_nl = CasEncoder(encode='one')
    encoder_pl = CasEncoder(encode='one')
    classifier = SimpleCasClassifier()
    #===================================读取训练好的模型===========================================
    logging.info("开始加载模型...")
    if os.path.exists(config.saved_path+"/encoder3.pt"):
        encoder_nl.load_state_dict(torch.load(config.saved_path+"/encoder3.pt"))
        encoder_pl.load_state_dict(torch.load(config.saved_path+"/encoder3.pt"))

    if os.path.exists(config.saved_path+"/classifier2.pt"):
        classifier.load_state_dict(torch.load(config.saved_path+"/classifier2.pt"))
    if config.use_cuda == True:
        encoder_nl = encoder_nl.cuda()
        encoder_pl = encoder_pl.cuda()
        classifier = classifier.cuda()
    # =========================================加载代码库===========================================
    logging.info("正在加载代码库")
    codebase = load_codebase(config.test_path,config,encoder_pl)

    #=======================================测试模式===============================================
    if config.run_way == 'test':
        ans_k = {1:0,5:0,10:0,15:0}
        mrr = 0
        # 再逐个拿出query进行测试
        with open(config.test_path,'r') as f:
            line_no = 0
            for line in f.readlines():
                js = json.loads(line)
                query = js['docstring']
                query_tokens = config.tokenizer.tokenize(query)
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
        logging.info("ans_k:{},mrr:{}".format(ans_k,mrr))
                    
    #===================================真实查询==============================================
    elif config.run_way == 'truth':
        while(True):
            query = input("你想查询什么？(退出输入c)")
            if query == 'c':
                break
            start_time = datetime.datetime.now()
            query_tokens = query.split(' ')
            query_vec = query_to_vec(query,config,encoder_nl).cpu()
            scores = cos_similarity(query_vec,codebase.code_vecs)
            scores = scores.detach().numpy()
            pre = get_priliminary(scores,codebase,config)
            for _pre in pre:
                final = rerank(query_tokens,_pre,classifier,config)
                get_info(final)
            final_time = datetime.datetime.now()
            time_cost = (final_time-start_time).seconds
            logging.info("本次查询消耗时间：{}s".format(time_cost))
    
if __name__ == "__main__":
    run()