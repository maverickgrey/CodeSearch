from config_class import Config
from model import CasEncoder,SimpleCasClassifier,BiFuncModel
from utils import cos_similarity,get_priliminary,get_info,rerank,load_codebase,query_to_vec,load_codebase_old
import torch
import json
import os
import logging
import time

logging.getLogger().setLevel(logging.INFO)
# logging.basicConfig(filename="./model_saved/run_log.log",level=logging.INFO)
def run():
    config = Config()
    encoder_nl = CasEncoder(encode='one')
    classifier = SimpleCasClassifier()
    #===================================读取训练好的模型===========================================
    logging.info("开始加载模型...")
    if os.path.exists(config.saved_path+"/encoder3.pt"):
        encoder_nl.load_state_dict(torch.load(config.saved_path+"/encoder3.pt"))

    if os.path.exists(config.saved_path+"/classifier2.pt"):
        classifier.load_state_dict(torch.load(config.saved_path+"/classifier2.pt"))
    if config.use_cuda == True:
        encoder_nl = encoder_nl.cuda()
        classifier = classifier.cuda()
    # =========================================加载代码库===========================================
    logging.info("正在加载代码库")
    codebase = load_codebase(config.test_path,config.code_vec_path,config)
    # codebase = load_codebase_old(config.test_path,config,encoder_nl)
    logging.info("代码库加载完成")

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
                # score=(query_vec[:,None,:]*codebase.code_vecs[None,:,:]).sum(-1)
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
                if line_no %200 == 0:
                    print("当前第{}条".format(line_no))
        mrr /= codebase.base_size
        logging.info("ans_k:{},mrr:{}".format(ans_k,mrr))
                    
    #===================================真实查询==============================================
    elif config.run_way == 'truth':
        with open("logs/run_time.txt",'a') as logs:
            logs.write("=======encoder_K:{} , classifier_K:{}=======\n".format(config.filter_K,config.final_K))
        while(True):
            query = input("你想查询什么？(退出输入c)")
            if query == 'c':
                break
            start_time = time.perf_counter()
            query_tokens = query.split(' ')
            query_vec = query_to_vec(query,config,encoder_nl).cpu()
            score_time_begin = time.perf_counter()
            # 计算矩阵乘法时时间开销很小，但之后的步骤开销较大
            scores = cos_similarity(query_vec,codebase.code_vecs)
            scores = scores.detach().numpy()
            score_time_end = time.perf_counter()
            pre_time_begin = time.perf_counter()
            pre = get_priliminary(scores,codebase,config)
            pre_time_end = time.perf_counter()
            rerank_time_begin = time.perf_counter()
            # rerank时是时间开销的大头，使用cpu在第一阶段K=30时甚至能占到4-5s，其中的主要开销又来源于生成拼接向量的时候
            for _pre in pre:
                final = rerank(query_tokens,_pre,classifier,config)
                get_info(final)
            rerank_time_end = time.perf_counter()
            final_time = time.perf_counter()
            time_cost = (final_time-start_time)
            logging.info("本次查询消耗时间：{}s".format(time_cost))
            with open("logs/run_time.txt",'a') as logs:
                logs.write("计算相似度矩阵花费:{}s\n".format(score_time_end-score_time_begin))
                logs.write("获得初排结果花费:{}s\n".format(pre_time_end-pre_time_begin))
                logs.write("重排结果花费：{}s\n".format(rerank_time_end-rerank_time_begin))
                logs.write("本次查询消耗时间：{}s\n\n".format(time_cost))

    
if __name__ == "__main__":
    run()