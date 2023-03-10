from config_class import Config
from model import BiFuncModel,CasEncoder
from run import load_codebase
from utils import cos_similarity,get_priliminary,rerank,get_info
import logging
import time
import torch
import numpy as np

def test2_func():
    config = Config()
    model = BiFuncModel()
    if config.use_cuda:
        model = model.cuda()
    logging.info("正在加载代码库")
    codebase = load_codebase(config.test_path,config.code_vec_path,config)
    logging.info("代码库加载完成")
    with open("logs/run_time.txt",'a') as logs:
            logs.write("=======encoder_K:{} , classifier_K:{}===========\n".format(config.filter_K,config.final_K))
    while(True):
        query = input("你想查询什么？(退出输入c)")
        if query == 'c':
            break
        start_time = time.perf_counter()
        query_tokens = query.split(' ')
        query_vec = query_to_vec(query,config,model).cpu()
        score_time_begin = time.perf_counter()
        scores = cos_similarity(query_vec,codebase.code_vecs)
        scores = scores.detach().numpy()
        score_time_end = time.perf_counter()
        pre_time_begin = time.perf_counter()
        pre = get_priliminary(scores,codebase,config)
        pre_time_end = time.perf_counter()
        rerank_time_begin = time.perf_counter()
        for _pre in pre:
            final = rerank(query_tokens,_pre,model,config)
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

def query_to_vec(query,config,encoder):
    query_tokens = config.tokenizer.tokenize(query)
    query_tokens = query_tokens[:config.max_seq_length-2]
    query_tokens = [config.tokenizer.cls_token]+query_tokens+[config.tokenizer.sep_token]
    padding_length = config.max_seq_length - len(query_tokens)
    query_tokens += padding_length*[config.tokenizer.pad_token]
    query_ids = torch.tensor([config.tokenizer.convert_tokens_to_ids(query_tokens)])
    if config.use_cuda:
        query_ids = query_ids.cuda()
    query_vec = encoder(query_ids,None)
    return query_vec

# 在用快速编码器得到初步的结果之后，用慢速分类器对初步的结果进行re-rank
def rerank(query_tokens,pre_results,classifier,config):
  final = []
  re_scores = np.array([])
  #pre_results会拿到经过encoder的一个初步结果，pre_results是一个列表，每个元素是一个用来表示每条代码段的数据结构（CodeStruct）
  #接下来是用查询的query_tokens和每个pre_result拼在一起送入分类器，判断它们相匹配的概率，并把概率存到re_scores中
  tokenize_time = 0
  model_time = 0
  for pr in pre_results:
    tokenize_begin_time = time.perf_counter()
    code_tokens = pr.code_tokens
    input_tokens = [config.tokenizer.cls_token]+query_tokens+[config.tokenizer.sep_token]
    input_tokens += code_tokens
    input_tokens = input_tokens[:config.max_seq_length-1]
    input_tokens += [config.tokenizer.sep_token]
    padding_length = config.max_seq_length - len(input_tokens)
    input_tokens += padding_length*[config.tokenizer.pad_token]
    input_ids = torch.tensor([config.tokenizer.convert_tokens_to_ids(input_tokens)])
    tokenize_end_time = time.perf_counter()
    tokenize_time += (tokenize_end_time-tokenize_begin_time)
    if config.use_cuda:
      classifier = classifier.cuda()
      input_ids = input_ids.cuda()
    model_begin_time = time.perf_counter()
    logit = classifier(None,input_ids)
    model_end_time = time.perf_counter()
    model_time += (model_end_time-model_begin_time)
    probs = torch.reshape(torch.softmax(logit,dim=-1).cpu().detach(),(2,))
    re_scores = np.append(re_scores,probs[1].item())
  print("本次tokenizer time:{}".format(tokenize_time))
  print("本次model time:{}".format(model_time))
  #print("预处理结果中与查询匹配的概率：",re_scores)
  sort_time_begin = time.perf_counter()
  script = np.argsort(-re_scores,-1,'quicksort',None)
  sort_time_end = time.perf_counter()
  print("本次sort time:{}".format(sort_time_end-sort_time_begin))
  #print("预处理结果中按概率降序的下标:",script)
  for i in script:
    if len(final)<config.final_K:
      final.append(pre_results[i])
  return final

# async def lll():
#    print("lll")

# async def zzz():
#    print("zzz")
#    await asyncio.sleep(2)
#    print("zzz end")

# async def run():
#    f1= zzz()
#    f2= lll()
#    await asyncio.gather(f1,f2)
#    return "lll"


if __name__ == "__main__":
  # loop = asyncio.get_event_loop()
  # loop.run_until_complete(run())
  # loop.close()
  config = Config()
  tokenizer = config.tokenizer
  text = "如何实现快速排序"
  t_text = tokenizer.tokenize(text)
  text_ids = tokenizer.convert_tokens_to_ids(t_text)
  print(text_ids)
