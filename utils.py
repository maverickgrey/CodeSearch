from config_class import Config
import torch
import math
import numpy as np
import json
from datastruct import CodeBase,CodeStruct
from model import CasEncoder
import time
from functools import wraps
  
def print_features(features):
  for f in features:
    print("idx:{},nl_tokens:{},nl_ids:{},pl_tokens:{},pl_ids:{}".format(f.idx,f.nl_tokens,f.nl_ids,f.pl_tokens,f.pl_ids))


# 计算一个向量和一个矩阵中的向量之间的余弦相似度
# 训练时不会使用到这个函数，该函数会在整个流程运行时才会用到
# 所以目前实际情况下的查询mat_a实际只会是一个向量
def cos_similarity(mat_a,mat_b):
  # mode_begin = time.perf_counter()
  # 计算向量的模长并存储起来
  a_mode = torch.norm(mat_a,p=2,dim=1)
  b_mode = torch.norm(mat_b,p=2,dim=1)
  # mode_end = time.perf_counter()
  # print("计算模长的时间:{}".format(mode_end-mode_begin))

  # final_begin = time.perf_counter()
  a_norm = mat_a/a_mode
  b_norm = mat_b
  for col in range(len(b_norm)):
    b_norm[col] /= b_mode[col]
  # print(a_mode)
  # print(b_mode)
  res = torch.matmul(a_norm,b_norm.T)
  # final_end = time.perf_counter()
  # print("计算最终分数的时间：{}".format(final_end-final_begin))
  return res

# 利用编码器输出的向量表示，经相似度计算后获得的一个初步的结果——result返回二维数组，其中每一行为nl对pl的相似度的降序排序
# 其中K表示的是初步选取相似度最高的K个结果
def get_priliminary(score,codebase,config):
  # np.argsort的功能是给一个数组排序，返回排序后的数字对应原来数字所在位置的下标
  # 默认升序排序，这里添加负号即可实现降序
  sort_ids = np.argsort(-score,axis=-1,kind='quicksort',order=None)
  results = []
  for sort_id in sort_ids:
    result = []
    for index in sort_id:
      if len(result)<config.filter_K:
        result.append(codebase.code_base[index])
    results.append(result)
  return results


# 在用快速编码器得到初步的结果之后，用慢速分类器对初步的结果进行re-rank
def rerank(query_tokens,pre_results,classifier,config):
    final = []
    re_scores = np.array([])
    #pre_results会拿到经过encoder的一个初步结果，pre_results是一个列表，每个元素是一个用来表示每条代码段的数据结构（CodeStruct）
    #接下来是用查询的query_tokens和每个pre_result拼在一起送入分类器，判断它们相匹配的概率，并把概率存到re_scores中
    model_time = 0
    input_batch = []
    max_len = 0
    for pr in pre_results:
      code_tokens = pr.code_tokens
      input_tokens = [config.tokenizer.cls_token]+query_tokens+[config.tokenizer.sep_token]
      input_tokens += code_tokens
      input_tokens = input_tokens[:config.max_seq_length-1]
      input_tokens += [config.tokenizer.sep_token]
      input_batch.append(input_tokens)

    for _input in input_batch: 
      if len(_input) > max_len:
        max_len = len(_input)

    for i in range(len(input_batch)):
      padding_length = max_len - len(input_batch[i])
      input_batch[i] += padding_length*[config.tokenizer.pad_token]
      input_batch[i] = config.tokenizer.convert_tokens_to_ids(input_batch[i])

      
    input_batch = torch.tensor(input_batch,device=config.device)
    model_begin_time = time.perf_counter()
    logit = classifier(input_batch)
    model_end_time = time.perf_counter()
    model_time += (model_end_time-model_begin_time)
    #probs计算耗时大约为0.0001秒左右
    probs = torch.reshape(torch.softmax(logit,dim=-1).cpu().detach(),(config.filter_K,2)).numpy()
    
    re_scores = probs[:,1]
    print("本次model time:{}".format(model_time))
    #print("预处理结果中与查询匹配的概率：",re_scores)
    script = np.argsort(-re_scores,-1,'quicksort',None)
    #print("预处理结果中按概率降序的下标:",script)
    for i in script:
      if len(final)<config.final_K:
        final.append(pre_results[i].code)
    #get_info(final)

# 读取从codebase里获取的结果数组的信息
def get_info(result):
  for res in result:
    print(res)

# 将代码转换成向量并存入数据库中(旧版本)
def load_codebase_old(data_path,config,encoder):
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
            # if config.use_cuda:
            #     pl_ids = pl_ids.cuda()
            pl_ids = pl_ids.to(config.devices)
            pl_vec = torch.reshape(encoder(pl_ids,None),(768,)).cpu().tolist()
            code_struct = CodeStruct(pl_vec,origin_pl_tokens,origin_pl,code_no)
            code_base.append(code_struct)
            code_no += 1
    return CodeBase(code_base)

def load_codebase(code_path,vec_path,config)->CodeBase:
    code_base = []
    code_file = open(code_path,'r')
    vec_file = open(vec_path,'r')
    codes = code_file.readlines()
    vecs = vec_file.readlines()
    for vec in vecs:
        vec_js = json.loads(vec)
        code_no = vec_js['code_no']
        code_vec = vec_js['code_vec']
        code_js = json.loads(codes[code_no])
        code = code_js['code']
        code_tokens = ' '.join(code_js['code_tokens'])
        code_tokens = config.tokenizer.tokenize(code_tokens)
        code_struct = CodeStruct(code_vec,code_tokens,code,code_no)
        code_base.append(code_struct)
    return CodeBase(code_base)



# 将一条自然语言查询转换为向量，这个向量的维数为(1,768)，注意是二维的
def query_to_vec(query,config,encoder):
    query_tokens = config.tokenizer.tokenize(query)
    query_tokens = query_tokens[:config.max_seq_length-2]
    query_tokens = [config.tokenizer.cls_token]+query_tokens+[config.tokenizer.sep_token]
    padding_length = config.max_seq_length - len(query_tokens)
    query_tokens += padding_length*[config.tokenizer.pad_token]
    query_ids = torch.tensor([config.tokenizer.convert_tokens_to_ids(query_tokens)])
    # if config.use_cuda:
    #     query_ids = query_ids.cuda()
    query_ids = query_ids.to(config.device)
    query_vec = encoder(None,query_ids)
    return query_vec

# 求两个矩阵的海明距离
def matrix_hamming(mat_a,mat_b):
   pass

# 自定义函数实现
def vec_hamming(vec1, vec2):
    """返回等长序列之间的汉明距离"""
    if len(vec1) != len(vec2):
        raise ValueError("两向量长度不等，无法计算hamming距离！")
    return sum(el1 != el2 for el1, el2 in zip(vec1, vec2))

# 统计函数耗时
def timer(func):
   @wraps(func)
   def wrapper(*args,**kwargs):
      begin = time.perf_counter()
      result = func(*args,**kwargs)
      end = time.perf_counter()
      print("{} cost {} seconds.".format(func.__name__,end-begin))
      return result
   return wrapper