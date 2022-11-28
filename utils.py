from config_class import Config
import torch
import math
import numpy as np
import json
from datastruct import CodeBase,CodeStruct
  
def print_features(features):
  for f in features:
    print("idx:{},nl_tokens:{},nl_ids:{},pl_tokens:{},pl_ids:{}".format(f.idx,f.nl_tokens,f.nl_ids,f.pl_tokens,f.pl_ids))


# 计算两个矩阵中向量之间的余弦相似度——返回的scores是一个二维数组，每一行为nl对每个pl的相似度得分
def cos_similarity(mat_a,mat_b):
  scores = torch.matmul(mat_a,mat_b.T)
  # 分别存储a、b矩阵中各个向量的模长
  a_mode = []
  b_mode = []
  # 计算向量的模长并存储起来
  for vec_a in mat_a:
    a_mode.append(math.sqrt(torch.matmul(vec_a,vec_a.T)))
  for vec_b in mat_b:
    b_mode.append(math.sqrt(torch.matmul(vec_b,vec_b.T)))
  for row in range(len(a_mode)):
    for col in range(len(b_mode)):
      scores[row][col] /= a_mode[row]*b_mode[col]
  # print(a_mode)
  # print(b_mode)
  return scores

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
  for pr in pre_results:
    code_tokens = pr.code_tokens
    input_tokens = [config.tokenizer.cls_token]+query_tokens+[config.tokenizer.sep_token]
    input_tokens += code_tokens
    input_tokens = input_tokens[:config.max_seq_length-1]
    input_tokens += [config.tokenizer.sep_token]
    padding_length = config.max_seq_length - len(input_tokens)
    input_tokens += padding_length*[config.tokenizer.pad_token]
    input_ids = torch.tensor([config.tokenizer.convert_tokens_to_ids(input_tokens)])
    if config.use_cuda:
      classifier = classifier.cuda()
      input_ids = input_ids.cuda()
    logit = classifier(input_ids)
    probs = torch.reshape(torch.softmax(logit,dim=-1).cpu().detach(),(2,))
    re_scores = np.append(re_scores,probs[1].item())
  
  #print("预处理结果中与查询匹配的概率：",re_scores)
  script = np.argsort(-re_scores,-1,'quicksort',None)
  #print("预处理结果中按概率降序的下标:",script)
  for i in script:
    if len(final)<config.final_K:
      final.append(pre_results[i])
  return final

# 读取从codebase里获取的结果数组的信息
def get_info(result):
  for res in result:
    print(res.code)

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