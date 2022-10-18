from config_class import Config
import torch
import math
import numpy as np

# 为了方便处理encoder的数据所使用的数据结构
class InputFeatures(object):
  def __init__(self,nl_tokens,nl_ids,pl_tokens,pl_ids,id):
    self.nl_tokens = nl_tokens
    self.nl_ids = nl_ids
    self.pl_tokens = pl_tokens
    self.pl_ids = pl_ids
    self.id = id

# 为了方便处理classifier的数据使用的数据结构
class ClassifierFeatures(object):
  def __init__(self,tokens,token_ids,label):
    self.token_ids=token_ids
    self.tokens = tokens
    self.label = label

# 运行整个流程，即进行代码搜索时使用的数据结构
class CodeStruct(object):
  def __init__(self,code_vec,code_tokens,code):
    self.code_tokens = code_tokens
    self.code_vec = code_vec
    self.code = code

# 用来暂时模拟代码库的数据结构，里面存放的是codestruct
class CodeBase(object):
  def __init__(self,code_base):
    self.base_size = len(code_base)
    self.code_base = code_base
    self.code_vecs = self.get_code_vecs(code_base)
  
  def get_code_vecs(self):
    code_vecs = []
    for code in self.code_base:
      code_vecs.append(code.code_vec)
    return code_vecs
  
  def get_code(self,index):
    return self.code_base[index].code
  
  def get_code_vec(self,index):
    return self.code_base[index].code_vec

# 把数据转换成模型能够处理的形式
def convert_examples_to_features(js,id,config,classifier=False):
  if classifier==False:
    nl = ' '.join(js['docstring_tokens'])
    nl_tokens = config.tokenizer.tokenize(nl)
    nl_tokens = nl_tokens[:config.max_seq_length-2]
    nl_tokens = [config.tokenizer.cls_token]+nl_tokens+[config.tokenizer.sep_token]
    nl_ids = config.tokenizer.convert_tokens_to_ids(nl_tokens)
    padding_length = config.max_seq_length - len(nl_ids)
    nl_ids += [config.tokenizer.pad_token_id]*padding_length

    pl = ' '.join(js['code_tokens'])
    pl_tokens = config.tokenizer.tokenize(pl)
    pl_tokens = pl_tokens[:config.max_seq_length-2]
    pl_tokens = [config.tokenizer.cls_token]+pl_tokens+[config.tokenizer.sep_token]
    pl_ids = config.tokenizer.convert_tokens_to_ids(pl_tokens)
    padding_length = config.max_seq_length - len(pl_ids)
    pl_ids += [config.tokenizer.pad_token_id]*padding_length
    return InputFeatures(nl_tokens,nl_ids,pl_tokens,pl_ids,id)
  else:
    nl = ' '.join(js['docstring_tokens'])
    nl_tokens = config.tokenizer.tokenize(nl)
    pl = ' '.join(js['code_tokens'])
    pl_tokens = config.tokenizer.tokenize(pl)
    input_tokens = [config.tokenizer.cls_token]+nl_tokens+[config.tokenizer.sep_token]
    input_tokens += pl_tokens
    input_tokens = input_tokens[:config.max_seq_length-1]
    input_tokens +=[config.tokenizer.sep_token]
    padding_length = config.max_seq_length - len(input_tokens)
    input_tokens += [config.tokenizer.pad_token]*padding_length
    input_ids = config.tokenizer.convert_tokens_to_ids(input_tokens)
    label = js['label']
    return ClassifierFeatures(input_tokens,input_ids,label)
    
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
  for row in range(len(scores)):
    for col in range(len(scores)):
      scores[row][col] /= a_mode[row]*b_mode[col]
  return scores

# 利用编码器输出的向量表示，经相似度计算后获得的一个初步的结果——result返回二维数组，其中每一行为nl对pl的相似度的降序排序
# 其中K表示的是初步选取相似度最高的K个结果
def get_priliminary(score,codebase,K):
  # np.argsort的功能是给一个数组排序，返回排序后的数字对应原来数字所在位置的下标
  # 默认升序排序，这里添加负号即可实现降序
  sort_id = np.argsort(-score,axis=-1,kind='quicksort',order=None)
  result = []
  for index in sort_id:
    if len(result<K):
      result.append(codebase.code_base[index])
  return result


# 在用快速编码器得到初步的结果之后，用慢速分类器对初步的结果进行re-rank
def rerank(query,pre_results,classifier,config):
  final = []
  for pr in pre_results:
    code_tokens = pr.code_tokens
    #TODO
  return final