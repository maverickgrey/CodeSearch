import torch

# 为了方便处理encoder的数据所使用的数据结构
class EncoderFeatures(object):
  def __init__(self,nl_tokens,nl_ids,pl_tokens,pl_ids,id):
    self.nl_tokens = nl_tokens
    self.nl_ids = nl_ids
    self.pl_tokens = pl_tokens
    self.pl_ids = pl_ids
    self.id = id

# 为了方便处理simpleclassifier的数据使用的数据结构
class SimpleClassifierFeatures(object):
  def __init__(self,tokens,token_ids,label):
    self.token_ids=token_ids
    self.tokens = tokens
    self.label = label

#
class CasClassifierFeatures:
  def __init__(self,pl_tokens,pl_ids,nl_tokens,nl_ids,label):
    self.pl_tokens = pl_tokens
    self.pl_ids = pl_ids
    self.nl_tokens = nl_tokens
    self.nl_ids = nl_ids
    self.label = label

# 运行整个流程，即进行代码搜索时使用的数据结构
class CodeStruct(object):
  def __init__(self,code_vec,code_tokens,code,no):
    self.code_tokens = code_tokens
    self.code_vec = code_vec
    self.code = code
    self.no = no

# 用来暂时模拟代码库的数据结构，里面存放的是codestruct
class CodeBase(object):
  def __init__(self,code_base):
    self.base_size = len(code_base)
    self.code_base = code_base
    self.code_vecs = self.get_code_vecs()
  
  def get_code_vecs(self):
    code_vecs = []
    for code in self.code_base:
      code_vecs.append(code.code_vec)
    return torch.tensor(code_vecs)
  
  def get_code(self,index):
    return self.code_base[index].code
  
  def get_code_vec(self,index):
    return self.code_base[index].code_vec
  
  def get_info(self):
    for c in self.code_base:
      print("code:",c.code)
