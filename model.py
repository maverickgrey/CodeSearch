from transformers import RobertaModel
import torch
import torch.nn as nn
import time

# CasCode的快速部分——编码器，这里使用了CodeBert预训练模型.
# 共享参数，为自然语言查询和代码进行编码
class CasEncoder(nn.Module):
  def __init__(self,encode='both'):
    super(CasEncoder,self).__init__()
    self.encode = encode
    self.encoder = RobertaModel.from_pretrained("microsoft/codebert-base")
  
  def forward(self,pl_inputs,nl_inputs):
    # 对NL-PL都进行编码
    encode = self.encode
    if encode=="both":
    # pl_inputs的输入格式为tensor(batch_size,seq_len)
    # code_len在这里的含义即为batch_size
      code_len = pl_inputs.shape[0]
      inputs = torch.cat((pl_inputs,nl_inputs),0)
      outputs = self.encoder(inputs,attention_mask=inputs.ne(1)).pooler_output
    # 由于inputs是由code和nl组合在一起的，故而inputs的大小为(code_batch_size+nl_batch_size,seq_len)
    # 在取出输出的结果向量时就需要根据code_len变量来分别取出code_vec和nl_vec
      code_vec = outputs[:code_len]
      nl_vec = outputs[code_len:]
      return code_vec,nl_vec
    
    # 只会对NL或PL中的一种进行编码
    elif encode == "one":
      if (pl_inputs is not None) and (nl_inputs==None):
        inputs = pl_inputs
        code_vec = self.encoder(inputs,attention_mask=inputs.ne(1)).pooler_output
        return code_vec
      elif (pl_inputs==None) and (nl_inputs is not None):
        inputs = nl_inputs
        nl_vec = self.encoder(inputs,attention_mask=inputs.ne(1)).pooler_output
        return nl_vec
      else:
        raise("此模式下code或nl只能有一个不为空")
    else:
      print("编码器的模式只能为both、nl、code!")



# CasCode的慢速部分——分类器：给定一对NL-PL对，判断PL符合NL的概率是多少
# 用本身带的transformer编码器为输入进行编码后再经全连接层进行分类，而不进行CasEncoder的参数共享
# 是原论文描述的简化版本，直接将NL-PL接在一起后就进行二分类
class SimpleCasClassifier(nn.Module):
  def __init__(self):
    super(SimpleCasClassifier,self).__init__()
    self.encoder = RobertaModel.from_pretrained("microsoft/codebert-base")
    self.fc = nn.Linear(768,2)
  
  def forward(self,inputs):
    vec = self.encoder(inputs,attention_mask = inputs.ne(1)).pooler_output
    # fc层的运算的时间可以忽略
    outputs = self.fc(vec)
    #outputs = torch.softmax(outputs,dim=-1)
    return outputs


# 根据原论文的描述所实现的慢速编码器部分，需要涉及NL-PL对，同时还要拼接它们的按位相减的结果，按位相乘的结果
class CasClassifier(nn.Module):
  def __init__(self,mode='train'):
    super(CasClassifier,self).__init__()
    self.encoder = RobertaModel.from_pretrained("microsoft/codebert-base")
    # 通过fc1后，再经由tanh函数获得NL-PL的关系向量(relation embedding)
    self.fc1 = nn.Linear(3*768,768)
    # 关系向量通过fc2后，再经由sigmoid函数得到NL-PL对的相似分数
    self.fc2 = nn.Linear(768,2)
    self.mode = mode

  def forward(self,pl_inputs,nl_inputs):
    # relation = self.encoder(inputs,attention_mask = inputs.ne(1)).pooler_output
    if self.mode == 'train':
      code_vec = self.encoder(pl_inputs,attention_mask = pl_inputs.ne(1)).pooler_output
      nl_vec = self.encoder(nl_inputs,attention_mask = nl_inputs.ne(1)).pooler_output
    else:
      code_vec = pl_inputs
      nl_vec = nl_inputs
    add = nl_vec+code_vec
    #将两个向量按位相减
    diff = nl_vec - code_vec
    #将两个向量按位相乘
    mul = nl_vec * code_vec
    rel_vec = torch.cat((add,diff,mul),1)
    out1 = self.fc1(rel_vec)
    out1 = torch.tanh(out1)
    out2 = self.fc2(out1)
    out = torch.sigmoid(out2)
    return out
  

# 将编码器和分类器用一个codebert模型表示
class BiFuncModel(nn.Module):
  def __init__(self):
    super(BiFuncModel,self).__init__()
    self.encoder = RobertaModel.from_pretrained("microsoft/codebert-base")
    self.fc = nn.Linear(768,2)
  
  # 这个模型能够接收两个参数：
  # encode_example：等待要编码的样本
  # classify_example：等待要分类的样本
  # 默认情况下它只对样本进行编码，但当有参数传递进classify_example时则不会编码，而是对其进行分类
  def forward(self,encode_example,classify_example=None):
    if classify_example != None:
      begin_time = time.perf_counter()
      vec = self.encoder(classify_example,attention_mask = classify_example.ne(1)).pooler_output
      end_time = time.perf_counter()
      print("vec generated in {}s".format(end_time-begin_time))
      return self.fc(vec)
    else:
      if encode_example is not None:
        return self.encoder(encode_example,attention_mask = encode_example.ne(1)).pooler_output
      else:
        raise("参数encode_example不能为空")