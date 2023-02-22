from transformers import RobertaModel
import torch
import torch.nn as nn
import os


# 代码搜索模块的哈希模型
class HashEncoder(nn.Module):
  def __init__(self,encode='all'):
    super(HashEncoder,self).__init__()
    self.encode = encode
    self.encoder = RobertaModel.from_pretrained("microsoft/codebert-base")
    if os.path.exists("../model_saved/encoder3.pt"):
       self.encoder.load_state_dict(torch.load("../model_saved/encoder3.pt"))
    self.hash = HashModule()
  
  def forward(self,inputs):
    if self.encode == 'hash':
        hash_vec = self.hash(inputs)
        return hash_vec
    elif self.encode == 'all':
        vec = self.encoder(inputs,attention_mask=inputs.ne(1)).pooler_output
        hash_vec = self.hash(vec)
        return vec,hash_vec


# 根据论文中的描述，哈希模块就是在原来的模型的基础上添加三个全连接层，并使用tanh激活函数
class HashModule(nn.Module):
    def __init__(self):
        super(HashModule,self).__init__()
        self.fc1 = nn.Linear(768,768)
        self.fc2 = nn.Linear(768,512)
        self.fc3 = nn.Linear(512,128)

    def forward(self,inputs):
        out = self.fc1(inputs)
        out = self.fc2(out)
        out = self.fc3(out)
        return torch.tanh(out)


# 论文中的自定义损失函数，lambda1、lambda2、mu分别为3个超参数
class DeepHashLoss(nn.Module):
    def __init__(self,lambda1,lambda2,mu):
      super(DeepHashLoss,self).__init__()
      self.lambda1 = lambda1
      self.lambda2 = lambda2
      self.mu = mu
    
    # 需要有三个输入，S_F:表示的是欧式空间中的相似度矩阵，为m*m维
    # B_C:其中一个哈希向量矩阵
    # B_D:另外一个哈希向量矩阵,B_C和B_D为m*d维矩阵，m是mini-batch的大小，d是hash向量的长度（这里为128）
    # TODO 用torch.norm(x,'fro')来计算F范数
    def forward(self,S_F,B_C,B_D):
       d = B_C.shape[1]
       S_F = self.mu*S_F
       S_F[S_F>=1]=1
       T_1 = torch.norm(S_F - (torch.matmul(B_C,B_D.T)/d),"fro")**2
       T_2 = self.lambda1*(torch.norm(S_F - (torch.matmul(B_C,B_C.T)/d),"fro")**2)
       T_3 = self.lambda2*(torch.norm(S_F - (torch.matmul(B_D,B_D.T)/d),"fro")**2)
       return T_1+T_2+T_3







       
