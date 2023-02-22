from config_class import Config
from transformers import get_linear_schedule_with_warmup
import torch
import torch.nn as nn
import os
from eval_encoder import eval_encoder
from log_util import get_logger
from hash_model import HashEncoder,DeepHashLoss
import math

"""
对哈希模块进行训练，训练的目的是使得在欧几里得空间的向量分布与距离和在二值向量空间的向量分布于距离尽可能的逼近
"""
# 求两个矩阵的余弦相似度
def matrix_cos_sim(mat_a,mat_b):
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


logger = get_logger()


def train(train_dataloader,hash_model,beta=0.6,eta=0.4,config=Config()):
    if config.use_cuda:
        encoder = encoder.cuda()
    encoder.zero_grad()
    optimizer = torch.optim.AdamW(encoder.parameters(),lr=1e-5)
    num_training_steps = len(train_dataloader)*config.encoder_epoches
    scheduler = get_linear_schedule_with_warmup(optimizer,num_warmup_steps=num_training_steps/10,num_training_steps=num_training_steps)
    loss_func = DeepHashLoss(0.1,0.1,1.5)

    if os.path.exists(config.saved_path+"/hashmodel.pt"):
        encoder.load_state_dict(torch.load(config.saved_path+"/hashmodel.pt"))
    if os.path.exists(config.saved_path+"/h_optimizer.pt"):
        optimizer.load_state_dict(torch.load(config.saved_path+"/h_optimizer.pt"))
    if os.path.exists(config.saved_path+"/h_scheduler.pt"):
        scheduler.load_state_dict(torch.load(config.saved_path+"/h_scheduler.pt"))
    
    for epoch in range(config.encoder_epoches):
        encoder.train()
        total_loss = 0
        tr_num = 0
        for step,examples in enumerate(train_dataloader):
            nl_ids = examples[0]
            pl_ids = examples[1]
            if config.use_cuda:
               nl_ids = nl_ids.cuda()
               pl_ids = pl_ids.cuda()
            nl_vecs,nl_hash = hash_model(nl_ids)
            pl_vecs,pl_hash = hash_model(pl_ids)
            S_D = matrix_cos_sim(nl_vecs,nl_vecs.T)
            S_C = matrix_cos_sim(pl_vecs,pl_vecs.T)
            _S = beta*S_C + (1-beta)*S_D
            S = (1-eta)*_S + eta*(torch.matmul(_S,_S.T)/config.train_batch_size)
            loss = loss_func(S,pl_hash,nl_hash)
            loss.backward()
            total_loss += loss.item()
            current_loss = loss.item()
            tr_num += 1
            optimizer.step()
            optimizer.zero_grad()
            scheduler.step()
            if step%100 == 0:
                logger.info("epoch:{},step:{},avg_loss:{},current_loss:{}".format(epoch+1,step,total_loss/tr_num,current_loss))
            if step%15000 == 0:
                torch.save(encoder.state_dict(),config.saved_path+"/encoder3.pt")
                torch.save(optimizer.state_dict(),config.saved_path+"/e_optimizer3.pt")
                torch.save(scheduler.state_dict(),config.saved_path+"/e_scheduler3.pt")
