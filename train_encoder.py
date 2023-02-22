# 对快速编码器进行训练
from config_class import Config
from transformers import get_linear_schedule_with_warmup
import torch
import torch.nn as nn
import os
from eval_encoder import eval_encoder
import logging
from log_util import LogUtil


log = LogUtil()
logger = log.get_logger()
def train_encoder(train_dataloader,eval_dataloader,encoder,config):
  max_mrr = 0
  if not os.path.exists(config.saved_path):
    os.makedirs(config.saved_path)

  if config.use_cuda:
    encoder = encoder.cuda()

  encoder.zero_grad()
  optimizer = torch.optim.AdamW(encoder.parameters(),lr=1e-5)
  num_training_steps = len(train_dataloader)*config.encoder_epoches
  scheduler = get_linear_schedule_with_warmup(optimizer,num_warmup_steps=num_training_steps/10,num_training_steps=num_training_steps)
  loss_func = torch.nn.CrossEntropyLoss()

  if os.path.exists(config.saved_path+"/encoder3.pt"):
    encoder.load_state_dict(torch.load(config.saved_path+"/encoder3.pt"))
  if os.path.exists(config.saved_path+"/e_optimizer3.pt"):
    optimizer.load_state_dict(torch.load(config.saved_path+"/e_optimizer3.pt"))
  if os.path.exists(config.saved_path+"/e_scheduler3.pt"):
    scheduler.load_state_dict(torch.load(config.saved_path+"/e_scheduler3.pt"))
    
  for epoch in range(config.encoder_epoches):
    total_loss = 0
    tr_num = 0
    encoder.train()
    for step,example in enumerate(train_dataloader):
      pl_ids = example[0]
      nl_ids = example[1]
      if config.use_cuda:
        pl_ids = pl_ids.cuda()
        nl_ids = nl_ids.cuda()
      code_vecs,nl_vecs = encoder(pl_ids,nl_ids)
      '''
        对于损失函数的解释：我们使用了余弦相似度为NL对每个PL计算了得分，
        余弦相似度越高的，得分也就越高，也就意味着我们可以将得分高的PL认为是
        NL的回答，也就意味着我们可以将对应位置的PL作为NL的分类。
        例如当batch-size=4时，每次从编码器中得到4个NL以及4个PL，计算得到
        的余弦相似度为4*4的张量；每一行的0、1、2、3可以看作4个分类，由于输入时我们是将
        nl及对应的pl一同输入的，故而按理说第i行的分类也应该是i。
        CrossEntropy的输入则为bs*bs维的预测值及bs维的标签
      
      '''
      #scores = cos_similarity(nl_vecs,code_vecs)
      scores=(nl_vecs[:,None,:]*code_vecs[None,:,:]).sum(-1)
      # print(scores)
      labels = torch.arange(code_vecs.shape[0])
      # print(labels)
      if config.use_cuda:
        labels = labels.cuda()
      loss = loss_func(scores,labels)
      # print(loss)
      # loss.requires_grad_(True)
      loss.backward()
      # torch.nn.utils.clip_grad_norm(encoder.parameters(),1.0)
      total_loss += loss.item()
      current_loss = loss.item()
      tr_num += 1
      optimizer.step()
      optimizer.zero_grad()
      scheduler.step()
 
      if step%100 == 0:
        logger.info("epoch:{},step:{},avg_loss:{},current_loss:{}".format(epoch+1,step,total_loss/tr_num,current_loss))
      if step%15000 == 0:
        avg_loss,mrr,ans_k = eval_encoder(eval_dataloader,encoder=encoder,config=config,test=False,ret=True,during_train=True)
        logger.info("max_mrr:{},current_mrr:{},ans_k:{}".format(max_mrr,mrr,ans_k))
        if mrr>max_mrr:
          max_mrr = mrr
          torch.save(encoder.state_dict(),config.saved_path+"/encoder3.pt")
          torch.save(optimizer.state_dict(),config.saved_path+"/e_optimizer3.pt")
          torch.save(scheduler.state_dict(),config.saved_path+"/e_scheduler3.pt")
          logger.info("evaluation---avg_loss:{},mrr:{},ans_k:{}".format(avg_loss,mrr,ans_k)+"\n")

# 使用triplet loss来对训练好的encoder进行调整，使用的数据为利用encoder生成的classifier数据，以query为anchor，正样本代码段为positive，负样本代码段为negative
def triplet_train_encoder(encoder,train_dataloader,eval_dataloader,config):
  loss_func = nn.TripletMarginLoss(margin=0.6)
  optimizer = torch.optim.AdamW(encoder.parameters(),1e-5)
  total_loss = 0
  total_step = 0
  if config.use_cuda:
    encoder = encoder.cuda()

  if os.path.exists(config.saved_path+"/encoder3_tuned.pt"):
    logger.info("loading tuned model...")
    encoder.load_state_dict(torch.load(config.saved_path+"/encoder3_tuned.pt"))
  elif os.path.exists(config.saved_path+"/encoder3.pt"):
    logger.info("loading model...")
    encoder.load_state_dict(torch.load(config.saved_path+"/encoder3.pt"))

  if os.path.exists(config.saved_path+"/e_optimizer3_tuned.pt"):
    logger.info("loading optimizer...")
    optimizer.load_state_dict(torch.load(config.saved_path+"/e_optimizer3.pt"))

  for epoch in range(2):
    encoder.train()
    for step,example in enumerate(train_dataloader):
      anchor = example[0]
      positive = example[1]
      negative = example[2]
      if config.use_cuda:
        anchor = anchor.cuda()
        positive = positive.cuda()
        negative = negative.cuda()
      anchor_vec = encoder(None,anchor)
      positive_vec = encoder(positive,None)
      negative_vec = encoder(negative,None)
      loss = loss_func(anchor_vec,positive_vec,negative_vec)
      total_loss += loss.item()
      current_loss = loss.item()
      loss.backward()
      optimizer.step()
      optimizer.zero_grad()
      if step%500 == 0:
        logger.info("epoch:{},step:{},current_loss:{}".format(epoch+1,step,current_loss))
    torch.save(encoder.state_dict(),config.saved_path+"/encoder3_tuned.pt")
    torch.save(optimizer.state_dict(),config.saved_path+"/e_optimizer3_tuned.pt")

      
    

if __name__ == '__main__':
  config = Config()
  # train_dataset = CodeSearchDataset(config,'train')
  # train_sampler = RandomSampler(train_dataset)
  # train_dataloader = DataLoader(train_dataset,config.train_batch_size,sampler=train_sampler,collate_fn=train_dataset.collate_fn)
  # eval_dataset = CodeSearchDataset(config,'eval')
  # eval_dataloader = DataLoader(eval_dataset,config.eval_batch_size,shuffle=True,collate_fn=eval_dataset.collate_fn)
  # encoder = CasEncoder()
  # train_encoder(train_dataloader,eval_dataloader,encoder,config)
  # tune_dataset = TripletTrainData(config)
  # tune_dataloader = DataLoader(tune_dataset,config.train_batch_size)
  # encoder = CasEncoder('one')
  # triplet_train_encoder(encoder,tune_dataloader,None,config)