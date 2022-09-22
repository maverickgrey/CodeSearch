from config_class import Config
import os
import torch
import torch.nn as nn
import numpy as np
import json
from utils import cos_similarity,get_priliminary


config = Config()
def eval_encoder(dataloader,encoder,test = False):
  if os.path.exists(config.model_save_path+"/encoder.pt"):
    encoder.load_state_dict(torch.load(config.model_save_path+"/encoder.pt"))

  loss_func = torch.nn.CrossEntropyLoss()
  total_loss = 0
  num_step = 0

  if config.use_cuda:
    encoder = encoder.cuda()

  encoder.eval()
  code_vecs = []
  nl_vecs = []

  for step,example in enumerate(dataloader):
    pl_ids = example[0]
    nl_ids = example[1]

    if config.use_cuda:
      pl_ids = pl_ids.cuda()
      nl_ids = nl_ids.cuda()

    with torch.no_grad():
      code_vec,nl_vec = encoder(pl_ids,nl_ids)
      scores = cos_similarity(nl_vec,code_vec)
      labels = torch.arange(code_vec.shape[0])
      if config.use_cuda:
        labels = labels.cuda()
      loss = loss_func(scores,labels)
      total_loss += loss.item()
      num_step += 1
      code_vecs.append(code_vec)
      nl_vecs.append(nl_vec)
    num_step += 1
  code_vecs = torch.cat(code_vecs,0)
  nl_vecs = torch.cat(nl_vecs,0)
  scores = cos_similarity(nl_vecs,code_vecs).cpu().numpy()

  # 计算mrr值
  rank = []
  # 存储当前nl的答案所在的下标
  nl_no = 0
  for score in scores:
    script = np.argsort(-score,axis=-1,kind='quicksort')
    loc = 1
    for i in script:
      if i == nl_no:
        rank.append(1/loc)
      else:
        loc += 1
    nl_no += 1
  mrr = np.mean(rank)
  print("Current Loss:{},Current MRR :{}".format(total_loss/num_step ,mrr))
  if test:
    return scores

#用测试集对encoder进行测试，并且对NL查询按相似度排序返回结果
def test_encoder(dataloader,encoder,dataset,log = False,ret = False):
    test_result_path = config.data_path+"java_test_0.jsonl"
    if (ret == True and log == True):
        scores = eval_encoder(dataloader,encoder,True)
        results,_ = get_priliminary(scores,dataset,config.filter_K)
        if log:
            nl_no = 1
        log = open(test_result_path,'w')
        for result in results:
            js = {}
            js['nl_idx'] = nl_no
            js['answers'] = []
            for res in result:
                pl_no = res.ids
                js['answers'].append(pl_no)
                log.write(json.dumps(js)+"\n")
        log.close()
        return results

    elif (ret == False and log == True):
        scores = eval_encoder(dataloader,encoder,True)
        results,_ = get_priliminary(scores,dataset,config.filter_K)
        nl_no = 1
        log = open(test_result_path,'w')
        for result in results:
            js = {}
            js['nl_idx'] = nl_no
            js['answers'] = []
            for res in result:
                pl_no = res.ids
                js['answers'].append(pl_no)
            log.write(json.dumps(js)+"\n")
            log.close() 
    else:
        eval_encoder(dataloader,encoder)