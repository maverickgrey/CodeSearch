from config_class import Config
import os
import torch
import torch.nn as nn

config = Config()
def train_classifier(dataloader,classifier):
    #TODO load model

    if config.use_cuda:
        classifier = classifier.cuda
    total_loss = 0
    total_step = 0
    classifier.zero_grad()
    loss_func = torch.nn.CrossEntorpyLoss()
    optimizer = torch.optim.AdamW(classifier.parameters(),lr=1e-5)
    for epoch in config.epoches:
        classifier.train()
        for step,example in enumerate(dataloader):
            pl_ids = example[0]
            nl_ids = example[1]
            labels = example[2]
            if config.use_cuda:
                pl_ids = pl_ids.cuda()
                nl_ids = nl_ids.cuda()
                labels = labels.cuda()
            logits = classifier(pl_ids,nl_ids)
            loss = loss_func(logits,labels)
            total_loss += loss.item()
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            total_step += 1
            if step%500 == 0:
                print("epoch:{},step:{},avg_loss:{}".formate(epoch,step,total_loss/total_step))
        #TODO save model



