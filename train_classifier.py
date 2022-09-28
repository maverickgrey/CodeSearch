from asyncio.log import logger
from config_class import Config
import os
import torch
import torch.nn as nn
from dataset import ClassifierDataset
from transformers import RobertaConfig
from torch.utils.data import DataLoader
from model import SimpleCasClassifier

def train_classifier(dataloader,classifier,config):
    if not os.path.exists(config.saved_path):
        os.makedirs(config.saved_path)

    classifier.zero_grad()
    loss_func = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(classifier.parameters(),lr=1e-5)

    if os.path.exists(config.saved_path+"/classifier.pt"):
        classifier.load_state_dict(torch.load(config.saved_path+"/classifier.pt"))
    if os.path.exists(config.saved_path+"/c_optimizer.pt"):
        optimizer.load_state_dict(torch.load(config.saved_path+"/c_optimizer.pt"))
    if config.use_cuda:
        classifier = classifier.cuda()

    total_loss = 0
    total_step = 0
    for epoch in range(config.epoches):
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
                print("epoch:{},step:{},avg_loss:{}".format(epoch+1,step,total_loss/total_step))
        torch.save(classifier.state_dict(),config.saved_path+"/classifier.pt")
        torch.save(optimizer.state_dict(),config.saved_path+"/c_optimmizer.pt")


if __name__ == "__main__":
    config = Config()
    classifier = SimpleCasClassifier()
    for train_no in range(16):
        print("加载数据集{}".format(train_no))
        dataset = ClassifierDataset(config,train_no,'train')
        dataloader = DataLoader(dataset,batch_size=config.train_batch_size)
        print("在数据集{}上进行训练".format(train_no))
        train_classifier(dataloader,classifier,config)