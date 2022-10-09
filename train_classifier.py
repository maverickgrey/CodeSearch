from asyncio.log import logger
from audioop import avg
from config_class import Config
import os
import torch
import torch.nn as nn
from dataset import ClassifierDataset,ClassifierDataset2
from transformers import RobertaConfig,get_linear_schedule_with_warmup
from torch.utils.data import DataLoader
from model import SimpleCasClassifier
from eval_classifier import eval_classifier

def train_classifier(dataloader,classifier,config):
    if not os.path.exists(config.saved_path):
        os.makedirs(config.saved_path)

    classifier.zero_grad()
    loss_func = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(classifier.parameters(),lr=4e-6)
    num_training = len(dataloader)*config.epoches
    scheduler = get_linear_schedule_with_warmup(optimizer,num_warmup_steps=num_training/10,num_training_steps=num_training)


    if os.path.exists(config.saved_path+"/classifier3.pt"):
        classifier.load_state_dict(torch.load(config.saved_path+"/classifier3.pt"))
    if os.path.exists(config.saved_path+"/c_optimizer3.pt"):
        optimizer.load_state_dict(torch.load(config.saved_path+"/c_optimizer3.pt"))
    if os.path.exists(config.saved_path+"/scheduler3.pt"):
        scheduler.load_state_dict(torch.load(config.saved_path+"/c_scheduler3.pt"))
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
            scheduler.step()
            total_step += 1
            if step%500 == 0:
                log_file = open('./model_saved/log.txt','a')
                log = "epoch:{},step:{},avg_loss:{}".format(epoch+1,step,total_loss/total_step)
                print(log)
                log_file.write(log+"\n")
                log_file.close()
            if step%40000 == 0:
                log_file = open('./model_saved/log.txt','a')
                log = "开始evaluation..."
                print(log)
                log_file.write(log+"\n")
                avg_loss,acc = eval_classifier(dataloader,classifier,config,True)
                log_file.write("evaluation: avg_{},acc:{}".format(avg_loss,acc))
                log_file.close()

        torch.save(classifier.state_dict(),config.saved_path+"/classifier3.pt")
        torch.save(optimizer.state_dict(),config.saved_path+"/c_optimizer3.pt")
        torch.save(scheduler.state_dict(),config.saved_path+"/c_scheduler3.pt")

def train_classifier2(classifier,config):
    if not os.path.exists(config.saved_path):
        os.makedirs(config.saved_path)

    classifier.zero_grad()
    loss_func = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(classifier.parameters(),lr=5e-6)

    if os.path.exists(config.saved_path+"/classifier2.pt"):
        classifier.load_state_dict(torch.load(config.saved_path+"/classifier2.pt"))
    if os.path.exists(config.saved_path+"/c_optimizer2.pt"):
        optimizer.load_state_dict(torch.load(config.saved_path+"/c_optimizer2.pt"))
    if config.use_cuda:
        classifier = classifier.cuda()

    total_loss = 0
    total_step = 0
    for epoch in range(config.epoches):
        classifier.train()
        for train_no in range(16):
            print("正在加载数据集{}".format(train_no))
            dataset = ClassifierDataset(config,train_no,'train')
            dataloader = DataLoader(dataset,batch_size=config.train_batch_size)
            print("开始对数据集{}第{}轮的训练".format(train_no,epoch+1))
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
            torch.save(classifier.state_dict(),config.saved_path+"/classifier2.pt")
            torch.save(optimizer.state_dict(),config.saved_path+"/c_optimmizer2.pt")


if __name__ == "__main__":
    choice = 3
    config = Config()
    classifier = SimpleCasClassifier()
    if choice == 1:
        for train_no in range(16):
            dataset = ClassifierDataset(config,train_no,'train')
            dataloader = DataLoader(dataset,config.train_batch_size)
            train_classifier(dataloader,classifier,config)
    elif choice == 2:
        train_classifier2(classifier,config)
    elif choice == 3:
        dataset = ClassifierDataset2(config,'train')
        dataloader = DataLoader(dataset,config.train_batch_size)
        train_classifier(dataloader,classifier,config)