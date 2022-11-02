from config_class import Config
import os
import torch
import torch.nn as nn
from dataset import ClassifierDataset,ClassifierDataset2
from transformers import RobertaConfig,get_linear_schedule_with_warmup
from torch.utils.data import DataLoader
from model import SimpleCasClassifier
from eval_classifier import eval_classifier,eval_classifier2

# 训练简单分类器
def train_classifier(dataloader,eval_dataloader,classifier,config):
    max_acc = 0
    if not os.path.exists(config.saved_path):
        os.makedirs(config.saved_path)

    classifier.zero_grad()
    loss_func = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(classifier.parameters(),lr=2e-6)
    num_training = len(dataloader)*config.classifier_epoches
    scheduler = get_linear_schedule_with_warmup(optimizer,num_warmup_steps=num_training/10,num_training_steps=num_training)


    if os.path.exists(config.saved_path+"/classifier2.pt"):
        classifier.load_state_dict(torch.load(config.saved_path+"/classifier2.pt"))
    if os.path.exists(config.saved_path+"/c_optimizer2.pt"):
        optimizer.load_state_dict(torch.load(config.saved_path+"/c_optimizer2.pt"))
    if os.path.exists(config.saved_path+"/scheduler2.pt"):
        scheduler.load_state_dict(torch.load(config.saved_path+"/c_scheduler2.pt"))
    if config.use_cuda:
        classifier = classifier.cuda()

    total_loss = 0
    total_step = 0
    for epoch in range(config.classifier_epoches):
        classifier.train()
        for step,example in enumerate(dataloader):
            inputs = example[0]
            labels = example[1]
            if config.use_cuda:
                inputs = inputs.cuda()
                labels = labels.cuda()
            logits = classifier(inputs)
            loss = loss_func(logits,labels)
            total_loss += loss.item()
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            scheduler.step()
            total_step += 1
            if step%500 == 0:
                log_file = open('./model_saved/log_epoch1.txt','a')
                log = "epoch:{},step:{},avg_loss:{}".format(epoch+1,step,total_loss/total_step)
                print(log)
                log_file.write(log+"\n")
                log_file.close()
            if step%35000 == 0:
                log_file = open('./model_saved/log_epoch1.txt','a')
                log = "开始evaluation..."
                print(log)
                log_file.write(log+"\n")
                avg_loss,acc = eval_classifier(eval_dataloader,classifier,config,train=True,ret=True)
                log_file.write("evaluation: avg_{},acc:{},max_acc:{}".format(avg_loss,acc,max_acc))
                log_file.close()
                if acc>max_acc:
                    max_acc = acc
                    torch.save(classifier.state_dict(),config.saved_path+"/classifier2.pt")
                    torch.save(optimizer.state_dict(),config.saved_path+"/c_optimizer2.pt")
                    torch.save(scheduler.state_dict(),config.saved_path+"/c_scheduler2.pt")

# 训练原版分类器
def train_classifier2(dataloader,classifier,config):
    if not os.path.exists(config.saved_path):
        os.makedirs(config.saved_path)

    classifier.zero_grad()
    loss_func = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(classifier.parameters(),lr=4e-6)

    if os.path.exists(config.saved_path+"/classifier2.pt"):
        classifier.load_state_dict(torch.load(config.saved_path+"/classifier2.pt"))
    if os.path.exists(config.saved_path+"/c_optimizer2.pt"):
        optimizer.load_state_dict(torch.load(config.saved_path+"/c_optimizer2.pt"))
    if config.use_cuda:
        classifier = classifier.cuda()

    total_loss = 0
    total_step = 0
    for epoch in range(config.classifier_epoches):
        classifier.train()
        for step,example in enumerate(dataloader):
            pl_ids = example[0]
            nl_ids = example[1]
            label = example[2]
            if config.use_cuda:
                pl_ids = pl_ids.cuda()
                nl_ids = nl_ids.cuda()
                label = label.cuda()
            logit = classifier(pl_ids,nl_ids)
            loss = loss_func(logit,label)
            total_loss += loss.item()
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            total_step += 1
            if step%500 == 0:
                log_file = open('./model_saved/log_classifier.txt','a')
                log = "epoch:{},step:{},avg_loss:{}".format(epoch+1,step,total_loss/total_step)
                print(log)
                log_file.write(log+"\n")
                log_file.close()
            if step%35000 == 0:
                log_file = open('./model_saved/log_classifier.txt','a')
                log = "开始evaluation..."
                print(log)
                log_file.write(log+"\n")
                avg_loss,acc = eval_classifier2(eval_dataloader,classifier,config,train=True,ret=True)
                log_file.write("evaluation: avg_{},acc:{},max_acc:{}".format(avg_loss,acc,max_acc))
                log_file.close()
                if acc>max_acc:
                    max_acc = acc
                    torch.save(classifier.state_dict(),config.saved_path+"/classifier2.pt")
                    torch.save(optimizer.state_dict(),config.saved_path+"/c_optimizer2.pt")


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
        train_dataset = ClassifierDataset2(config,'train')
        train_dataloader = DataLoader(train_dataset,config.train_batch_size)
        eval_dataset = ClassifierDataset2(config,'eval')
        eval_dataloader = DataLoader(eval_dataset,config.eval_batch_size)
        train_classifier(train_dataloader,eval_dataloader,classifier,config)