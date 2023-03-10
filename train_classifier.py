from config_class import Config
import os
import torch
import torch.nn as nn
from dataset import ClassifierDataset,ClassifierDataset2
from transformers import RobertaConfig,get_linear_schedule_with_warmup
from torch.utils.data import DataLoader
from model import SimpleCasClassifier,CasClassifier
from eval_classifier import eval_classifier,eval_classifier2
from log_util import LogUtil


log = LogUtil()
logger = log.get_logger()
TRAIN_DATE = "307"
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
    # if config.use_cuda:
    #     classifier = classifier.cuda()
    classifer = classifer.to(config.device)

    total_loss = 0
    total_step = 0
    for epoch in range(config.classifier_epoches):
        classifier.train()
        for step,example in enumerate(dataloader):
            inputs = example[0]
            labels = example[1]
            # if config.use_cuda:
            #     inputs = inputs.cuda()
            #     labels = labels.cuda()
            logits = classifier(inputs)
            loss = loss_func(logits,labels)
            total_loss += loss.item()
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            scheduler.step()
            total_step += 1
            if step%500 == 0:
                logger.info("epoch:{},step:{},avg_loss:{}".format(epoch+1,step,total_loss/total_step))
            if step%35000 == 0:
                logger.info("开始evaluation")
                avg_loss,acc = eval_classifier(eval_dataloader,classifier,config,train=True,ret=True)
                logger.info("evaluation: avg_{},acc:{},max_acc:{}".format(avg_loss,acc,max_acc))
                if acc>max_acc:
                    max_acc = acc
                    torch.save(classifier.state_dict(),config.saved_path+"/classifier2.pt")
                    torch.save(optimizer.state_dict(),config.saved_path+"/c_optimizer2.pt")
                    torch.save(scheduler.state_dict(),config.saved_path+"/c_scheduler2.pt")

# 训练原版分类器
def train_classifier2(dataloader,classifier,config,eval_dataloader=None):
    if not os.path.exists(config.saved_path):
        os.makedirs(config.saved_path)

    classifier.zero_grad()
    loss_func = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(classifier.parameters(),lr=2e-6)

    if os.path.exists(config.saved_path / "classifier_{}.pt".format(TRAIN_DATE)):
        classifier.load_state_dict(torch.load(config.saved_path / "classifier_{}.pt".format(TRAIN_DATE)))
    if os.path.exists(config.saved_path / "c_optimizer_{}.pt".format(TRAIN_DATE)):
        optimizer.load_state_dict(torch.load(config.saved_path / "c_optimizer_{}.pt".format(TRAIN_DATE)))
    # if config.use_cuda:
    #     classifier = classifier.cuda()
    classifier = classifier.to(config.device)

    total_loss = 0
    total_step = 0
    for epoch in range(config.classifier_epoches):
        classifier.train()
        for step,example in enumerate(dataloader):
            # inputs_ids = example[0]
            pl_ids = example[1]
            nl_ids = example[2]
            label = example[3]
            # if config.use_cuda:
            #     inputs_ids = inputs_ids.cuda()
            #     pl_ids = pl_ids.cuda()
            #     nl_ids = nl_ids.cuda()
            #     label = label.cuda()
            logit = classifier(pl_ids,nl_ids)
            loss = loss_func(logit,label)
            total_loss += loss.item()
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            total_step += 1
            if step%500 == 0:
                logger.info("epoch:{},step:{},avg_loss:{}".format(epoch+1,step,total_loss/total_step))
            if step%35000 == 0:
                # logger.info("开始evaluation...")
                # avg_loss,acc = eval_classifier2(eval_dataloader,classifier,config,train=True,ret=True)
                # logger.info("evaluation: avg_{},acc:{},max_acc:{}".format(avg_loss,acc,max_acc))
                # if acc>max_acc:
                #     max_acc = acc
                torch.save(classifier.state_dict(),config.saved_path / "classifier_{}.pt".format(TRAIN_DATE))
                torch.save(optimizer.state_dict(),config.saved_path / "c_optimizer_{}.pt".format(TRAIN_DATE))


if __name__ == "__main__":
    config = Config()
    classifier = CasClassifier(mode='train')
    train_dataset = ClassifierDataset(config,'train')
    train_dataloader = DataLoader(train_dataset,config.train_batch_size)
    # eval_dataset = ClassifierDataset(config,'eval')
    # eval_dataloader = DataLoader(eval_dataset,config.eval_batch_size)
    train_classifier2(train_dataloader,classifier,config)