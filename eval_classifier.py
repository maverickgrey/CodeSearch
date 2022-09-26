from distutils.command.config import config
from config_class import Config
import torch
import torch.nn as nn
import os

config = Config()
def eval_classifier(dataloader,classifier):
    if os.path.exists(config.saved_path+"/classifier.pt"):
        classifier.load_state_dict(torch.load(config.saved_path+"/classifier.pt"))
    total_loss = 0
    total_step = 0
    correct = 0
    preds = []
    labels = []
    classifier.eval()
    loss_func = nn.CrossEntropyLoss()
    for step,example in enumerate(dataloader):
        pl_ids = example[0]
        nl_ids = example[1]
        label = example[2]
        if config.use_cuda:
            pl_ids = pl_ids.cuda()
            nl_ids = nl_ids.cuda()
            labels = labels.cuda()
        logits = classifier(pl_ids,nl_ids)
        pred = torch.softmax(logits,dim=-1)
        loss = loss_func(pred,label)
        total_loss += loss.item()
        total_step += 1
        if config.use_cuda:
            pred = pred.cpu().tolist()
            label = label.cpu().tolist()
        else:
            pred = pred.tolist()
            label = label.tolist()
        labels.extend(label)
        preds.extend(pred)
    for i in range(len(labels)):
        if preds[i] == labels[i]:
            correct += 1
    acc = correct/len(labels)
    print("avg_loss:{},acc:{}".format(total_loss/total_step,acc))

        
def test_classifer(dataloader,classifer):
    eval_classifier(dataloader,classifer)