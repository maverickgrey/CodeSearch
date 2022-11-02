from config_class import Config
from dataset import ClassifierDataset2, CodeSearchDataset,ClassifierDataset
from torch.utils.data import DataLoader
from model import SimpleCasClassifier
import torch
import torch.nn as nn
import os

def eval_classifier(dataloader,classifier,config,train=False,ret=False):
    if os.path.exists(config.saved_path+"/classifier3.pt") and train==False:
        classifier.load_state_dict(torch.load(config.saved_path+"/classifier3.pt"))
    classifier.zero_grad()
    total_loss = 0
    total_step = 0
    correct = 0
    preds = []
    labels = []
    classifier.eval()
    loss_func = nn.CrossEntropyLoss()
    for step,example in enumerate(dataloader):
        inputs = example[0]
        label = example[1]
        if config.use_cuda:
            inputs = inputs.cuda()
            label = label.cuda()
        logits = classifier(inputs)
        pred = torch.argmax(logits,dim=1)
        loss = loss_func(logits,label)
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

    print("eval_result: avg_loss:{},acc:{}".format(total_loss/total_step,acc))
    if ret:
        return (total_loss/total_step,acc)


def eval_classifier2(dataloader,classifier,config,train=False,ret=False):
    if os.path.exists(config.saved_path+"/classifier2.pt") and train==False:
        classifier.load_state_dict(torch.load(config.saved_path+"/classifier2.pt"))
    classifier.zero_grad()
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
        label = example[1]
        if config.use_cuda:
            pl_ids = pl_ids.cuda()
            nl_ids = nl_ids.cuda()
            label = label.cuda()
        logits = classifier(pl_ids,nl_ids)
        pred = logits.tolist()
        for p in pred:
            if p[0]>=config.confidence:
                p = 1
            else:
                p=0
        loss = loss_func(logits,label)
        total_loss += loss.item()
        total_step += 1
        if config.use_cuda:
            label = label.cpu().tolist()
        else:
            label = label.tolist()
        labels.extend(label)
        preds.extend(pred)
    for i in range(len(labels)):
        if preds[i] == labels[i]:
            correct += 1
    acc = correct/len(labels)

    print("eval_result: avg_loss:{},acc:{}".format(total_loss/total_step,acc))
    if ret:
        return (total_loss/total_step,acc)
        
def test_classifer(dataloader,classifer,config):
    eval_classifier(dataloader,classifer,config)



if __name__ == "__main__":
    config = Config()
    dataset = ClassifierDataset2(config,mode='test')
    dataloader = DataLoader(dataset,batch_size=config.eval_batch_size)
    classifier = SimpleCasClassifier()
    if config.use_cuda:
        classifier = classifier.cuda()
    avg_loss,acc = eval_classifier(dataloader,classifier,config,False,True)
    print(avg_loss,acc)