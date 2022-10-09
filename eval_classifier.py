from config_class import Config
from dataset import CodeSearchDataset,ClassifierDataset
from torch.utils.data import DataLoader
from model import SimpleCasClassifier
import torch
import torch.nn as nn
import os

def eval_classifier(dataloader,classifier,config,ret=False):
    if os.path.exists(config.saved_path+"/classifier3.pt"):
        classifier.load_state_dict(torch.load(config.saved_path+"/classifier3.pt"))
    classifier.zero_grad()
    if config.use_cuda:
        classifier = classifier.cuda()
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
            label = label.cuda()
        logits = classifier(pl_ids,nl_ids)
        pred = torch.argmax(logits,dim=1)
        print(logits)
        print(pred)
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

        
def test_classifer(dataloader,classifer,config):
    eval_classifier(dataloader,classifer,config)



if __name__ == "__main__":
    config = Config()
    dataset = ClassifierDataset(config,0,mode='eval')
    dataloader = DataLoader(dataset,batch_size=config.eval_batch_size)
    classifier = SimpleCasClassifier()
    avg_loss,acc = eval_classifier(dataloader,classifier,config,True)
    print(avg_loss,acc)