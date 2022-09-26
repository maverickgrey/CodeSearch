from config_class import Config
import os
import torch
import torch.nn as nn

config = Config()
def train_classifier(dataloader,classifier):
    if not os.path.exists(config.saved_path):
        os.makedirs(config.saved_path)

    classifier.zero_grad()
    loss_func = torch.nn.CrossEntorpyLoss()
    optimizer = torch.optim.AdamW(classifier.parameters(),lr=1e-5)

    if os.path.exists(config.model_save_path+"/classifier.pt"):
        classifier.load_state_dict(torch.load(config.model_save_path+"/classifier.pt"))
    if os.path.exists(config.model_save_path+"/c_optimizer.pt"):
        optimizer.load_state_dict(torch.load(config.model_save_path+"/c_optimizer.pt"))
    if config.use_cuda:
        classifier = classifier.cuda

    total_loss = 0
    total_step = 0
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
        torch.save(classifier.state_dict(),config.saved_path+"/classifier.pt")
        torch.save(optimizer.state_dict(),config.saved_path+"/c_optimmizer.pt")



