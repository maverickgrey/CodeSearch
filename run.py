from dataset import CodeSearchDataset
from torch.utils.data import DataLoader
from config_class import Config
from model import CasEncoder,CasClassifier,SimpleCasClassifier
from utils import cos_similarity,get_priliminary,rerank
from eval_encoder import eval_encoder
import numpy as np
import torch


config = Config()
def run():
    dataset = CodeSearchDataset(config.test_path)
    dataloader = DataLoader(dataset,batch_size=8,shuffle=True)  
    encoder = CasEncoder()
    classifier = CasClassifier()

    if config.use_cuda == True:
        encoder = encoder.cuda()
        classifier = classifier.cuda()

    nl_vecs = []
    code_vecs = []
    for step,example in enumerate(dataloader):
        pl_ids = example[0]
        nl_ids = example[1]
        if config.use_cuda:
            pl_ids = pl_ids.cuda()
            nl_ids = nl_ids.cuda()
        out2,out3 = encoder(pl_ids,nl_ids)
        nl_vecs.append(out2.cpu().detach().numpy())
        code_vecs.append(out3.cpu().detach().numpy())

    nl_vecs = np.concatenate(nl_vecs,0)
    code_vecs = np.concatenate(code_vecs,0)
    scores = cos_similarity(nl_vecs,code_vecs)
    # np.argsort的功能是给一个数组排序，返回排序后的数字对应原来数字所在位置的下标
    # 默认升序排序，这里添加符号即可实现降序
    sort_ids = np.argsort(-scores,axis=-1,kind='quicksort',order=None)
    print(scores)
    print(sort_ids)
    pre,examples = get_priliminary(scores,dataset)
    res = rerank(pre,examples,classifier)
    print(res)

def test2_func():
    nl_text = "This is a test"
    pl_text = "def func ( x , y ) : return x + y "
    nl_tokens = config.tokenizer.tokenize(nl_text)
    nl_tokens = nl_tokens[:config.max_seq_length-2]
    nl_tokens = [config.tokenizer.cls_token]+nl_tokens+[config.tokenizer.sep_token]

    pl_tokens = config.tokenizer.tokenize(pl_text)
    pl_tokens = pl_tokens[:config.max_seq_length-2]
    pl_tokens = [config.tokenizer.cls_token]+pl_tokens+[config.tokenizer.sep_token]

    nl_ids = config.tokenizer.convert_tokens_to_ids(nl_tokens)
    pl_ids = config.tokenizer.convert_tokens_to_ids(pl_tokens)

    padding_nl_len = config.max_seq_length - len(nl_ids)
    padding_pl_len = config.max_seq_length - len(pl_ids)

    nl_ids += [config.tokenizer.pad_token_id]*padding_nl_len
    pl_ids += [config.tokenizer.pad_token_id]*padding_pl_len

    nl_ids = torch.tensor([nl_ids])
    pl_ids = torch.tensor([pl_ids])
    model1 = SimpleCasClassifier()
    res = model1(pl_ids,nl_ids)
    model2 = CasClassifier()
    res2 = model2(pl_ids,nl_ids)
    print(res)
    print(res2)



if __name__ == "__main__":
    encoder = CasEncoder()
    #train_dataset = CodeSearchDataset('train')
    eval_dataset = CodeSearchDataset('eval')
    #train_dataloader = DataLoader(train_dataset,batch_size=4)
    eval_dataloader = DataLoader(eval_dataset,batch_size=16)
    #train_encoder(train_dataloader,eval_dataloader,encoder)
    eval_encoder(eval_dataloader,encoder)
    #test_encoder(eval_dataloader,encoder,False,False)