from torch.utils.data import Dataset,DataLoader
import json
import torch
from config_class import Config,DATA_PATH
from hash_model import HashEncoder
from pathlib import Path

"""
用于定义代码搜索采用hash加速的offline阶段的一些数据集
"""

# 用来训练hash模型的数据集，它会返回nl和pl的id
class Alignment(Dataset):
    def __init__(self,mode):
        self.config = Config()
        self.mode = mode
        self.encoder = HashEncoder('all')
        self.data = self.load_data(self.mode)

    def __getitem__(self, index):
        nl_ids = self.data[index].get("nl_ids")
        pl_ids = self.data[index].get("pl_ids")
        return (torch.tensor(nl_ids),torch.tensor(pl_ids))
    
    def __len__(self):
        return len(self.data)

    def load_data(self,mode):
        examples = []
        if mode == 'train':
            train_path = Path.joinpath(DATA_PATH,"filtered_data","java_train_new.jsonl")
            with open(train_path,'r') as f:
                for line in f.readlines():
                    js = json.loads(line)
                    example = self.convert_examples_to_features(js,self.config)
                    examples.append(example)
        else:
            eval_path = Path.joinpath(DATA_PATH,"filtered_data","java_valid_new.jsonl")
            with open(eval_path,'r') as f:
                for line in f.readlines():
                    js = json.loads(line)
                    example = self.convert_examples_to_features(js,self.config)
                    examples.append(example)
        return examples
    
    def convert_examples_to_features(self,js,config):
        nl = js['docstring']
        nl_tokens = config.tokenizer.tokenize(nl)
        nl_tokens = nl_tokens[:config.max_seq_length-2]
        nl_tokens = [config.tokenizer.cls_token]+nl_tokens+[config.tokenizer.sep_token]
        nl_ids = config.tokenizer.convert_tokens_to_ids(nl_tokens)
        padding_length = config.max_seq_length - len(nl_ids)
        nl_ids += [config.tokenizer.pad_token_id]*padding_length
        # 现在nl、pl的padding都在dataloader中使用collate_fn函数进行
        # padding_length = config.max_seq_length - len(nl_ids)
        # nl_ids += [config.tokenizer.pad_token_id]*padding_length
        pl = js['code']
        pl_tokens = config.tokenizer.tokenize(pl)
        pl_tokens = pl_tokens[:config.max_seq_length-2]
        pl_tokens = [config.tokenizer.cls_token]+pl_tokens+[config.tokenizer.sep_token]
        pl_ids = config.tokenizer.convert_tokens_to_ids(pl_tokens)
        padding_length = config.max_seq_length - len(pl_ids)
        pl_ids += [config.tokenizer.pad_token_id]*padding_length
        return {"nl_ids":nl_ids,"pl_ids":pl_ids}


if __name__ == "__main__":
    # dataset = Alignment('train')
    # dataloader = DataLoader(dataset,4,False)
    # model = HashEncoder('all')
    # for step,examples in enumerate(dataloader):
    #     nl_ids = examples[0]
    #     pl_ids = examples[1]
    #     nl_vecs,nl_hash = model(nl_ids)
    #     pl_vecs,pl_hash = model(pl_ids)
    #     print("nl_vecs:",nl_vecs.shape)
    #     print("pl_vecs:",pl_vecs.shape)
    pass