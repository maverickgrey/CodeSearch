from torch.utils.data import Dataset
import json
import torch
from utils import convert_examples_to_features
import config_class


class CodeSearchDataset(Dataset):
    def __init__(self,tokenizer,mode='train'):
        self.mode = mode
        self.data = self.load_example(self.mode)
        self.tokenizer = tokenizer

    def __getitem__(self,idx):
        return (torch.tensor(self.data[idx].pl_ids),torch.tensor(self.data[idx].nl_ids))
    
    def __len__(self):
        return len(self.data)

  # 从jsonl中读取所需要的数据，其中每一项数据都是InputFeatures类型，其中包含了每条数据的idx、url、nl信息和pl信息
    def load_example(self,mode):
        examples = []
        if mode == 'test':
            test_path = config_class.data_path + "/java_test_0.jsonl"
            with open(test_path,'r',encoding='utf-8') as f:
                num = 0
                for line in f.readlines():
                    num += 1
                    js = json.loads(line)
                    example = convert_examples_to_features(js,self.tokenizer,num)
                    examples.append(example)
                    if num>10:
                        break
                return examples
        elif mode == 'eval':
            eval_path = config_class.data_path + "/java_valid_0.jsonl"
            with open(eval_path,'r',encoding='utf-8') as f:
                num = 0
                for line in f.readlines():
                    num += 1
                    js = json.loads(line)
                    example = convert_examples_to_features(js,self.tokenizer,num)
                    examples.append(example)
                    # if num>999:
                    #   break
            return examples
        else:
            num = 0
            train_path = config_class.data_path +"/java_train_"
            for i in range(1):
                postfix = str(i)+".jsonl"
                train = train_path + postfix
                with open(train,'r') as f:
                    for line in f.readlines():
                        num+=1
                        js = json.loads(line)
                        example = convert_examples_to_features(js,self.tokenizer,num)
                        examples.append(example)
                        # if num>9999:
                        #   break
            return examples