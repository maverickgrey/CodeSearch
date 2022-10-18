from torch.utils.data import Dataset
import json
import torch
from utils import convert_examples_to_features


# 训练encoder的数据集与一般使用时的数据集
class CodeSearchDataset(Dataset):
    def __init__(self,config,mode='train'):
        self.config = config
        self.mode = mode
        self.data = self.load_example(self.mode)
        

    def __getitem__(self,idx):
        return (torch.tensor(self.data[idx].pl_ids),torch.tensor(self.data[idx].nl_ids))
    
    def __len__(self):
        return len(self.data)

  # 从jsonl中读取所需要的数据，其中每一项数据都是InputFeatures类型，其中包含了每条数据的idx、url、nl信息和pl信息
    def load_example(self,mode):
        examples = []
        if mode == 'test':
            test_path = self.config.data_path +"/origin_data"+"/java_test_0.jsonl"
            with open(test_path,'r',encoding='utf-8') as f:
                num = 0
                for line in f.readlines():
                    num += 1
                    js = json.loads(line)
                    example = convert_examples_to_features(js,num,self.config,num)
                    examples.append(example)
                    if num>10:
                        break
            return examples
        elif mode == 'eval':
            eval_path = self.config.data_path + "/origin_data"+"/java_valid_0.jsonl"
            with open(eval_path,'r',encoding='utf-8') as f:
                num = 0
                for line in f.readlines():
                    num += 1
                    js = json.loads(line)
                    example = convert_examples_to_features(js,num,self.config,classifier=False)
                    examples.append(example)
            return examples
        else:
            num = 0
            train_path = self.config.data_path +"/filtered_data"+"/java_train.jsonl"
            with open(train_path,'r') as f:
                for line in f.readlines():
                    num+=1
                    js = json.loads(line)
                    example = convert_examples_to_features(js,num,self.config,classifier=False)
                    examples.append(example)
                    # if num>9999:
                    #   break
            return examples

# 专门用来训练Classifier的数据集，训练集在16个文件中，需要指定训练哪个数据集
class ClassifierDataset(Dataset):
    def __init__(self,config,train_no,mode='train'):
        self.config = config
        self.mode = mode
        self.train_no = train_no
        self.data = self.load_examples(self.mode)
    
    def load_examples(self,mode):
        examples = []
        if mode == 'eval':
            num = 0
            eval_path = self.config.data_path+"/classifier/cvalid_0.jsonl"
            with open(eval_path,'r') as f:
                for line in f.readlines():
                    js = json.loads(line)
                    example = convert_examples_to_features(js,-1,self.config,True)
                    examples.append(example)
                    num += 1
                    # if num > 100:
                    #     break
            return examples
        elif mode == 'test':
            pass
        else:
            postfix = self.config.data_path+"/classifier/ctrain_"
            train_path = postfix + str(self.train_no)+".jsonl"
            with open(train_path,'r') as f:
                for line in f.readlines():
                    js = json.loads(line)
                    example = convert_examples_to_features(js,-1,self.config,True)
                    examples.append(example)
            return examples

    def __getitem__(self, index):
        inputs_ids = self.data[index].token_ids
        label = self.data[index].label
        return (torch.tensor(inputs_ids),torch.tensor(label))
    
    def __len__(self):
        return len(self.data)


# 专门用来训练Classifier的数据集,但是训练集全在一个文件中
class ClassifierDataset2(Dataset):
    def __init__(self,config,mode='train'):
        self.config = config
        self.mode = mode
        self.data = self.load_examples(self.mode)
    
    def load_examples(self,mode):
        if mode == 'eval':
            num = 0
            eval_path = self.config.data_path+"/classifier/cvalid_0.jsonl"
            with open(eval_path,'r') as f:
                for line in f.readlines():
                    js = json.loads(line)
                    example = convert_examples_to_features(js,-1,self.config,True)
                    examples.append(example)
                    num += 1
                    # if num > 100:
                    #     break
            return examples
        elif mode == 'test':
            pass
        else:
            examples = []
            path = "./CodeSearchNet/classifier/java_train_c.jsonl"
            with open(path,'r') as f:
                for line in f.readlines():
                    js = json.loads(line)
                    example = convert_examples_to_features(js,-1,self.config,True)
                    examples.append(example)
            return examples

    def __getitem__(self, index):
        input_ids = self.data[index].token_ids
        label = self.data[index].label
        return (torch.tensor(input_ids),torch.tensor(label))
    
    def __len__(self):
        return len(self.data)