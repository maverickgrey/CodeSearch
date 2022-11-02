from torch.utils.data import Dataset,DataLoader
import json
import torch
from utils import convert_examples_to_features
from config_class import Config


# 训练encoder的数据集与一般使用时的数据集
class CodeSearchDataset(Dataset):
    def __init__(self,config,mode='train'):
        self.config = config
        self.mode = mode
        self.data = self.load_example(self.mode)
        

    def __getitem__(self,idx):
        #return (torch.tensor(self.data[idx].pl_ids),torch.tensor(self.data[idx].nl_ids))
        return (self.data[idx].pl_ids,self.data[idx].nl_ids)
    
    def __len__(self):
        return len(self.data)
    
    def collate_fn(self,batch):
        max_len = 0
        result = []
        pl_res = []
        nl_res = []
        for example in batch:
            nl_length = len(example[1])
            pl_length = len(example[0])
            if nl_length > max_len:
                max_len = nl_length
            if pl_length > max_len:
                max_len = pl_length
        for example in batch:
            pl_ids = example[0]
            nl_ids = example[1]
            nl_padding_length = max_len - len(nl_ids)
            nl_ids += nl_padding_length*[self.config.tokenizer.pad_token_id]
            pl_padding_length = max_len - len(pl_ids)
            pl_ids += pl_padding_length*[self.config.tokenizer.pad_token_id]
            pl_res.append(pl_ids)
            nl_res.append(nl_ids)
        result.append(torch.LongTensor(pl_res))
        result.append(torch.LongTensor(nl_res))
        return result

            

  # 从jsonl中读取所需要的数据，其中每一项数据都是InputFeatures类型，其中包含了每条数据的idx、url、nl信息和pl信息
    def load_example(self,mode):
        examples = []
        if mode == 'test':
            test_path = self.config.test_path
            with open(test_path,'r',encoding='utf-8') as f:
                num = 0
                for line in f.readlines():
                    num += 1
                    js = json.loads(line)
                    example = convert_examples_to_features(js,num,self.config,type=0)
                    examples.append(example)
                    # if num>10:
                    #     break
            return examples
        elif mode == 'eval':
            eval_path = self.config.data_path + "/filtered_data/e_valid.jsonl"
            with open(eval_path,'r',encoding='utf-8') as f:
                num = 0
                for line in f.readlines():
                    num += 1
                    js = json.loads(line)
                    example = convert_examples_to_features(js,num,self.config,type=0)
                    examples.append(example)
            return examples
        else:
            num = 0
            train_path = self.config.data_path +"/filtered_data"+"/java_train_sf.jsonl"
            with open(train_path,'r') as f:
                for line in f.readlines():
                    num+=1
                    js = json.loads(line)
                    example = convert_examples_to_features(js,num,self.config,type=0)
                    examples.append(example)
                    # if num>9999:
                    #   break
            return examples

# 专门用来训练CasClassifier的数据集
class ClassifierDataset(Dataset):
    def __init__(self,config,mode='train'):
        self.config = config
        self.mode = mode
        self.data = self.load_examples(self.mode)
    
    def load_examples(self,mode):
        examples = []
        if mode == 'eval':
            num = 0
            eval_path = self.config.test_path
            with open(eval_path,'r') as f:
                for line in f.readlines():
                    js = json.loads(line)
                    example = convert_examples_to_features(js,-1,self.config,2)
                    examples.append(example)
                    num += 1
                    # if num > 100:
                    #     break
            return examples
        elif mode == 'test':
            pass
        else:
            postfix = self.config.data_path+"/classifier/java_train_"
            train_path = postfix + str(self.train_no)+".jsonl"
            with open(train_path,'r') as f:
                for line in f.readlines():
                    js = json.loads(line)
                    example = convert_examples_to_features(js,-1,self.config,2)
                    examples.append(example)
            return examples

    def __getitem__(self, index):
        pl_ids = self.data[index].pl_ids
        nl_ids = self.data[index].nl_ids
        label = self.data[index].label
        return (torch.tensor(pl_ids),torch.tensor(nl_ids),torch.tensor(label))
    
    def __len__(self):
        return len(self.data)


# 专门用来训练SimpleCasClassifier的数据集,但是训练集全在一个文件中
class ClassifierDataset2(Dataset):
    def __init__(self,config,mode='train'):
        self.config = config
        self.mode = mode
        self.data = self.load_examples(self.mode)
    
    def load_examples(self,mode):
        if mode == 'eval':
            num = 0
            examples = []
            eval_path = self.config.data_path+"/classifier/c_valid_1.jsonl"
            with open(eval_path,'r') as f:
                for line in f.readlines():
                    js = json.loads(line)
                    example = convert_examples_to_features(js,-1,self.config,1)
                    examples.append(example)
                    num += 1
                    # if num > 100:
                    #     break
            return examples
        elif mode == 'test':
            num = 0
            examples = []
            eval_path = self.config.data_path+"/classifier/c_test_0.jsonl"
            with open(eval_path,'r') as f:
                for line in f.readlines():
                    js = json.loads(line)
                    example = convert_examples_to_features(js,-1,self.config,1)
                    examples.append(example)
                    num += 1
                    # if num > 100:
                    #     break
            return examples
        else:
            examples = []
            path = "./CodeSearchNet/classifier/java_train_c3.jsonl"
            with open(path,'r') as f:
                for line in f.readlines():
                    js = json.loads(line)
                    example = convert_examples_to_features(js,-1,self.config,1)
                    examples.append(example)
            return examples

    def __getitem__(self, index):
        input_ids = self.data[index].token_ids
        label = self.data[index].label
        return (torch.tensor(input_ids),torch.tensor(label))
    
    def __len__(self):
        return len(self.data)


if __name__ == "__main__":
    config = Config()
    dataset = CodeSearchDataset(config,'test')
    dataloader = DataLoader(dataset,batch_size=4,collate_fn=dataset.collate_fn)
    for _,example in enumerate(dataloader):
        print(example[0].shape)
        print(example[1].shape)
    
    # dataset = ClassifierDataset(config,'eval')
    # dataloader = DataLoader(dataset,batch_size = 4)
    # for _,example in enumerate(dataloader):
    #     print("pl_ids",example[0].shape)
