from torch.utils.data import Dataset,DataLoader
import json
import torch
from config_class import Config
import datastruct

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
                    example = self.convert_examples_to_features(js,num,self.config,type=0)
                    examples.append(example)
                    # if num>10:
                    #     break
            return examples
        elif mode == 'eval':
            eval_path = self.config.data_path + "/filtered_data/java_valid_new.jsonl"
            with open(eval_path,'r',encoding='utf-8') as f:
                num = 0
                for line in f.readlines():
                    num += 1
                    js = json.loads(line)
                    example = self.convert_examples_to_features(js,num,self.config,type=0)
                    examples.append(example)
            return examples
        else:
            num = 0
            train_path = self.config.data_path +"/filtered_data"+"/java_train_new.jsonl"
            with open(train_path,'r') as f:
                for line in f.readlines():
                    num+=1
                    js = json.loads(line)
                    example = self.convert_examples_to_features(js,num,self.config)
                    examples.append(example)
                    # if num>9999:
                    #   break
            return examples
    
    def convert_examples_to_features(js,no,config):
            nl = js['docstring']
            nl_tokens = config.tokenizer.tokenize(nl)
            nl_tokens = nl_tokens[:config.max_seq_length-2]
            nl_tokens = [config.tokenizer.cls_token]+nl_tokens+[config.tokenizer.sep_token]
            nl_ids = config.tokenizer.convert_tokens_to_ids(nl_tokens)
            # 现在nl、pl的padding都在dataloader中使用collate_fn函数进行
            # padding_length = config.max_seq_length - len(nl_ids)
            # nl_ids += [config.tokenizer.pad_token_id]*padding_length

            pl = js['code']
            pl_tokens = config.tokenizer.tokenize(pl)
            pl_tokens = pl_tokens[:config.max_seq_length-2]
            pl_tokens = [config.tokenizer.cls_token]+pl_tokens+[config.tokenizer.sep_token]
            pl_ids = config.tokenizer.convert_tokens_to_ids(pl_tokens)
            return datastruct.EncoderFeatures(nl_tokens,nl_ids,pl_tokens,pl_ids,no)

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
                    example = self.convert_examples_to_features(js,-1,self.config,2)
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
                    example = self.convert_examples_to_features(js,self.config)
                    examples.append(example)
            return examples

    def __getitem__(self, index):
        pl_ids = self.data[index].pl_ids
        nl_ids = self.data[index].nl_ids
        label = self.data[index].label
        return (torch.tensor(pl_ids),torch.tensor(nl_ids),torch.tensor(label))
    
    def __len__(self):
        return len(self.data)
    
    def convert_examples_to_features(js,config):
            nl = js['docstring']
            nl_tokens = config.tokenizer.tokenize(nl)
            nl_tokens = nl_tokens[:config.max_seq_length-2]
            nl_tokens = [config.tokenizer.cls_token]+nl_tokens+[config.tokenizer.sep_token]
            nl_ids = config.tokenizer.convert_tokens_to_ids(nl_tokens)
            padding_length = config.max_seq_length - len(nl_ids)
            nl_ids += [config.tokenizer.pad_token_id]*padding_length

            pl = js['code']
            pl_tokens = config.tokenizer.tokenize(pl)
            pl_tokens = pl_tokens[:config.max_seq_length-2]
            pl_tokens = [config.tokenizer.cls_token]+pl_tokens+[config.tokenizer.sep_token]
            pl_ids = config.tokenizer.convert_tokens_to_ids(pl_tokens)
            padding_length = config.max_seq_length - len(pl_ids)
            pl_ids += [config.tokenizer.pad_token_id]*padding_length
            label = js['label']
            return datastruct.CasClassifierFeatures(pl_tokens,pl_ids,nl_tokens,nl_ids,label)


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
                    example = self.convert_examples_to_features(js,-1,self.config,1)
                    examples.append(example)
                    num += 1
                    # if num > 100:
                    #     break
            return examples
        elif mode == 'test':
            num = 0
            examples = []
            eval_path = self.config.data_path+"/classifier/c_test_1.jsonl"
            with open(eval_path,'r') as f:
                for line in f.readlines():
                    js = json.loads(line)
                    example = self.convert_examples_to_features(js,-1,self.config,1)
                    examples.append(example)
                    num += 1
                    # if num > 100:
                    #     break
            return examples
        else:
            examples = []
            path = "./CodeSearchNet/classifier/java_train_classifier_p1n1.jsonl"
            with open(path,'r') as f:
                for line in f.readlines():
                    js = json.loads(line)
                    example = self.convert_examples_to_features(js,-1,self.config,1)
                    examples.append(example)
            return examples

    def __getitem__(self, index):
        input_ids = self.data[index].token_ids
        label = self.data[index].label
        return (torch.tensor(input_ids),torch.tensor(label))
    
    def __len__(self):
        return len(self.data)
    
    def convert_examples_to_features(js,config):
        nl = js['docstring']
        nl_tokens = config.tokenizer.tokenize(nl)
        pl = js['code']
        pl_tokens = config.tokenizer.tokenize(pl)
        input_tokens = [config.tokenizer.cls_token]+nl_tokens+[config.tokenizer.sep_token]
        input_tokens += pl_tokens
        input_tokens = input_tokens[:config.max_seq_length-1]
        input_tokens +=[config.tokenizer.sep_token]
        padding_length = config.max_seq_length - len(input_tokens)
        input_tokens += [config.tokenizer.pad_token]*padding_length
        input_ids = config.tokenizer.convert_tokens_to_ids(input_tokens)
        label = js['label']
        return datastruct.SimpleClassifierFeatures(input_tokens,input_ids,label)

class TripletTrainData(Dataset):
    def __init__(self,config):
        self.config = config
        self.data = self.load_examples()
    
    def load_examples(self):
        examples = []
        with open(self.config.data_path+"/filtered_data/java_encoder_triplet.jsonl") as f:
            for line in f.readlines():
                js = json.loads(line)
                example = self.convert_example_to_features(js=js)
                examples.append(example)
        return examples
    
    def convert_example_to_features(self,js):
        example = {}
        anchor = js['anchor']
        positive = js['positive']
        negative = js['negative']
        anchor_token = self.config.tokenizer.tokenize(anchor)
        anchor_token = anchor_token[:self.config.max_seq_length-2]
        anchor_token = [self.config.tokenizer.cls_token]+anchor_token+[self.config.tokenizer.sep_token]
        padding_len = self.config.max_seq_length-len(anchor_token)
        anchor_token += padding_len*[self.config.tokenizer.pad_token]
        anchor_ids = self.config.tokenizer.convert_tokens_to_ids(anchor_token)

        positive_token = self.config.tokenizer.tokenize(positive)
        positive_token = positive_token[:self.config.max_seq_length-2]
        positive_token = [self.config.tokenizer.cls_token]+positive_token+[self.config.tokenizer.sep_token]
        padding_len = self.config.max_seq_length-len(positive_token)
        positive_token += padding_len*[self.config.tokenizer.pad_token]
        positive_ids = self.config.tokenizer.convert_tokens_to_ids(positive_token)

        negative_token = self.config.tokenizer.tokenize(negative)
        negative_token = negative_token[:self.config.max_seq_length-2]
        negative_token = [self.config.tokenizer.cls_token]+negative_token+[self.config.tokenizer.sep_token]
        padding_len = self.config.max_seq_length-len(negative_token)
        negative_token += padding_len*[self.config.tokenizer.pad_token]
        negative_ids = self.config.tokenizer.convert_tokens_to_ids(negative_token)

        example['anchor'] = anchor_ids
        example['positive'] = positive_ids
        example['negative'] = negative_ids
        return example
    
    def __getitem__(self, index):
        _data = self.data[index]
        anchor = _data['anchor']
        positive = _data['positive']
        negative = _data['negative']
        return (torch.tensor(anchor),torch.tensor(positive),torch.tensor(negative))
    
    def __len__(self):
        return len(self.data)
    
    def collate_fn(self,batch):
        pass
        #TODO implement


if __name__ == "__main__":
    config = Config()
    dataset = TripletTrainData(config)
    dataloader = DataLoader(dataset,batch_size=4)
    for _,example in enumerate(dataloader):
        print(example)
    
    # dataset = ClassifierDataset(config,'eval')
    # dataloader = DataLoader(dataset,batch_size = 4)
    # for _,example in enumerate(dataloader):
    #     print("pl_ids",example[0].shape)
