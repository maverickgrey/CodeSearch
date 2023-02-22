import torch
import json
from config_class import Config
from model import CasEncoder
import os
from utils import load_codebase,load_codebase_old

# 从run_time.txt文件中计算平均查询时间
def avg(path):
    with open(path,'r') as f:
        time = 0
        line_no = 0
        for line in f.readlines():
            if line[0] == '=':
                if line_no !=0:
                    print("{}条查询的平均查询时间为:{}".format(line_no,time/line_no))
                print(line)
                time=0
                line_no=0
            if line[0] == '本':
                line_no += 1
                line = line.split('：')
                single_time = float(line[1].strip().replace('s',''))
                time += single_time

# 为某个文件中的代码段生成相应的向量表示，并且存储在jsonl文件中
def code_to_vec(input_path,output_path,encoder,config):
  if config.use_cuda:
    encoder = encoder.cuda()
  with open(input_path,'r') as code_file:
    code_no = 0
    for line in code_file.readlines():
      js = json.loads(line)
      pl = ' '.join(js['code_tokens'])
      pl_tokens = config.tokenizer.tokenize(pl)
      pl_tokens = pl_tokens[:config.max_seq_length-2]
      pl_tokens = [config.tokenizer.cls_token] + pl_tokens +[config.tokenizer.sep_token]
      padding_length = config.max_seq_length-len(pl_tokens)
      pl_tokens += padding_length*[config.tokenizer.pad_token]
      pl_ids = torch.tensor([config.tokenizer.convert_tokens_to_ids(pl_tokens)])
      if config.use_cuda:
          pl_ids = pl_ids.cuda()
      pl_vec = torch.reshape(encoder(pl_ids,None),(768,)).cpu().tolist()
      with open(output_path,'a') as vec_file:
        vec = {"code_no":code_no,"code_vec":pl_vec}
        vec_file.write(json.dumps(vec)+"\n")
      code_no += 1

def cos_similarity(mat_a,mat_b):
    a_mode = torch.norm(mat_a,p=2,dim=1)
    b_mode = torch.norm(mat_b,p=2,dim=1)
    # a_norm = mat_a/a_mode
    # b_norm = mat_b/b_mode
    a_norm = mat_a/a_mode
    b_norm = mat_b
    for col in range(len(b_norm)):
        b_norm[col] /= b_mode[col]
    res = torch.matmul(a_norm,b_norm.T)
    print(res)


if __name__ == "__main__":
    config = Config()
    encoder = CasEncoder('one')
    if os.path.exists("./model_saved/encoder3.pt"):
        encoder.load_state_dict(torch.load("./model_saved/encoder3.pt"))
    input_path = "./CodeSearchNet/filtered_data/java_test_part.jsonl"
    output_path = "./CodeSearchNet/code_vec/java_test_part_vec.jsonl"
    # code_to_vec(input_path,output_path,encoder,config)
    codebase1 = load_codebase(input_path,output_path,config)
    codebase2 = load_codebase_old(input_path,config,encoder)

    print(codebase1.base_size)
    print(codebase2.base_size)
        