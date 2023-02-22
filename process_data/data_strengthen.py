import os
import json
import random


"""
对分类器进行数据增强的文件，正负样本的增强方式如下：
negative: In-Batch Augmentation————————在一个批次内随机自然语言查询和代码段构成负样本
positive: Rewritter Augmentation————————将查询重写，主要有以下三种方式：
        （1）随机删除一个词
        （2）随机替换两个词的位置
        （3）随机拷贝一个词
        （其中论文显示交换两个词的位置能够相对更好地提升性能）
"""

# 统计文件的行数(jsonl中即可统计样本数)
def simple_cnt(filename):
    lines = 0 
    for _ in open(filename):
        lines += 1
    return lines

# 从文件中读取一组batch大小为batch_size的encoder数据
def read_batch(sample_list,batch_size,offset):
    batch = []
    try:
        for line_no in range(offset,offset+batch_size):
            example = sample_list[line_no]
            example_js = json.loads(example)
            batch.append(example_js)
    except:
        return batch
    return batch

# 对batch内的数据进行in-batch augmentation同时构建正负样本
# 一条example随机选取2条别的样本构建负样本,并自己生成一条正样本
def batch_augmentation(batch):
    result = []
    for i in range(len(batch)):
        current_nl = batch[i].get('docstring')
        current_nl_tokens = batch[i].get('docstring_tokens')
        current_pl = batch[i].get('code')
        current_pl_tokens = batch[i].get('code_tokens')
        positive = {"docstring":current_nl,"docstring_tokens":current_nl_tokens,"code":current_pl,"code_tokens":current_pl_tokens,"label":1}

       # 目前生成两个负样本 
        new_index1 = random.randint(0,len(batch)-1)
        while(new_index1 == i):
            new_index1 = random.randint(0,len(batch)-1)
        another_pl = batch[new_index1].get('code')
        another_pl_tokens = batch[new_index1].get('code_tokens')
        negative = {"docstring":current_nl,"docstring_tokens":current_nl_tokens,"code":another_pl,"code_tokens":another_pl_tokens,"label":0}
        
        result.append(negative)

        new_index2 = random.randint(0,len(batch)-1)
        while (new_index2 == i):
            new_index2 = random.randint(0,len(batch)-1)
        another_pl = batch[new_index2].get('code')
        another_pl_tokens = batch[new_index2].get('code_tokens')
        negative = {"docstring":current_nl,"docstring_tokens":current_nl_tokens,"code":another_pl,"code_tokens":another_pl_tokens,"label":0}
        
        result.append(positive)
        result.append(negative)
    return result

# 对batch内的数据进行rewritten-augmentation
# 目前只对查询(即只针对docstring字段)的随机两个词进行随机调换而不进行拷贝、删除等增强
def rewritten_augmentation(batch):
    result = batch
    for i in range(len(batch)):
        if batch[i].get("label") == 1:
            current_nl = batch[i].get('docstring').split(" ")
            pos1 = random.randint(0,len(current_nl)-1)
            pos2 = random.randint(0,len(current_nl)-1)
            # while (pos2==pos1):
            #     pos2 = random.randint(0,len(current_nl)-1)
            current_nl[pos1],current_nl[pos2] = current_nl[pos2],current_nl[pos1]
            positive = {"docstring":" ".join(current_nl),
                        "docstring_tokens":batch[i].get('docstring_tokens'),
                        "code":batch[i].get('code'),
                        "code_tokens":batch[i].get('code_tokens'),
                        "label":1}
            result.append(positive)
    return result

def write_batch_to_file(batch,output_file):
    for i in range(len(batch)):
        output_file.write(json.dumps(batch[i])+"\n")

def generate_file(input_file,output_file,batch_size=10):
    file_cnt = simple_cnt(input_file)
    f_input = open(input_file,'r')
    sample_list = f_input.readlines()
    offset = 0
    out = open(output_file,'a')
    while offset < file_cnt:
        batch = read_batch(sample_list,batch_size,offset)
        batch = batch_augmentation(batch)
        batch = rewritten_augmentation(batch)
        write_batch_to_file(batch,out)
        offset += batch_size
    out.close()

    # batch = read_batch(input_file,batch_size,offset=)

if __name__ == "__main__":
    input_file = "../CodeSearchNet/filtered_data/java_train_new.jsonl"
    output_file = "../CodeSearchNet/classifier/java_classifier_0220.jsonl"
    generate_file(input_file,output_file)
        
