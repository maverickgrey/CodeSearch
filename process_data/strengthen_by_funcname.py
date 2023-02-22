from datafilter import DataFilterer
from process_data import read_data,write_to_file,build_examples
from process_data import TrainData
import os
import json

"""
该文件使用CodeSearchNet中的funcname和所在库的类名来对查询进行一定程度的增强
"""

# 将训练集数据按照函数名分类，每个类别之间空一行
def category_by_funcname(datapath,out_path):
    out_file = open(out_path,'w')
    current_name_prefix = None
    with open(datapath,'r') as f:
        line0 = f.readline()
        js0 = json.loads(line0)
        func_name0 = js0['func_name']
        current_name_prefix = func_name0.split('.')[0]
        out_file.write(line0)
        for line in f.readlines():
            js = json.loads(line)
            func_name = js['func_name']
            func_name = func_name.split('.')
            name_prefix = func_name[0]
            if name_prefix == current_name_prefix:
                out_file.write(line)
            else:
                out_file.write("\n"+line)
                current_name_prefix = name_prefix
    out_file.close()

# 用分类好的数据构建分类数据集
def build_by_categorized(data_path,out_path):
    out = open(out_path,'w')
    result = []
    with open(data_path,'r') as f:
        data = []
        unique_prefix_data = []
        for line in f.readlines():
            if line != "\n":
                js = json.loads(line)
                data.append(TrainData(js['code_tokens'],js['docstring_tokens'],js['code'],js['docstring'],js['func_name'],js['repo']))
            else:
                if len(data) == 1:
                    unique_prefix_data.extend(data)
                    data = []
                else:
                    result.extend(build_examples(data,shuffle=True))
                    data=[]
        result.extend(build_examples(unique_prefix_data,shuffle=True))
    for r in result:
        js = {}
        js['docstring'] = r.docstring
        js['docstring_tokens'] = r.nl_tokens
        js['code'] = r.code
        js['code_tokens'] = r.code_tokens
        js['label'] = r.label
        out.write(json.dumps(js)+"\n")
    out.close()

# 用func_name和repo来增强查询
def strengthen_query(data):
    for example in data:
        nl_tokens = example.repo.split('/'[-1])
        nl_tokens.extend(example.func_name.split('.'))
        nl_tokens.extend(example.nl_tokens)
        example.nl_tokens = nl_tokens


# 处理数据的pipeline:读入一个原始数据文件，分别输出一个encoder的文件以及一个classifier的文件
# 处理流程包括了查询增强、过滤数据、按照func_name分类、构建classifier数据、将数据写入到输出文件等
def pipeline(data_path,encoder_out_path,classifier_out_path,filter = True):
    # 读入数据
    data = read_data(data_path)
    #查询增强
    #strengthen_query(data)

    #过滤数据并写入文件
    if filter == True:
        df = DataFilterer(data_path=None,data=data)
        data = df.filter(df.data)
    write_to_file(data,encoder_out_path,'encoder')

    #按照func_name分类
    temp_file = "./CodeSearchNet/classifier/temp.jsonl"
    category_by_funcname(encoder_out_path,temp_file)
    #构建classifier数据
    build_by_categorized(temp_file,classifier_out_path)
    os.remove(temp_file)
