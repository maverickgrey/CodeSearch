import json
import string
import re

class ExampleData:
    def __init__(self,code_tokens,nl_tokens,code,docstring,func_name,repo):
        self.code_tokens = code_tokens
        self.nl_tokens = nl_tokens
        self.code = code
        self.docstring = docstring
        self.func_name = func_name
        self.repo = repo

# 对查询进行过滤的过滤器
class DataFilterer:
    def __init__(self,data_path=None,data=None):
        if data is None:
            self.data_path = data_path
            self.data = self.load_data(self.data_path)
        else:
            self.data = data
        self.filter_table = ['()','=','/','{','}','(',')','*','-',',','[',']','#','"','\'','\\']

    def filter(self,data):
        result = []
        for i in range(len(data)):
            data[i].docstring = self.remove_html(data[i])
            data[i].docstring = self.truncate(data[i])
            if self.too_short(data[i]):
                continue
            elif self.has_chinese(data[i]):
                continue
            else:
                data[i].docstring = self.has_java(data[i])
                result.append(data[i])
        return result

    def load_data(self,data_path):
        data = []
        with open(data_path,'r') as f:
            for line in f.readlines():
                js = json.loads(line)
                data.append(ExampleData(js['code_tokens'],js['docstring_tokens'],js['code'],js['docstring'],js['func_name'],js['repo']))
        return data
    
    # 一些过滤规则，True则应该过滤，False则不应该过滤
    # 第一条过滤规则：nl_tokens长度过小的数据需要被过滤.注意，判断nl_tokens的长度是在去除标点符号后判断的
    def too_short(self,example):
        punc = string.punctuation
        nl_tokens = example.docstring.split(' ')
        length = 0
        for token in nl_tokens:
            if token not in punc:
                length += 1
        return True if length<=4 else False
    
    # 第二条过滤规则：将中文的NL过滤出去，因为它们在数据集中是以unicode的形式存储的，这在模型中无法提供有用的语义信息
    def has_chinese(self,example):
        nl = example.docstring
        for str in nl:
            if u'\u4e00' <= str <= u'\u9fff':
                return True
        return False
    
    # 第三条规则：只保留docstring的第一句话(以句号或者换行符作为句子结束标志),同时过滤一些黑名单中的符号
    def truncate(self,example):
        nl = ""
        for char in example.docstring:
            if (char != '.') and (char not in self.filter_table):
                if char == '\n':
                    nl += ' '
                else:
                    nl += char
            elif (char != '.') and (char in self.filter_table):
                continue
            else:
                break
        return nl
        
    # 第四条规则：将javadoc符号拿走
    def has_java(self,example):
        nl_tokens = example.docstring.split(' ')
        nl = ""
        for token in nl_tokens:
            if '@' not in token:
                nl += token
                nl += ' '
        return nl.strip()
    
    #第五条规则：将html标签拿走
    def remove_html(self,example):
        nl = example.docstring
        new_nl = re.sub("</?[\w]+ ?[\S]*>",'',nl)
        return new_nl
    
    #将过滤后的数据写入到指定文件
    def write_filtered_data(self,out_put):
        filtered_data = self.filter(self.data)
        self.write_to_file(filtered_data,out_put)
    
    #将数据写入文件
    def write_to_file(data,output_path,type='encoder'):
        with open(output_path,'w') as f:
            for example in data:
                js = {}
                if type == 'encoder':
                    js['repo'] = example.repo
                    js['func_name'] = example.func_name
                js['docstring'] = example.docstring
                js['docstring_tokens'] = example.nl_tokens
                js['code'] = example.code
                js['code_tokens'] = example.code_tokens
                if type == 'classifier':
                    js['label'] = example.label
                f.write(json.dumps(js)+"\n")