from transformers import AutoModelForSeq2SeqLM,AutoTokenizer,AutoModel
import jieba
import json
import torch

MODEL_NAME = "Helsinki-NLP/opus-mt-zh-en"
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model2 = AutoModel.from_pretrained(MODEL_NAME)

#将中文查询中的缩写专用术语根据词表拓展成为全称
def expand_zh_query(query,table):
    tokenized_query = " ".join(jieba.cut(query)).split(' ')
    for i in range(len(tokenized_query)):
        if tokenized_query[i] in table:
            tokenized_query[i] = table.get(tokenized_query[i])
    return ''.join(tokenized_query)

def read_abbr(path):
    with open(path,'r') as f:
        table = json.load(f)
    return table

def translate(zh_text,model,tokenizer):
    table = read_abbr("./abbreviation.json")
    text = expand_zh_query(zh_text,table)
    text_tokens = tokenizer([text],return_tensors='pt')
    translation = model.generate(**text_tokens)
    translated_text = tokenizer.batch_decode(translation,skip_special_token=True)
    return translation



if __name__ == "__main__":
    zh_text = "如何实现快速排序"
    tokens = tokenizer.tokenize(zh_text)
    token_ids = torch.tensor([tokenizer.convert_tokens_to_ids(tokens)])
    res = model2(input_ids =token_ids)
    print(res)


