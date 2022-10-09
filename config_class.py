from transformers import RobertaModel,RobertaConfig,RobertaTokenizer

class Config:
    def __init__(self,
                epoches=4,
                data_path = "./CodeSearchNet",
                saved_path = "./model_saved",
                train_batch_size = 8,
                eval_batch_size = 8,
                use_cuda = True,
                max_seq_length=512,
                filter_K = 100
                ):
        self.epoches = epoches
        self.data_path = data_path
        self.saved_path = saved_path
        self.train_batch_size = train_batch_size
        self.eval_batch_size = eval_batch_size
        self.use_cuda = use_cuda
        self.max_seq_length = max_seq_length
        self.filter_K = filter_K
        self.tokenizer = RobertaTokenizer.from_pretrained("microsoft/codebert-base-mlm")
        self.test_path = self.data_path+"/java_test_0.jsonl"