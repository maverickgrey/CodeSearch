from transformers import RobertaModel,RobertaConfig,RobertaTokenizer

class Config:
    def __init__(self,
                encoder_epoches=4,
                classifier_epoches=3,
                data_path = "./CodeSearchNet",
                saved_path = "./model_saved",
                train_batch_size = 4,
                eval_batch_size = 16,
                use_cuda = False,
                max_seq_length=512,
                filter_K = 5,
                final_K = 15,
                run_way = 'truth',
                confidence = 0.5,
                encoder_loss = "triplet",
                distance_type = "cosine"
                ):
        self.encoder_epoches = encoder_epoches
        self.classifier_epoches=classifier_epoches
        self.data_path = data_path
        self.saved_path = saved_path
        self.train_batch_size = train_batch_size
        self.eval_batch_size = eval_batch_size
        self.use_cuda = use_cuda
        self.max_seq_length = max_seq_length
        self.filter_K = filter_K
        self.final_K = final_K
        self.tokenizer = RobertaTokenizer.from_pretrained("microsoft/codebert-base-mlm")
        self.test_path = self.data_path+"/filtered_data/java_test_new.jsonl"
        self.code_vec_path = self.data_path+"/code_vec/java_test_new_vec.jsonl"
        self.run_way = run_way
        self.confidence = confidence
        self.encoder_loss = encoder_loss
        self.distance_type = distance_type