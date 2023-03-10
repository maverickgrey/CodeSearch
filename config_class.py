from transformers import RobertaTokenizer
from pathlib import Path

# 各种路径定义
ROOT_DIR = Path.cwd()
DATA_PATH = Path.joinpath(ROOT_DIR,"CodeSearchNet")
MODEL_PATH = Path.joinpath(ROOT_DIR,"model_saved")
TEST_PATH = Path.joinpath(DATA_PATH,"filtered_data","java_test_new.jsonl")
CODE_VEC_PATH = Path.joinpath(DATA_PATH,"code_vec","java_test_new_vec.jsonl")
LOGS_DIR = Path.joinpath(ROOT_DIR,"logs")

class Config:
    def __init__(self,
                encoder_epoches=4,
                classifier_epoches=3,
                data_path = DATA_PATH,
                saved_path = MODEL_PATH,
                train_batch_size = 8,
                eval_batch_size = 2,
                use_cuda = True,
                max_seq_length=512,
                filter_K = 30,
                final_K = 5,
                run_way = 'truth',
                confidence = 0.5,
                encoder_loss = "triplet",
                distance_type = "cosine",
                log_path = LOGS_DIR,
                device = 'cuda:0'
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
        self.test_path = TEST_PATH
        self.code_vec_path = CODE_VEC_PATH
        self.run_way = run_way
        self.confidence = confidence
        self.encoder_loss = encoder_loss
        self.distance_type = distance_type
        self.log_path = log_path
        self.device = device
