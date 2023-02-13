from config_class import Config
from model import SimpleCasClassifier,CasEncoder
from run import load_codebase,query_to_vec
from utils import cos_similarity,get_priliminary,rerank,get_info,code_to_vec

def test2_func():
    config = Config()
    classifier = SimpleCasClassifier()
    encoder_pl = CasEncoder('code')
    encoder_nl = CasEncoder('nl')
    if config.use_cuda:
        encoder_pl=encoder_pl.cuda()
        encoder_nl=encoder_nl.cuda()
        classifier = classifier.cuda()
    code_base = load_codebase(config.test_path,config,encoder_pl)
    query = "Concatenates a variable number of ObservableSource sources"
    query_tokens = query.split(' ')
    query_vec = query_to_vec(query,config,encoder_nl).cpu()
    scores = cos_similarity(query_vec,code_base.code_vecs)
    scores = scores.detach().numpy()
    print(scores)
    pre = get_priliminary(scores,code_base,10)
    for _pre in pre:
        final = rerank(query_tokens,_pre,classifier,config)
    get_info(final)

if __name__ == "__main__":
    input_path = "./CodeSearchNet/filtered_data/java_test_new.jsonl"
    output_path = "./CodeSearchNet/code_vec/java_test_new_vec.jsonl"
    encoder = CasEncoder('one')
    config = Config()
    code_to_vec(input_path,output_path,encoder,config)