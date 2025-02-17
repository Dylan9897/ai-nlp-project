import os
import torch
import argparse
import numpy as np
from module.data_process import read_file,make_vocab

# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

RANDOM_SEED = 2022

np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
torch.cuda.manual_seed(RANDOM_SEED)
torch.backends.cudnn.deterministic = True  #

parser = argparse.ArgumentParser(description="NER")
parser.add_argument("--data",default="data/MedicalNER",type=str,help="Provide a data path")
parser.add_argument("--model",default="bert",type=str,help="Use bert or Not")
args = parser.parse_args()

if __name__ == '__main__':
    model_name = args.model
    data_path = args.data

    x_train, y_train = read_file(os.path.join(data_path, "train.txt"))
    x_valid, y_valid = read_file(os.path.join(data_path, "dev.txt"))
    x_test, y_test = read_file(os.path.join(data_path, "test.txt"))
    word2id = make_vocab(os.path.join(data_path, "word2id.pkl"), x_train)
    tag2id = make_vocab(os.path.join(data_path, "tag2id.pkl"), y_train)

    print(f"========================== {data_path} ============================")
    print(f"Num x_train:  {len(x_train)}, Num y_train: {len(y_train)}, ")
    print(f"Num x_valid:  {len(x_valid)}, Num y_valid: {len(y_valid)}, ")
    print(f"Num x_test:  {len(x_test)}, Num y_test: {len(y_test)}, ")
    print(f"Num word2idx:  {len(word2id)}, Num tag2idx: {len(tag2id)}, ")

    if not model_name:
        from bilstm_crf.train import train
        train(
            x_train,
            y_train,
            x_valid,
            y_valid,
            word2id,
            tag2id,
            data_path
        )
    else:
        from bert_bilstm_crf.train import train
        train(
            x_train,
            y_train,
            x_valid,
            y_valid,
            word2id,
            tag2id,
            data_path
        )