
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import argparse
from importlib import import_module
from module.dataloader import build_dataset,create_data_loader
from sklearn.model_selection import train_test_split
from Teacher.models.bert import TextCNN_Classifier,Config,tokenizer
from Student.models.TextCNN import Model
from Student.models.TextCNN import Config as StudentConfig
from module.train import train


parser = argparse.ArgumentParser(description="Long Text Classification")
parser.add_argument('--epoch',default=5,help='num of epoches')
parser.add_argument('--batch',default=4,help='num of batchsize')
args = parser.parse_args()

MAX_LEN = 256
BATCH_SIZE = args.batch
RANDOM_SEED = 2022

np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
torch.cuda.manual_seed(RANDOM_SEED)
torch.backends.cudnn.deterministic = True  #

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

config = Config()
config.RANDOM_SEED = RANDOM_SEED
config.file_path = "data/labeled_data.csv"
config.vocab_path = "Student/data/vocab.pkl"
config.padding_size =256
config.MODEL_DIR = "Teacher/saved_dict"
config.num_epoches = args.epoch
config.tokenizer = tokenizer
vocab,df_train,df_val,df_test = build_dataset(config)

train_data_loader = create_data_loader(df_train,vocab,config,BATCH_SIZE)
valid_data_loader = create_data_loader(df_val,vocab,config,BATCH_SIZE)
test_data_loader = create_data_loader(df_test,vocab,config,BATCH_SIZE)

# 加载Teacher模型
teacher_model = TextCNN_Classifier()
teacher_model = teacher_model.to(device)
teacher_model.load_state_dict(torch.load("Teacher/saved_dict/best_model_state.ckpt"))

print("Checkpoint",teacher_model)

# 加载Student模型
student_config = StudentConfig("Student","random")
student_config.n_vocab = len(vocab)
student_config.init_method = False
student_model = Model(student_config).to(device)

print("Checkpoint",student_model)
train(teacher_model,student_model,train_data_loader,valid_data_loader,test_data_loader,student_config)

