import os
import torch
from module.logger import logger as logging
from module.utils import read_pkl_file
from bilstm_crf.model import Config,Model
from bilstm_crf.dataloader import NerDataset,NerLoader

from module.evaluating import Metrics


def evaluate(x_test,y_test,data_path):
    ################# load word2id and tag2id #################
    word2id = read_pkl_file(os.path.join(data_path,"word2id.pkl"))
    tag2id = read_pkl_file(os.path.join(data_path,"tag2id.pkl"))
    logging.info(f"loading word2id num is {len(word2id)}")
    logging.info(f"loading tag2id num is {len(tag2id)}")

    ################## loading model #########################
    config = Config()
    config.WORD_PAD_ID = word2id["<PAD>"]
    config.LABEL_O_ID = tag2id["O"]
    testset = NerDataset(x_test,y_test,word2id,tag2id)
    config.batch_size = 1
    loader = NerLoader(config)
    test_dataloader = loader._return_dataloader(testset)
    config.VOCAB_SIZE = len(word2id)
    config.TARGET_SIZE = len(tag2id)
    
    model = Model(config).to(config.device)
    model.load_state_dict(torch.load(config.model_path))
    ######################### testing ##########################
    id2tag = {v:k for k,v in tag2id.items()}
    true_label = []
    pred_label = []
    for i,(features,target,mask) in enumerate(test_dataloader):
        features = features.to(config.device)
        target = target.to(config.device).detach().cpu().tolist()[0]
        mask = mask.to(config.device)
        y_pred = model.predict(features,mask)[0]

        true_label.append([id2tag[unit] for unit in target])
        pred_label.append([id2tag[unit] for unit in y_pred])

    metrics = Metrics(true_label,pred_label)
    metrics.report_scores()







