import torch
import torch.nn as nn
from module.logger import logger as logging
from bert_bilstm_crf.model import Config,Model,tokenizer
from bert_bilstm_crf.dataloader import NerDataset,create_data_loader
from transformers import AutoConfig,AutoModel,AutoTokenizer,AdamW,get_linear_schedule_with_warmup

MAX_LEN = 128
torch.cuda.empty_cache()
# 使用差分学习率
def get_parameters(model, model_init_lr, multiplier, classifier_lr):
    parameters = []
    lr = model_init_lr
    for layer in range(12,-1,-1):
        layer_params = {
            'params': [p for n,p in model.named_parameters() if f'encoder.layer.{layer}.' in n],
            'lr': lr
        }
        parameters.append(layer_params)
        lr *= multiplier
    classifier_params = {
        'params': [p for n,p in model.named_parameters() if 'layer_norm' in n or 'linear' in n
                   or 'pooling' in n],
        'lr': classifier_lr
    }
    parameters.append(classifier_params)
    return parameters



def train(x_train,y_train,x_valid,y_valid,word2id,tag2id,data_path):
    logging.info(f"the size of vocab is {len(word2id)}")
    config = Config()
    config.VOCAB_SIZE = len(word2id)
    config.TARGET_SIZE = len(tag2id)
    config.model_path = config.model_path + data_path.replace("/",'-')+'-bert_bilstm_crf.pth'
    print(config.model_path)
    logging.info("="*30+" start to operate dataset "+"="*30)

    train_loader = create_data_loader(x_train,y_train,tokenizer,tag2id,MAX_LEN,config.batch_size)
    valid_loader = create_data_loader(x_valid,y_valid,tokenizer,tag2id,MAX_LEN,config.batch_size)

    logging.info('='*30+" start to init model "+"="*30)
    model = Model(config).to(config.device)

    # parameters=get_parameters(model,2e-5,0.95, 1e-4)
    # optimizer = AdamW(parameters)
    optimizer = AdamW(model.parameters(),lr=2e-5,correct_bias=False)

    total_steps = len(train_loader)*config.EPOCHES
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=0,
        num_training_steps=total_steps
    )

    best_loss = 1e18
    logging.info("start to train")
    for e in range(config.EPOCHES):

        step = 0
        losses = 0.
        for i,unit in enumerate(train_loader):
            input_ids = unit["input_ids"].to(config.device)
            attention_mask = unit["attention_mask"].to(config.device)
            targets = unit["labels"].to(config.device)
            loss = model(input_ids,attention_mask,targets)
            losses+=loss
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            step+=1
            if step % config.print_step == 0:
                total_step = len(train_loader)+1
                logging.info("Epoch {}, step/total_step: {}/{} {:.2f}% Loss:{:.4f}".format(
                            e, step, total_step,
                            100. * step / total_step,
                            losses / config.print_step
                        ))
                losses = 0.
        cur_loss = evaluate(valid_loader,model,config)
        model.train()
        logging.info("Epoch {} evaluete valid Loss is :{:.4f}".format(
                                    e,
                                    cur_loss
                                ))
        if best_loss > cur_loss:
            best_loss = cur_loss
            torch.save(model.state_dict(),config.model_path)

def evaluate(dataloader,model,config):
    model.eval()
    losses = 0.
    for i,unit in enumerate(dataloader):
        input_ids = unit["input_ids"].to(config.device)
        attention_mask = unit["attention_mask"].to(config.device)
        targets = unit["labels"].to(config.device)
        loss = model(input_ids,attention_mask,targets)
        losses+=loss.detach().cpu().tolist()
    return losses/len(dataloader)

    
    
        





