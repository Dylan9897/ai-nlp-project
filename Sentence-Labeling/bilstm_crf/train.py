import torch
from module.logger import logger as logging
from bilstm_crf.model import Config,Model
from bilstm_crf.dataloader import NerDataset,NerLoader


def train(x_train,y_train,x_valid,y_valid,word2id,tag2id,data_path):
    logging.info(f"the size of vocab is {len(word2id)}")
    config = Config()
    config.WORD_PAD_ID = word2id["<PAD>"]
    config.LABEL_O_ID = tag2id["O"]
    config.model_path = config.model_path + data_path.replace("/",'-')+'-bilstm_crf.pth'
    print(config.model_path)

    logging.info("="*30+" start to operate dataset "+"="*30)
    trainset = NerDataset(x_train,y_train,word2id,tag2id)
    validset = NerDataset(x_valid,y_valid,word2id,tag2id)

    loader = NerLoader(config)
    train_dataloader = loader._return_dataloader(trainset)
    valid_dataloader = loader._return_dataloader(validset)

    logging.info('='*30+" start to init model "+"="*30)
    config.VOCAB_SIZE = len(word2id)
    config.TARGET_SIZE = len(tag2id)
    model = Model(config).to(config.device)
    optimizer = torch.optim.Adam(model.parameters(),lr=config.LR)
    model.train()
    best_loss = 1e18

    logging.info("="*30+" start to train "+"="*30)
    for e in range(config.EPOCHES):
        step = 0
        losses = 0.
        for i,(features,target,mask) in enumerate(train_dataloader):
            features = features.to(config.device)
            target = target.to(config.device)
            mask = mask.to(config.device)

            loss = model(features,target,mask)
            losses+=loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            step+=1
            if step % config.print_step == 0:
                total_step = len(train_dataloader)+1
                logging.info("Epoch {}, step/total_step: {}/{} {:.2f}% Loss:{:.4f}".format(
                            e, step, total_step,
                            100. * step / total_step,
                            losses / config.print_step
                        ))
                losses = 0.
        cur_loss = evaluate(valid_dataloader,model,config)
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
    for i,(features,target,mask) in enumerate(dataloader):
        features = features.to(config.device)
        target = target.to(config.device)
        mask = mask.to(config.device)
        loss = model(features,target,mask)
        losses+=loss
    return losses/len(dataloader)
    
        





