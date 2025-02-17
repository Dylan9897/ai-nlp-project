import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import time
import os
from tensorboardX import SummaryWriter
from collections import defaultdict
from transformers import AutoConfig,AutoModel,AutoTokenizer,AdamW,get_linear_schedule_with_warmup,logging

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


def train(model,train_data_loader,valid_data_loader,test_data_loader,config):
    if not config.optim:
        optimizer = AdamW(model.parameters(),lr=2e-5,correct_bias=False)
    else:
        parameters=get_parameters(model,2e-5,0.95, 1e-4)
        optimizer = AdamW(parameters)

    writer = SummaryWriter(log_dir=config.log_path+'/'+time.strftime('%m-%d_%H.%M',time.localtime()))

    total_steps = len(train_data_loader)*config.num_epoches
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=0,
        num_training_steps=total_steps
    )
    loss_fn = nn.CrossEntropyLoss()
    model = model.train()
    
    
    history = defaultdict(list)
    best_accuracy = 0
    total_steps = 0
    for epoch in range(config.num_epoches):
        losses = []
        correct_predictions = 0
        for i,unit in enumerate(train_data_loader):
            
            input_ids = unit["input_ids"].to(config.device)
            attention_mask = unit["attention_mask"].to(config.device)
            targets = unit["labels"].to(config.device)
            outputs = model(
                input_ids=input_ids,
                attention_mask = attention_mask
            )
            _,preds = torch.max(outputs,dim=1)
            loss = loss_fn(outputs,targets)
            correct_predictions += torch.sum(preds == targets)
            losses.append(loss.item())
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            total_steps += 1

        train_acc = correct_predictions.double()/config.train_examples
        train_loss = np.mean(losses)
        val_acc, val_loss = eval_model(
                        model,
                        valid_data_loader,
                        loss_fn,
                        config.device,
                        config.valid_examples
        )
        print("epoch:{},total step:{},Train loss is {},Train acc is {},valid loss is {},valid acc is {}".format(epoch,total_steps,train_loss,train_acc,val_loss,val_acc))
        # print(f'Train loss {train_loss} accuracy {train_acc}')
        # print(f'Val   loss {val_loss} accuracy {val_acc}')
        history['train_acc'].append(train_acc)
        history['train_loss'].append(train_loss)
        history['val_acc'].append(val_acc)
        history['val_loss'].append(val_loss)
        if val_acc > best_accuracy:
            torch.save(model.state_dict(), os.path.join(config.MODEL_DIR,'best_model_state.ckpt'))
            best_accuracy = val_acc
    test_acc, _ = eval_model(model,test_data_loader,loss_fn,config.device,config.test_examples)
    print(f"test result is {test_acc.item()}")
    y_texts, y_pred, y_pred_probs, y_test = get_predictions(model,test_data_loader)
    class_names = ['教育', '家居', '时尚', '时政', '科技', '房产', '财经']
    print("accuracy is {}".format(test_acc))
    print(classification_report(y_test, y_pred, target_names=[str(label) for label in class_names]))
    


def eval_model(model, data_loader, loss_fn, device,n_examples):
    model = model.eval() # 验证预测模式
    losses = []
    correct_predictions = 0
    with torch.no_grad():
        for d in data_loader:
            input_ids = d["input_ids"].to(device)
            attention_mask = d["attention_mask"].to(device)
            targets = d["labels"].to(device)
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )
            _, preds = torch.max(outputs, dim=1)
            loss = loss_fn(outputs, targets)
            correct_predictions += torch.sum(preds == targets)
            losses.append(loss.item())
    return correct_predictions.double()/n_examples, np.mean(losses)

def get_predictions(model, data_loader):
    model = model.eval()

    texts = []
    predictions = []
    prediction_probs = []
    real_values = []

    with torch.no_grad():
        for d in data_loader:
            texts = d["texts"]
            input_ids = d["input_ids"].to(device)
            attention_mask = d["attention_mask"].to(device)
            targets = d["labels"].to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )
            _, preds = torch.max(outputs, dim=1)
            probs = F.softmax(outputs, dim=1)
            texts.extend(texts)
            predictions.extend(preds)
            prediction_probs.extend(probs)
            real_values.extend(targets)
    predictions = torch.stack(predictions).cpu()
    prediction_probs = torch.stack(prediction_probs).cpu()
    real_values = torch.stack(real_values).cpu()
    return texts, predictions, prediction_probs, real_values

        
