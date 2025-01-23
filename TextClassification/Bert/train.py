import os
import time

from functools import partial

import torch
from torch.utils.data import DataLoader
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, default_data_collator, get_scheduler
from module.DataLoader import get_dataloader

from module.printf import reset_console
from module.parser_args import return_args
from module.class_metrics import ClassEvaluator
from module.FocalLoss import FocalLoss

args = return_args()
reset_console(args)

# 使用差分学习率
def get_parameters(model, model_init_lr, multiplier, classifier_lr):
    parameters = []
    lr = model_init_lr
    for layer in range(12, -1, -1):
        layer_params = {
            'params': [p for n, p in model.named_parameters() if f'encoder.layer.{layer}.' in n],
            'lr': lr
        }
        parameters.append(layer_params)
        lr *= multiplier
    classifier_params = {
        'params': [p for n, p in model.named_parameters() if 'layer_norm' in n or 'linear' in n
                   or 'pooling' in n],
        'lr': classifier_lr
    }
    parameters.append(classifier_params)
    return parameters

def evaluate_model(model, metric, data_loader):
    """
    在测试集上评估当前模型的训练效果。

    Args:
        model: 当前模型
        metric: 评估指标类(metric)
        data_loader: 测试集的dataloader
        global_step: 当前训练步数
    """
    model.eval()
    with torch.no_grad():
        for step, batch in enumerate(data_loader):
            outputs = model(input_ids=batch['input_ids'].to(args.device),
                            attention_mask=batch['attention_mask'].to(args.device))
            predictions = outputs.logits.argmax(dim=-1).detach().cpu().numpy().tolist()
            metric.add_batch(pred_batch=predictions, gold_batch=batch["labels"].detach().cpu().numpy().tolist())
    eval_metric = metric.compute()
    model.train()
    return eval_metric['accuracy'], eval_metric['precision'], \
        eval_metric['recall'], eval_metric['f1'], \
        eval_metric['class_metrics']


def train():
    model = AutoModelForSequenceClassification.from_pretrained(args.model, num_labels=args.num_labels)
    tokenizer = AutoTokenizer.from_pretrained(args.model)

    train_dataloader = get_dataloader(file_path=args.train_path,tokenizer=tokenizer)
    valid_dataloader = get_dataloader(file_path=args.dev_path,tokenizer=tokenizer)


    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]
    optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=args.learning_rate)
    model.to(args.device)
    
    
    # 根据训练轮数计算最大训练步数，以便于scheduler动态调整lr
    num_update_steps_per_epoch = len(train_dataloader)
    max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    warm_steps = int(args.warmup_ratio * max_train_steps)
    lr_scheduler = get_scheduler(
        name='linear',
        optimizer=optimizer,
        num_warmup_steps=warm_steps,
        num_training_steps=max_train_steps,
    )
    
    loss_list = []
    tic_train = time.time()
    metric = ClassEvaluator()

    if args.loss_func == 'cross_entropy':
        criterion = torch.nn.CrossEntropyLoss()
    elif args.loss_func == 'focal_loss':
        criterion = FocalLoss()

    global_step, best_f1, best_acc, best_precision = 0, 0, 0, 0
    for epoch in range(1, args.num_train_epochs + 1):
        for batch in train_dataloader:
            outputs = model(input_ids=batch['input_ids'].to(args.device),
                            attention_mask=batch['attention_mask'].to(args.device))
            labels = batch['labels'].to(args.device)
            loss = criterion(outputs.logits, labels)
            loss.backward()
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
            loss_list.append(float(loss.cpu().detach()))

            global_step += 1
            if global_step % args.logging_steps == 0:
                time_diff = time.time() - tic_train
                loss_avg = sum(loss_list) / len(loss_list)
                # writer.add_scalar('train/train_loss', loss_avg, global_step)
                print("global step %d, epoch: %d, loss: %.5f, speed: %.2f step/s"
                      % (global_step, epoch, loss_avg, args.logging_steps / time_diff))
                tic_train = time.time()

            if global_step % args.valid_steps == 0:

                acc, precision, recall, f1, class_metrics = evaluate_model(model, metric, eval_dataloader)

                print("Evaluation precision: %.5f, recall: %.5f, F1: %.5f, acc: %.5f" % (precision, recall, f1, acc))
                # if f1 > best_f1 or acc > best_acc or precision > best_precision:
                if f1 > best_f1:
                    print(
                        f"best F1 performence has been updated: {best_f1:.5f} --> {f1:.5f}"
                    )
                    print(f'Each Class Metrics are: {class_metrics}')
                    best_f1 = f1
                    cur_save_dir = os.path.join(args.save_dir, "model_best")
                    if not os.path.exists(cur_save_dir):
                        os.makedirs(cur_save_dir)
                    model.save_pretrained(os.path.join(cur_save_dir))
                    tokenizer.save_pretrained(os.path.join(cur_save_dir))
                tic_train = time.time()


if __name__=="__main__":
    train()

