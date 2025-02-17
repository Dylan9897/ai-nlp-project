import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from sklearn import metrics
import time
from tensorboardX import SummaryWriter
from Student.module.Ranger import opt_func
from Student.module.utils import get_time_dif

def loss_fn_kd(students_output, labels, teacher_outputs, T, alpha):
    KD_loss = nn.KLDivLoss()(F.log_softmax(students_output / T, dim=1),
                             F.softmax(teacher_outputs / T, dim=1)) * (alpha * T * T) + \
              F.cross_entropy(students_output, labels) * (1. - alpha)
    return KD_loss

def train(teacher,student,train_data_loader,valid_data_loader,test_data_loader,config):
    start_time = time.time()
    for name, p in teacher.named_parameters():
        p.requires_grad = False
    student.train()
    optim = opt_func(student.parameters(),lr=1e-5)
    
    total_batch = 0 # 记录进行了多少个batch
    dev_best_acc = 0# float('inf')
    last_improve = 0
    flag = False
    writer = SummaryWriter(log_dir="log"+'/'+time.strftime('%m-%d_%H.%M',time.localtime()))
    for epoch in range(config.num_epoches):
        print('Epoch [{}/{}]'.format(epoch + 1, config.num_epoches))
        for i,unit in enumerate(train_data_loader):
            embed = unit["embed"]
            input_ids = unit["input_ids"]
            attention_mask = unit["attention_mask"]
            labels = unit["labels"]

            teacher_out = teacher(
                input_ids = input_ids,
                attention_mask = attention_mask,
            )
            _, preds = torch.max(teacher_out, dim=1)
            
            student_out = student(embed)
            student.zero_grad()
            loss = loss_fn_kd(student_out,labels,teacher_out,10,0.9)
            loss.backward()
            optim.step()
            if total_batch % 100 == 0:
                true = labels.data.cpu()
                predic = torch.max(student_out.data, 1)[1].cpu()
                train_acc = metrics.accuracy_score(true, predic)
                dev_acc, dev_loss = evaluate(config, teacher,student, valid_data_loader)
                if dev_acc > dev_best_acc:
                    dev_best_loss = dev_loss
                    torch.save(student.state_dict(), "saved_dict"+'/checkpoint{}.pth'.format('_KD_best'))
                    improve = '*'
                    last_improve = total_batch
                else:
                    improve = ''
                time_dif = get_time_dif(start_time)
                msg = 'Iter: {0:>6},  Train Loss: {1:>5.2},  Train Acc: {2:>6.2%},  Val Loss: {3:>5.2},  Val Acc: {4:>6.2%},  Time: {5} {6}'
                print(msg.format(total_batch, loss.item(), train_acc, dev_loss, dev_acc, time_dif, improve))
                writer.add_scalar("loss/train", loss.item(), total_batch)
                writer.add_scalar("loss/dev", dev_loss, total_batch)
                writer.add_scalar("acc/train", train_acc, total_batch)
                writer.add_scalar("acc/dev", dev_acc, total_batch)
                student.train()
            total_batch += 1
            if total_batch - last_improve > config.require_improvement:
                # 验证集loss超过1000batch没下降，结束训练
                print("No optimization for a long time, auto-stopping...")
                flag = True
                break
        if flag:
            break
    writer.close()
    test(config, teacher,student, test_data_loader)


def test(config,teacher,student, test_iter):
    # test
    student.load_state_dict(torch.load("saved_dict"+'/checkpoint{}.pth'.format('_KD_best')))
    student.eval()
    start_time = time.time()
    test_acc, test_loss, test_report, test_confusion = evaluate(config, teacher,student, test_iter, test=True)
    msg = 'Test Loss: {0:>5.2},  Test Acc: {1:>6.2%}'
    print(msg.format(test_loss, test_acc))
    print("Precision, Recall and F1-Score...")
    print(test_report)
    print("Confusion Matrix...")
    print(test_confusion)
    time_dif = get_time_dif(start_time)
    print("Time usage:", time_dif)


def evaluate(config,teacher,student, data_iter, test=False):
    student.eval()
    loss_total = 0
    predict_all = np.array([], dtype=int)
    labels_all = np.array([], dtype=int)
    with torch.no_grad():
        for i, unit in enumerate(data_iter):
            embed = unit["embed"]
            input_ids = unit["input_ids"]
            attention_mask = unit["attention_mask"]
            labels = unit["labels"]

            teacher_out = teacher(
                input_ids = input_ids,
                attention_mask = attention_mask,
            )
            
            student_out = student(embed)
            loss = loss_fn_kd(student_out,labels,teacher_out,10,0.9)


            loss_total += loss
            labels = labels.data.cpu().numpy()
            predic = torch.max(student_out.data, 1)[1].cpu().numpy()
            labels_all = np.append(labels_all, labels)
            predict_all = np.append(predict_all, predic)

    acc = metrics.accuracy_score(labels_all, predict_all)
    if test:
        target_names = ['教育','家居','时尚','时政','科技','房产','财经']

        report = metrics.classification_report(labels_all, predict_all, target_names=target_names, digits=4)
        confusion = metrics.confusion_matrix(labels_all, predict_all)
        return acc, loss_total / len(data_iter), report, confusion
    return acc, loss_total / len(data_iter)


