import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt

from sklearn import metrics
from transformers import AdamW,get_linear_schedule_with_warmup
from tqdm import tqdm
from dataloader import traindataloader,validdataloader
from model import Model


SAVED_DIR = "saved_ckpt"
EPOCHES = 10
BERT_PATH = "../bert-base-chinese"
WARMUP_PROPORTION = 0.1
METHOD = "mean_pooling"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = Model.from_pretrained(BERT_PATH)
model.to(device)
print(model)
total_steps = len(traindataloader) * EPOCHES
optimizer = AdamW(model.parameters(), lr=5e-5)
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=int(WARMUP_PROPORTION * total_steps), num_training_steps=total_steps)

loss_vals = []
print("start to train")
for epoch in range(EPOCHES):
    model.train()
    epoch_loss = []
    pbar = tqdm(traindataloader)
    pbar.set_description("[Epoch {}]".format(epoch))
    for batch in pbar:
        model.zero_grad()
        loss = model.compute_loss(
            batch["input_ids_1"],
            batch["attention_mask_1"],
            batch["input_ids_2"],
            batch["attention_mask_2"],
            batch["labels"],
            METHOD
        )
        loss.backward()
        epoch_loss.append(loss.item())
        optimizer.step()
        scheduler.step()
        pbar.set_postfix(loss=loss.item())
    loss_vals.append(np.mean(epoch_loss)) 


    model.eval()
    for t in [0.5, 0.6, 0.7, 0.8, 0.9, 0.95]:
        predict_all = np.array([], dtype=int)
        labels_all = np.array([], dtype=int)
        with torch.no_grad():        
            for batch in validdataloader:
                labels = batch["labels"]
                pred = model.predict(
                    batch["input_ids_1"],
                    batch["attention_mask_1"],
                    batch["input_ids_2"],
                    batch["attention_mask_2"],
                    t,
                    METHOD
                )
                predict_all = np.append(predict_all, pred)   
                truth = labels.cpu().numpy()
                labels_all = np.append(labels_all, truth)
        acc = metrics.accuracy_score(labels_all, predict_all)
        print(f'Epoch-{epoch} Threshold-{t}: Accuracy on dev is {acc}')   

model.save_pretrained(f'{SAVED_DIR}_{METHOD}')
plt.plot(np.linspace(1, EPOCHES, EPOCHES).astype(int), loss_vals)