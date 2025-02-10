import argparse

from transformers import AutoModelForSequenceClassification, AutoTokenizer, Trainer, TrainingArguments
from datasets import load_dataset
from transformers import DataCollatorWithPadding
import evaluate
import numpy as np

parser = argparse.ArgumentParser(description="Bert Classification")
parser.add_argument('--model_path',default='/home/root123/workspace/model/bert-base-chinese/',type=str,help='')
parser.add_argument('--train_file',default="data/datasets/longnews/train.json",type=str,help='')
parser.add_argument('--valid_file',default="data/datasets/longnews/dev.json",type=str,help='')
parser.add_argument('--num_labels',default=7,type=int,help='')
parser.add_argument('--output',default="output_bert_base_chinese_longnews",type=str,help='')
args = parser.parse_args()

train_data = load_dataset("json", data_files=args.train_file)
valid_data = load_dataset("json", data_files=args.valid_file)

tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
model = AutoModelForSequenceClassification.from_pretrained(args.model_path,trust_remote_code=True,num_labels=args.num_labels).cuda()
bert_data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

def process_function(examples):
    examples["label"] = [int(unit) for unit in examples["label"]]
    return tokenizer(examples["content"], padding="max_length", truncation=True)

def load_data(dataset):
    dataset = dataset.map(process_function, batched=True)
    dataset = dataset.remove_columns(["content","metadata"])
    return dataset


def compute_metrics(eval_pred):

    precision_metric = evaluate.load("precision")
    recall_metric = evaluate.load("recall")
    f1_metric= evaluate.load("f1")
    accuracy_metric = evaluate.load("accuracy")

    logits, labels = eval_pred

    predictions = np.argmax(logits, axis=-1)
    precision = precision_metric.compute(predictions=predictions, references=labels,average=None)["precision"].tolist()
    recall = recall_metric.compute(predictions=predictions, references=labels,average=None)["recall"].tolist()
    f1 = f1_metric.compute(predictions=predictions, references=labels,average=None)["f1"].tolist()
    accuracy = accuracy_metric.compute(predictions=predictions, references=labels)["accuracy"]

    return {"precision": precision, "recall": recall, "f1-score": f1, 'accuracy': accuracy}


train_dataset = load_data(train_data)
valid_dataset = load_data(valid_data)

train_args = TrainingArguments(
    output_dir=args.output,
    eval_strategy="epoch",
    lr_scheduler_type="constant",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=5,
    warmup_ratio=0.1,
    weight_decay=0.001,
    save_strategy="epoch",
    load_best_model_at_end=True,
    save_total_limit=1,
    metric_for_best_model="accuracy"
)

trainer = Trainer(
    model=model,
    args=train_args,
    train_dataset=train_dataset["train"],
    eval_dataset=valid_dataset['train'],
    processing_class=tokenizer,
    data_collator=bert_data_collator,
    compute_metrics=compute_metrics
)

trainer.train()
