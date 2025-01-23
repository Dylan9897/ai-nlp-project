import torch
from transformers import AutoTokenizer
from torch.utils.data import Dataset, DataLoader

import json

class DataSet(Dataset):
    def __init__(self,data_path,tokenizer,config=None):
        self.tokenizer = tokenizer
        self.config = config if config else {}
        self.load_data(data_path)

    def __len__(self):
        return len(self.abstracts)

    def load_data(self,file_path):
        self.abstracts = []
        self.labels = []
        with open(file_path,"r",encoding="utf-8") as fl:
            for line in fl.readlines():
                line = json.loads(line)
                self.abstracts.append(line["content"])
                self.labels.append(line["label"])

    def __getitem__(self, index):
        encoding = self.tokenizer.encode_plus(
            self.abstracts[index],
            add_special_tokens=True,
            max_length=self.config.get("max_length",64),
            return_token_type_ids=True,
            pad_to_max_length=True,
            return_attention_mask=True,
            return_tensors='pt',
            truncation=True
        )
        return {
            'texts': self.abstracts[index],
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(int(self.labels[index]), dtype=torch.long)
        }
        
                


def get_dataloader(file_path,tokenizer):
    dataloader = DataLoader(dataset=DataSet(file_path,tokenizer), batch_size=32, shuffle=True, drop_last=True)
    return dataloader
