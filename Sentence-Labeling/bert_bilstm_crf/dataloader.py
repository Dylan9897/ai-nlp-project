import torch
from torch.utils.data import Dataset,DataLoader

PAD = "<PAD>"
UNK = "<UNK>"

class NerDataset(Dataset):
    def __init__(self,x_data,y_data,tokenizer,tag2id,max_len):
        super().__init__()
        self.x_data = x_data
        self.y_data = y_data
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.tag2id = tag2id

    def __len__(self):
        return len(self.x_data)

    def __getitem__(self,index):
        content = self.x_data[index]
        label = self.y_data[index]
        label = [self.tag2id[unit] for unit in label]
        if len(label) > self.max_len:
            label = label[:self.max_len]
        else:
            while len(label) < self.max_len:
                label.append(self.tag2id["O"])

        encoding = self.tokenizer.encode_plus(
            content,
            add_special_tokens=True,
            max_length = self.max_len,
            return_token_type_ids=True,
            pad_to_max_length=True,
            return_attention_mask=True,
            return_tensors='pt',
        )

        return {
            # 'texts': content,
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels':torch.tensor(label,dtype=torch.long)
        }


def create_data_loader(x_data,y_data,tokenizer,tag2id,max_len,batch_size):
    ds = NerDataset(
        x_data,
        y_data,
        tokenizer = tokenizer,
        tag2id=tag2id,
        max_len = max_len
    )
    return DataLoader(
        ds,
        batch_size=batch_size
    )





