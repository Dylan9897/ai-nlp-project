import torch
from torch.utils.data import Dataset,DataLoader

PAD = "<PAD>"
UNK = "<UNK>"

class NerDataset(Dataset):
    def __init__(self,x_data,y_data,word2id,tag2id):
        super().__init__()
        self.x_data = x_data
        self.y_data = y_data
        self.word2id = word2id
        self.tag2id = tag2id

    def __len__(self):
        return len(self.x_data)

    def __getitem__(self,index):
        content = self.x_data[index]
        label = self.y_data[index]
        x_features = [self.word2id.get(unit,self.word2id[UNK]) for unit in content]
        y_features = [self.tag2id.get(unit,self.tag2id["O"]) for unit in label]
        return x_features,y_features

class NerLoader():
    def __init__(self,config):
        self.WORD_PAD_ID = config.WORD_PAD_ID
        self.LABEL_O_ID = config.LABEL_O_ID
        self.batch_size = config.batch_size


    def collate_fn(self,batch):
        batch.sort(key=lambda x: len(x[0]), reverse=True)
        max_len = len(batch[0][0])
        inputs = []
        target = []
        mask = []
        for item in batch:
            pad_len = max_len - len(item[0])
            inputs.append(item[0] + [self.WORD_PAD_ID] * pad_len)
            target.append(item[1] + [self.LABEL_O_ID] * pad_len)
            mask.append([1] * len(item[0]) + [0] * pad_len)
        return torch.tensor(inputs), torch.tensor(target), torch.tensor(mask).bool()

    def _return_dataloader(self,dataset):
        loader = DataLoader(
            dataset,
            self.batch_size,
            shuffle=True,
            collate_fn = self.collate_fn
        )
        return loader




