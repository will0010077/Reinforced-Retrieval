
import json
import torch
from torch.utils.data import Dataset, DataLoader
import time

class JsonlDataset(Dataset):
    def __init__(self, file_path):
        self.data = []
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                json_obj = json.loads(line.strip())
                self.data.append(json_obj)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]['Question'],self.data[idx]['Answer']#,self.data[idx]['File_name'],self.data[idx]['Document']

# 使用示例
file_path = 'unique_smart_factory_qapairs.jsonl'
dataset = JsonlDataset(file_path)
print(len(dataset))

for i in range(len(dataset)):
    print('Q:',dataset[i][0])
    print('A:',dataset[i][1])
    print('-' * 20)
    time.sleep(1)
