import torch
from torch.utils.data import Dataset, DataLoader
import json
from tqdm import tqdm
import h5py
import numpy as np

def segment(data, window_size=288, overlap_size=50, output_file='app/data/segmented_data.h5'):
    '''load data, segmented to 288 token id, save to h5py'''

    # 將文本分割成token
    tokens = data.split()

    # 計算窗格的數量
    num_windows = len(tokens) // (window_size - overlap_size) + 1
    print(f"Total tokens: {len(tokens)}")
    print(f"Num windows: {num_windows}")
    
    # 初始化h5py文件
    with h5py.File(output_file, 'w') as f:
        # 創建一個dataset用於保存分割後的片段
        dset = f.create_dataset('segments', (num_windows, window_size), dtype='i')

        # 分割文本並保存到dataset
        for i in range(num_windows):
            start_idx = i * (window_size - overlap_size)
            end_idx = start_idx + window_size
            end_idx = min(end_idx, len(tokens))
            segment_data = tokens[start_idx:end_idx]

            # 將token轉換為ID，這裡使用示意性的轉換方式，實際使用時可能需要根據你的數據和模型進行適當處理
            segment_ids = [hash(token) for token in segment_data]

            # 將片段ID保存到dataset中
            dset[i, :] = segment_ids

class NQADataset(Dataset):
    def __init__(self, data_path='app/data/v1.0-simplified_simplified-nq-train.jsonl',num_samples=1000):
        self.data_path = data_path
        self.num_samples = num_samples
        self.data = self.load_data()#do not load 16GB!!!!
        #load h5py table, only ids-address and shape
        
    def load_data(self):
        data = []
        with open(self.data_path, 'r', encoding='utf-8') as f:
            for idx, line in enumerate(f):
                if idx == self.num_samples:
                    break  # Stop reading after reaching the desired number of samples
                # Assuming each line is a separate JSON object
                data.append(json.loads(line))
        return data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        '''input id, output segment
        dynamic load segment
        please check h5py
        '''

        sample = self.data[idx]
        
        #dict_keys(['document_text', 'long_answer_candidates', 'question_text', 'annotations', 'document_url', 'example_id'])
                
        return sample['document_text']#,sample['question_text']
    
class QAPairDataset():
    def __getitem__(id):
        '''input id, output(Q, short answer, long answer)
        dynamic load Pairs'''

def inspect_h5py_file(file_path,segment_id):
    with h5py.File(file_path, 'r') as f:
        print(f"Keys: {list(f.keys())}")

        # 查看 'segments' 數據集的形狀和數據類型
        segments_dataset = f['segments']
        
        print(f"Shape of 'segments' dataset: {segments_dataset.shape}")
        print(f"Dtype of 'segments' dataset: {segments_dataset.dtype}")

        # 一個片段
        print("Example segment:")
        print(segments_dataset[0, :])

        segment_data = f['segments'][segment_id, :]
        original_tokens = [str(token_hash) for token_hash in segment_data]

        # 將token列表結合成一個字符串
        original_text = ' '.join(original_tokens)

        return original_text

if __name__ == '__main__':
    
    qadataset = NQADataset(num_samples=1000)
    
    # Create a data loader
    batch_size = 128
    #print("stop")
    dataloader = DataLoader(qadataset, batch_size=batch_size, shuffle=False)

    output_file_path = 'app/data/question_list.txt'
    
    with open(output_file_path, 'w', encoding='utf-8') as output_file:
        # Iterate through the dataset
        for batch in tqdm(dataloader):
            
            # Get the document_text for each sample in the batch
            for document_text  in batch:
                # Write document_text to the output file, followed by a newline
                
                output_file.write(document_text + '\n')
    
    with open(output_file_path, 'r', encoding='utf-8') as file:
        segment(file.read())



    #demo
    print(inspect_h5py_file('app/data/segmented_data.h5',1))          
    