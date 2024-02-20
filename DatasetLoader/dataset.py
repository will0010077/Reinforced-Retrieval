import torch
from torch.utils.data import Dataset, DataLoader
import json
from tqdm import tqdm
import h5py
import numpy as np
from transformers import AutoTokenizer
import math
from threading import Thread, Lock, Event
from queue import Queue
import time
tokenizer = AutoTokenizer.from_pretrained("facebook/contriever")

def write_segment(f, s2c:Queue, qlock:Lock, flock:Lock, event:Event):
    while True:
        qlock.acquire()
        if s2c.qsize()>0:
            segment_data = []
            for _ in range(min(512, s2c.qsize())):
                segment_data.append(s2c.get())
            qlock.release()
            segment_data = torch.stack(segment_data)

            flock.acquire()
            f['segments'].resize((f['segments'].shape[0] + segment_data.shape[0]), axis=0)
            f['segments'][-segment_data.shape[0]:, :] = segment_data
            flock.release()
        else:
            qlock.release()
            if event.is_set():
                break
            time.sleep(0.001)

def segmentation(data, window_size, step, output_file='app/data/segmented_data.h5'):
    '''load data, segmented to 288 token id, save to h5py'''

    # Token indices sequence length is longer than the specified maximum sequence length for this model (14441966 > 512).
    # 將文本分割成token, use tokenizer!!
    first=True
    max_queue_size = 4096
    num_thread = 4
    seg_count=0
    skip=0
    docu_ids=set()


    output_lock = Lock()
    qlock = Lock()
    s2c = Queue()
    terminal_signal = Event()
    # 初始化h5py文件
    with h5py.File(output_file, 'w') as f:
        # 創建一個dataset用於保存分割後的片段
        bar = tqdm(data, total=310000)
        pool=[Thread(target = write_segment, args = (f, s2c, output_lock, qlock, terminal_signal)) for _ in range(num_thread)]
        [t.start() for t in pool]
        for text, url  in bar:
            if url in docu_ids:
                skip+=1
                continue
            else:
                docu_ids.add(url)
            tokens = tokenizer(text, return_tensors='pt').input_ids[0]
            # print(tokens)#Tensor [3214,  2005, 25439, 87,..., 2759]

            # 計算窗格的數量
            num_windows = int(math.ceil((len(tokens)-window_size+0) / (step)) + 1)
            seg_count+=num_windows
            bar.set_description_str(f'{seg_count:7.1e}/{seg_count*310000/(bar.last_print_n+1):7.1e}, skip:{skip}')
            # print(f"Total tokens: {len(tokens)}")
            # print(f"Num windows: {num_windows}")

            # 分割文本並保存到dataset
            segment_list=[]
            for i in range(num_windows):
                start_idx = i * (step)
                end_idx = start_idx + window_size
                end_idx = min(end_idx, len(tokens))
                segment_data = tokens[start_idx:end_idx]
                if len(segment_data) < window_size :
                    assert i == num_windows - 1
                    eos_padding = torch.zeros(window_size - len(segment_data), dtype=torch.long)
                    segment_data = torch.cat((segment_data, eos_padding))

                if first:
                    dset = f.create_dataset('segments', data= segment_data[None,:] ,maxshape=(None, window_size), dtype='i')
                    first=False
                else:

                    while s2c.qsize() >= max_queue_size:
                        time.sleep(0.001)
                    s2c.put(segment_data)
        terminal_signal.set()
        [t.join() for t in pool]


class NQADataset():
    def __init__(self, data_path='app/data/v1.0-simplified_simplified-nq-train.jsonl', num_samples=None):
        self.data_path = data_path
        self.num_samples = num_samples

    def load_data(self):
        with open(self.data_path, 'r', encoding='utf-8') as f:
            for idx, line in enumerate(f):
                if idx == self.num_samples:
                    break  # Stop reading after reaching the desired number of samples
                # Assuming each line is a separate JSON object
                a_line = json.loads(line)
                out = a_line['document_text'], a_line['document_url']
                if type(out[0])==str and type(out[1])==str:

                    yield out[0], out[1].strip()

    # def __len__(self):
    #     return len(self.data)

    # def __getitem__(self, idx):
    #     '''input id, output segment
    #     dynamic load segment
    #     please check h5py
    #     '''

    #     sample = self.data[idx]
    #     if type(sample)==list:
    #         out = [i['document_text'] for i in sample]
    #     else:
    #         out = sample['document_text']
    #     #dict_keys(['document_text', 'long_answer_candidates', 'question_text', 'annotations', 'document_url', 'example_id'])

    #     return out #,sample['question_text']

class QAPairDataset(Dataset):
    def __init__(self, data_path='/home/devil/workspace/nlg_progress/backend/app/data/cleandata.pt', num_samples=None):
        self.data_path = data_path
        self.num_samples = num_samples
        self.data = torch.load(self.data_path)
        if num_samples is not None:
            self.data=self.data[:num_samples]
        # self.load_data()
    def load_data(self):
        with open(self.data_path, 'r', encoding='utf-8') as f:
            for idx, line in enumerate(f):
                if idx == self.num_samples:
                    break  # Stop reading after reaching the desired number of samples
                # Assuming each line is a separate JSON object
                a_line = json.loads(line)
                q = a_line['question']
                a = a_line['answer']
                if type(q)==str and type(a[0])==str:
                    self.data.append([q, a])

    def __getitem__(self, idx):

        if type(idx)==int:
            q=self.data[idx]['question']
            a=self.data[idx]['short_answers']
            # q, a = self.data[idx]
            all_a=a
            a = a[np.random.randint(len(a))]
            return [q, a,all_a]
        else:
            data = []
            for i in idx:
                q=self.data[i]['question']
                a=self.data[i]['short_answers']
                # q, a = self.data[i]
                all_a=a
                a = a[np.random.randint(len(a))]
                data.append([q, a,all_a])
            return data

    def __len__(self):
        return len(self.data)

class cleanDataset(Dataset):
    def __init__(self, data_path='/home/contriever/v1.0-simplified_simplified-nq-train.jsonl',num_samples=None):
        self.data_path = data_path
    
        self.num_samples = num_samples
        
        self.data = self.load_data()

    def load_data(self):
        data = []
        with open(self.data_path, 'r', encoding='utf-8') as f:
            for idx, line in enumerate(f):
                if idx == self.num_samples:
                    break  

                data.append(json.loads(line))
        return data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):

        sample = self.data[idx]
        #dict_keys(['document_text', 'long_answer_candidates', 'question_text', 'annotations', 'document_url', 'example_id'])
        # print(sample['question_text'])
        a=sample['annotations'][0]['long_answer']#sample['long_answer_candidates'][random_number]
        
        long_answer=' '.join(sample['document_text'].split()[a['start_token']:a['end_token']])
        short_annotations=sample['annotations'][0]['short_answers']
        if short_annotations==[]:
            return None,None,None,None,None
        if type(short_annotations)==dict:
            short_annotations=[short_annotations]
        
        answers=[]
        for i in range(len(short_annotations)):
            answer=' '.join(sample['document_text'].split()[short_annotations[i]['start_token']:short_annotations[i]['end_token']])
            answers.append(answer)
        # print(answers)
        # print(len(sample['question_text']))
        return answers, long_answer, str(sample['document_text']), sample, sample['question_text']

def cleandata():
    data_path='/home/devil/workspace/nlg_progress/backend/app/data/v1.0-simplified_simplified-nq-train.jsonl'

    dataset=cleanDataset(data_path=data_path,num_samples=None)

    datasample=[]
    total_a_lenth=[]
    for i in tqdm(range(len(dataset))):
        ans,la,d ,s,q=dataset[i]
        if ans!=None:
            for a in ans:
                if len(a.split())>10:
                    ans.remove(a)
                    continue
                total_a_lenth.append(len(a.split()))
        if ans==None or la==None or ans ==[]:
            continue
        else:
            datasample.append(dict(short_answers=ans,long_answer=la,document=d,question=q))
    
    print(sum(total_a_lenth)/len(total_a_lenth))
    print(max(total_a_lenth))
    print('total:',len(datasample))#98708
    torch.save(datasample,'/home/devil/workspace/nlg_progress/backend/app/data/cleandata.pt')
    print('saved')



if __name__=="__main__":
    cleandata()