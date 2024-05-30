import torch
from torch.utils.data import Dataset, DataLoader
import json,re
from tqdm import tqdm
import h5py
import numpy as np
from transformers import AutoTokenizer, BertTokenizerFast
import math
from threading import Thread, Event,Lock
import multiprocessing

from queue import Queue
import time,os,gc

def generate_segments(tokens, window_size, step)-> torch.Tensor:

    segment_list=[]
    for token in tokens:
        for i in range(0, max(len(token)-window_size,1), step):
            segment_data = token[max(0, min(i, len(token)-window_size)):i+window_size]
            # print(segment_data.shape)
            if len(segment_data) < window_size:
                padding = torch.zeros(window_size - len(segment_data), dtype=torch.long)
                segment_data = torch.cat((segment_data, padding))
            segment_list.append(segment_data)

    segment_list=torch.stack(segment_list)
    return  segment_list

def Write_segment_Buffer(output_file, s2c:Queue, qlock:Lock, flock:Lock, event:Event):
    while True:
        qlock.acquire()
        if not s2c.empty():
            segment_data = s2c.get()
            qlock.release()
            flock.acquire()
            try:
                f=h5py.File(output_file, 'a')
                if 'segments' not in f:
                    f.create_dataset('segments', data=segment_data, maxshape=(None, segment_data.shape[1]), dtype='i')
                else:
                    f['segments'].resize((f['segments'].shape[0] + segment_data.shape[0]), axis=0)
                    f['segments'][-segment_data.shape[0]:, :] = segment_data
            finally:
                f.close()
            flock.release()
        else:
            qlock.release()
            if event.is_set() and s2c.empty():
                break
            time.sleep(0.001)

class NQADataset():
    def __init__(self, data_path='app/data/v1.0-simplified_simplified-nq-train.jsonl', num_samples=None):
        self.data_path = data_path
        self.num_samples = num_samples

    def load_data(self):
        with open(self.data_path, 'r', encoding='utf-8') as f:
            data=f.readlines()
            res=[]
            for idx, line in enumerate(data):
                if idx == self.num_samples:
                    break  # Stop reading after reaching the desired number of samples
                # Assuming each line is a separate JSON object
                a_line = json.loads(line)
                out = a_line['document_text'], a_line['document_url']
                if type(out[0])==str and type(out[1])==str:
                    res.append((out[0], out[1].strip()))
                    # yield out[0], out[1].strip()\
        return res

def segmentation(shared_dict, file_lock, shared_int, segment, window_size, step, output_file='app/data/segmented_data.h5'):
    '''load data, segmented to 288 token id, save to h5py'''

    # Please check cased/uncased, LexMAE use bert-base-uncased
    tokenizer:BertTokenizerFast = BertTokenizerFast.from_pretrained("google-bert/bert-base-uncased")
    tokenizer.model_max_length = 10**6
    # 將文本分割成token, use tokenizer!!
    first=True
    max_queue_size = 8
    num_thread = 4
    seg_count=0
    docu_ids=shared_dict

    output_lock = Lock()
    qlock = Lock()
    s2c = Queue()
    terminal_signal = Event()
    # 初始化h5py文件
    # 創建一個dataset用於保存分割後的片段
    def c(batch):
        return list(zip(*batch))
    bar = tqdm(DataLoader(segment, batch_size=64, collate_fn=c))
    pool=[Thread(target = Write_segment_Buffer, args = (output_file, s2c, output_lock, qlock, terminal_signal)) for _ in range(num_thread)]
    [t.start() for t in pool]

    for texts, urls  in bar:
        with file_lock:
            valid=[]
            for i in range(len(texts)):
                if urls[i] in docu_ids:
                    shared_int.value+=1
                else:
                    docu_ids.update({urls[i]:True})
                    valid.append(i)
                    # print(len(docu_ids))
        urls = [urls[i] for i in range(len(texts)) if i in valid]
        texts = [texts[i] for i in range(len(texts)) if i in valid]
        if len(texts)==0:
            continue

        texts = [re.sub("(<[/a-z0-9A-Z]*>)",'', string=t.strip()) for t in texts]
        tokens = [torch.tensor(i) for i in tokenizer(texts).input_ids]
       
        if first:
            segment_data=generate_segments(tokens, window_size, step)
            try:
                f=h5py.File(output_file, 'w', fs_strategy = 'page')
                if 'segments' not in f:
                    f.create_dataset('segments', data=segment_data, maxshape=(None, segment_data.shape[1]), dtype='i')
            finally:
                f.close()
            first=False
            continue

        while s2c.qsize() >= max_queue_size:
            time.sleep(0.001)
        s2c.put(generate_segments(tokens, window_size, step))
        bar.set_description_str(f"Process ID:{os.getpid()} Skip: {shared_int.value} Dict length:{len(docu_ids)}")
    terminal_signal.set()
    [t.join() for t in pool]


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

class DocumentDatasets():
    def __init__(self, path='app/data/segmented_data_', num_file=12) -> None:
        self.file_index = num_file

        self.file_list = [h5py.File(path+f"{i}.h5", 'r', page_buf_size = 2**7 * 2**20)[u'segments'] for i in range(self.file_index)]
        self.file_len = [f.shape[0] for f in self.file_list]
        self.offset = [0]
        for i in range(0, len(self.file_len)-1):
            self.offset.append(sum(self.file_len[0:i+1]))
        self.shape = torch.Size([self.__len__(), self.file_list[0].shape[1]])

    def get_single(self, idx):

        for i in reversed(range(len(self.offset))):
            if idx >= self.offset[i]:
                return torch.from_numpy(self.file_list[i][idx-self.offset[i]])
                
        raise IndexError(idx)
    
    def get_multi(self, ids, inverted_index, return_pt):
        if len(ids)>10**5:
            bar = tqdm(ids, ncols=0)
        else:
            bar = ids
        for idx in bar:
            return_pt[inverted_index[idx]] = self.get_single(idx)
            
    def __getitem__(self , ids):
        if hasattr(ids, '__iter__') or type(ids)==slice:
            if type(ids)==slice:
                start, stop, step = ids.start, ids.stop, ids.step
                if step is None:
                    step=1
                if start is None:
                    start=0

                ids = range(start, stop,step)

            inverted_index = {ele: i for i, ele in enumerate(ids)}
            return_pt=torch.empty([len(ids), self.shape[1]], dtype=torch.long)
            
            if len(ids)>=10**4:
                bar = tqdm(ids, ncols=0)
            else:
                bar = ids
            for i, idx in enumerate(bar):
                return_pt[i] = self.get_single(idx)
                
            #!!! this is slower!!!
            # num_thread = 8
            # step = len(ids)//num_thread+1
            # pool = [Thread(target=self.get_multi, args=(ids[i:i+step], inverted_index, return_pt)) for i in range(0, len(ids), step)]
            # [t.start() for t in pool]
            # [t.join() for t in pool]

            return return_pt

        return self.get_single(ids)

    def __len__(self):
        #return DocumentLeng
        return sum(self.file_len)

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
