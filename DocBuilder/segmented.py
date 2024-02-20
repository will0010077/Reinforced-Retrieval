import sys
sys.path.append('../..')
import dataset
import h5py
import torch

from Retriever import Contriever, DOC_Retriever

from torch.utils.data import DataLoader
from tqdm import tqdm
import random
import yaml

with open('app/lib/config.yaml', 'r') as yamlfile:
    config = yaml.safe_load(yamlfile)


seed = config['seed']
torch.manual_seed(seed)
random.seed(seed)
class DocumentDatasets():
    def __init__(self) -> None:

        self.file_list = [h5py.File(f"app/data/segmented_data_{i}.h5", 'r')[u'segments'] for i in range(4)]
        self.file_len = [f.shape[0] for f in self.file_list]
        self.offset = [0]
        for i in range(0, len(self.file_len)-1):
            self.offset.append(sum(self.file_len[0:i+1]))
        self.shape = [self.__len__(), self.file_list[0].shape[1]]


    def __getitem__(self , ids):
        if hasattr(ids, '__iter__') or type(ids)==slice:
            if type(ids)==slice:
                start, stop, step = ids.start, ids.stop, ids.step
                if step is not None:
                    ids = range(start, stop, step)
                else:
                    ids = range(start, stop)

            
            out=torch.empty([len(ids), self.shape[1]], dtype=torch.long)
            for i, idx in enumerate(ids):
                for i in reversed(range(len(self.offset))):
                    if idx >= self.offset[i]:
                        out[i]= torch.tensor(self.file_list[i][idx-self.offset[i]],dtype=torch.long)
                        break
            return out

        else:
            for i in reversed(range(len(self.offset))):
                if ids >= self.offset[i]:
                    return torch.tensor(self.file_list[i][ids-self.offset[i]],dtype=torch.long)
        # else:
        #     raise TypeError
        raise IndexError(f'index:{ids} out of range.')

    def __len__(self):
        #return DocumentLeng
        return sum(self.file_len)


if __name__ == '__main__':
    windows=config['data_config']['windows']
    step=config['data_config']['step']
    # qadataset = dataset.NQADataset()
    # qadataset = list(qadataset.load_data())
    # dataset.segmentation(qadataset[:60000],windows,step,output_file='app/data/segmented_data_0.h5')
    # dataset.segmentation(qadataset[60000:130000],windows,step,output_file='app/data/segmented_data_1.h5')
    # dataset.segmentation(qadataset[130000:210000],windows,step,output_file='app/data/segmented_data_2.h5')
    # dataset.segmentation(qadataset[210000:],windows,step,output_file='app/data/segmented_data_3.h5')

    
    data = DocumentDatasets()
    num_samples = config['data_config']['num_doc']
    random_sequence = torch.randperm(len(data))
    random_nnn = random_sequence[:num_samples]
    random_nnn=sorted(random_nnn)
    data=data[random_nnn]
    print(data.shape)
    retriever=DOC_Retriever(load_data_feature=False)
    retriever.build_index(data,'/home/devil/workspace/nlg_progress/backend/app/data/vecs_reduced.pt')

    
    torch.save(data,'/home/devil/workspace/nlg_progress/backend/app/data/data_reduced.pt')
    print('saved data_reduced.pt')

    # vecs=torch.load('/home/devil/workspace/nlg_progress/backend/app/data/key_feature.pt')
    # print(vecs.shape)
    
    # vecs=vecs[random_nnn]
    # torch.save(vecs,'/home/devil/workspace/nlg_progress/backend/app/data/vecs_reduced.pt')
    # print('saved')
  