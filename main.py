import sys
sys.path.append('../..')
sys.path.append("app/lib/DocBuilder/")
from DocBuilder.Retriever_k_means import cluster_builder, doc_retriever
from DocBuilder.utils import top_k_sparse, inner, unbind_sparse, Masking, tensor_retuen_type
from DocBuilder.LexMAE import lex_encoder,lex_decoder, lex_retriever
import dataset
import time,datetime
import h5py
import torch
from torch import Tensor
from torch.nn import functional as F
from torch.utils.data import DataLoader
import multiprocessing
from functools import partial
# from contriver import  DOC_Retriever,Contriever

from tqdm import tqdm
import random
import yaml,sys,os
from fintune_contriver import NQADataset

with open('config.yaml', 'r') as yamlfile:
    config = yaml.safe_load(yamlfile)

seed = config['seed']
# torch.manual_seed(seed)
# random.seed(seed)

if __name__ == '__main__':
    windows=config['data_config']['windows']
    step=config['data_config']['step']

    if len(sys.argv) < 2:
        print(f"please give the parameter for action: (segment / save_embed / doc_build) ")
        exit()
    elif sys.argv[1]=="segment": # 1hr

        manager = multiprocessing.Manager()
        shared_dict = manager.dict()
        shared_int = multiprocessing.Value("i", 0)  # "i"表示整数类型
        lock=manager.Lock()
        qadataset = dataset.NQADataset(data_path='data/v1.0-simplified_simplified-nq-train.jsonl',num_samples=None)
        qadataset = list(qadataset.load_data())
        print("Dataset Loaded!!")

        Cor_num=multiprocessing.cpu_count()
        datastep=len(qadataset)//Cor_num+1

        multi_processing = []
        for i in range(Cor_num):
            segment = qadataset[i * datastep:(i + 1) * datastep]
            p = multiprocessing.Process(
                target=partial(dataset.segmentation, shared_dict, lock, shared_int),
                args=(segment, windows, step, f'data/segment/segmented_data_{i}.h5')
            )
            multi_processing.append(p)
            p.start()

        # Add join with timeout
        for p in multi_processing:
            p.join()


    elif sys.argv[1]=="Train_Retriever":

        
        def collate(batch):
            train_x = torch.stack(batch)
            
            train_x = torch.cat([torch.ones([len(train_x),1], dtype=torch.long)*cls, train_x, torch.ones([len(train_x),1], dtype=torch.long)*eos], dim=1)#(B,256)

            targets = train_x
            bar_x = Masking(train_x, 0.3, enc.tokenizer)
            tilde_x = Masking(train_x, 0.3, enc.tokenizer, bar_x.masks)
            
            return targets, bar_x, tilde_x
        device='cuda'
        enc=lex_encoder()
        dec=lex_decoder()
        cls = enc.tokenizer.cls_token_id
        eos = enc.tokenizer.pad_token_id
        enc.train()
        dec.train()
        enc.to(device)
        dec.to(device)
        # Define checkpoint path
        checkpoint_path = 'save/LEX_MAE_retriever.pt'
        load_path = 'save/LEX_MAE_retriever_loss_7.1591.pt'

            
        if os.path.isfile(load_path):
            checkpoint = torch.load(load_path,map_location='cpu')
            enc.load_state_dict(checkpoint["enc_model_state_dict"])
            dec.load_state_dict(checkpoint["dec_model_state_dict"])
            best_loss = checkpoint['loss']
            print('load from checkpoint')
        else:
            print('from scratch')
            start_epoch = 0
            best_loss = 13

        if config['lex']['share_param']:
            print('share param of enc')
            dec.model.cls = enc.model.cls
            dec.model.bert.embeddings = enc.model.bert.embeddings
        
        optimizer=torch.optim.AdamW(
                params=list(enc.parameters())+list(dec.parameters()),
                lr=config['lex']['pre_lr'],
                betas=config['lex']['betas'],
                weight_decay=config['lex']['weight_decay'])
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)

        data = dataset.DocumentDatasets('data/segment/segmented_data_', 12)
        # print(data.shape)
        # print(data[:4])
        # print(enc.tokenizer.batch_decode(data[:4]))
        # small_data=data[:1000].repeat([100,1])
        dataloader=DataLoader(data, batch_size=128, shuffle=True, num_workers=12, collate_fn=collate)

        # Define checkpoint frequency (e.g., save every 5 epochs)
        checkpoint_freq = 1
        s_time =time.time()
        snow = datetime.datetime.now().strftime("%m_%d_%H_%M").strip()
        ma_loss=10
       
        for epoch in range(0,config['train_config']['max_epoch']):

            bar = tqdm(dataloader, ncols=0)
            count=0
            for targets, bar_x, tilde_x in bar:
                optimizer.zero_grad()
                count+=1
                targets, bar_x, tilde_x= map(lambda x:x.to(device), [targets, bar_x, tilde_x])
                targets:Tensor
                bar_x:tensor_retuen_type
                tilde_x:tensor_retuen_type
                
                enc_logits, a, b=enc.forward(bar_x)
                dec_logits=dec.forward(tilde_x, b=b)
                
                # print(targets.shape) #(B, N)
                enc_loss=-torch.log_softmax(enc_logits, dim=-1)[torch.arange(targets.shape[0])[:,None], torch.arange(targets.shape[1])[None,:], targets] #(B,N)
                dec_loss=-torch.log_softmax(dec_logits, dim=-1)[torch.arange(targets.shape[0])[:,None], torch.arange(targets.shape[1])[None,:], targets] #(B,N)
                enc_loss = (enc_loss*(1-bar_x.masks)).sum()/((1-bar_x.masks).sum()+1e-4)
                dec_loss = (dec_loss*(1-tilde_x.masks)).sum()/((1-bar_x.masks).sum()+1e-4)
                loss = enc_loss+dec_loss
                loss = loss.mean()
                loss.backward()
                optimizer.step()

                enc_pred = enc_logits.max(dim=-1).indices
                enc_acc=((enc_pred==targets).float()*(1-bar_x.masks)).sum()/((1-bar_x.masks).sum())
                dec_pred = dec_logits.max(dim=-1).indices
                dec_acc=((dec_pred==targets).float()*(1-bar_x.masks)).sum()/((1-bar_x.masks).sum())

                bar.set_description_str(f"Loss: {ma_loss:.2f}/{enc_loss:.2f}/{dec_loss:.2f}, Acc:{enc_acc:.2f}/{dec_acc:.2f}  Best Loss: {best_loss:.2f}, Save time: {snow}")
                if not torch.isnan(loss):
                    ma_loss=0.99*ma_loss+0.01*loss.item()

                if ma_loss < best_loss and count>5000:
                    count=0
                    best_loss=ma_loss
                    snow = datetime.datetime.now().strftime("%m_%d_%H_%M").strip()
                    
                    torch.save({
                    'enc_model_state_dict': enc.state_dict(),
                    'dec_model_state_dict': dec.state_dict(),
                    # 'optimizer_state_dict': optimizer.state_dict(),
                    'loss': best_loss,
                    # Add any other information you want to save
                    }, checkpoint_path.replace(".pt",f"_loss_{best_loss:.4f}.pt"))
            scheduler.step()

    elif sys.argv[1]=="save_embed": # 12hr
        device='cuda'
        data = dataset.DocumentDatasets('data/segment/segmented_data_', 12)

        num_samples = config['data_config']['num_doc']
        
        # with reduced documents
        print('randperm...')
        random_sequence = torch.randperm(len(data), device=device)
        print('sliceing...')
        random_select = random_sequence[:num_samples]
        print('sorting...')
        random_select=torch.sort(random_select).values.cpu().numpy()
        print('Loading...')
        data=data[random_select]
        
        
        print(data.shape)

        lex_MAE_retriver=lex_retriever()
        lex_MAE_retriver.to(device)
        load_path = 'save/LEX_MAE_retriever895.pt'
        lex_MAE_retriver.model.load_state_dict(torch.load(load_path, map_location='cpu')['enc_model_state_dict'])
        print('load weight from',load_path)
        
        feature = lex_MAE_retriver.get_feature(data, 256)
        # feature = torch.nested.nested_tensor(feature) # not supported
        torch.save(feature, f'data/vecs_reduced_{num_samples}.pt')
        print('saved vecs_reduced.pt')

        torch.save(data, f'data/data_reduced_{num_samples}.pt')
        print('saved data_reduced.pt')
        
        # print(torch.load(f'data/vecs_reduced_{num_samples}.pt'))
        # print(torch.load(f'data/data_reduced_{num_samples}.pt'))
    elif sys.argv[1]=="doc_build":
        cluster_config=config["cluster_config"]
        data=torch.load('data/vecs_reduced_1000000.pt', mmap=True) ## shape:(N,d)
        print('converting...')
        runer = unbind_sparse(data)
        del data
        data = runer.run()
        del runer

        ## Train
        print(len(data))
        print(data[:2])
        cluster = cluster_builder(k=cluster_config["k"])
        cluster_ids_x, centers = cluster.train(data, epoch=50, bs = cluster_config['bs'], tol=cluster_config['tol'], lr=cluster_config['lr'])
        del data
        cluster.build()
        name = cluster.save()
        cluster.load(name)
    elif sys.argv[1]=="test":
        cluster_config=config["cluster_config"]
        cluster = cluster_builder(k=cluster_config["k"])
        cluster.load('04_23_04_32')

        lex_MAE_retriver=lex_retriever()
        lex_MAE_retriver.to('cpu')
        lex_MAE_retriver.model.load_state_dict(torch.load('save/LEX_MAE_retriever838.pt', map_location='cpu')['enc_model_state_dict'])

        data=torch.load('data/data_reduced_5000000.pt') ## shape:(N,d)
        retriever = doc_retriever(model = lex_MAE_retriver, data = data, cluster=cluster)
        # retriever.to('cuda')
        # vec = torch.load('data/vecs_reduced_5000000.pt') ## shape:(N,d)

        # for i in tqdm(range(100000)):
        #     query = vec[i].to_dense()
        #     seg, emb = retriever.retrieve(query, 100, 50)
        #     print(inner(query[None,:], emb[0]))
            
        while True:
            user= input('user:')
            seg, emb = retriever.retrieve(user)
            print(seg.shape)
            print(emb.shape)
            print(retriever.tokenizer.batch_decode(seg[0]))
    elif sys.argv[1]=="show_dataset":
        data_path='app/data/cleandata.pt'
        dataset=NQADataset(data_path=data_path)
        for i in range(50,60):
            print(dataset[i])
        

        pass
    
    else:
        raise KeyError()


