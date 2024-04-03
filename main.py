import sys
sys.path.append('../..')
sys.path.append("app/lib/DocBuilder/")
from DocBuilder.Retriever_k_means import cluster_builder, cos_sim, doc_retriever
from LexMAE import lex_encoder,lex_decoder, lex_retriever, top_k_sparse
import dataset
import time,datetime
import h5py
import torch
from torch.nn import functional as F
import multiprocessing
from functools import partial
# from contriver import  DOC_Retriever,Contriever

from torch.utils.data import DataLoader
from tqdm import tqdm
import random
import yaml,sys,os
from fintune_contriver import NQADataset

with open('app/lib/config.yaml', 'r') as yamlfile:
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
    elif sys.argv[1]=="segment": # 3hr

        manager = multiprocessing.Manager()
        shared_dict = manager.dict()
        shared_int = multiprocessing.Value("i", 0)  # "i"表示整数类型
        lock=manager.Lock()
        qadataset = dataset.NQADataset(num_samples=None)
        qadataset = list(qadataset.load_data())
        print("Dataset Loaded!!")

        Cor_num=multiprocessing.cpu_count()//2
        datastep=len(qadataset)//Cor_num+1

        multi_processing = []
        for i in range(Cor_num):
            segment = qadataset[i * datastep:(i + 1) * datastep]
            p = multiprocessing.Process(
                target=partial(dataset.segmentation, shared_dict, lock, shared_int),
                args=(segment, windows, step, f'app/data/segmented_data_{i}.h5')
            )
            multi_processing.append(p)
            p.start()

        # Add join with timeout
        for p in multi_processing:
            p.join()

        # Ensure termination of processes
        # for p in multi_processing:
        #     if p.is_alive():
        #         p.terminate()


        ## 2024.03.19 Remove
        # multi_processing=[]
        # for i in range(Cor_num):
        #     segment = qadataset[i * datastep:(i + 1) * datastep]
        #     multi_processing.append(multiprocessing.Process(target=dataset.segmentation_para, args=(shared_set,lock,shared_int, segment,windows,step,f'app/data/segmented_data_{i}.h5')))
        # [p.start() for p in multi_processing]
        # [p.join() for p in multi_processing]

    elif sys.argv[1]=="Train_Retriever":

        device='cuda'
        enc=lex_encoder()
        dec=lex_decoder()
        cls = enc.tokenizer.cls_token_id
        eos = enc.tokenizer.pad_token_id
        enc.train()
        dec.train()
        enc.to(device)
        dec.to(device)
        data = dataset.DocumentDatasets()
        print(data.shape)
        # small_data=data[:1000].repeat([100,1])
        dataloader=DataLoader(data, batch_size=24, shuffle=True, num_workers=4)
        optimizer=torch.optim.AdamW(
            params=list(enc.parameters())+list(dec.parameters()),
            lr=1e-5,
            betas=config['train_config']['betas'],
            weight_decay=config['train_config']['weight_decay'])


        # Define checkpoint path
        checkpoint_path = 'app/save/LEX_MAE_retriever.pt'
        load_path = 'app/save/LEX_MAE_retriever_loss_5.8157.pt'
        def Masking(x,P,all_mask=None):
            x=x.clone()
            if all_mask is None:
                all_mask = torch.rand(x.shape, device=x.device) < P
            else:
                all_mask = all_mask.bool()+(torch.rand(x.shape, device=x.device) < P)
                
            all_mask[:,0], all_mask[:,-1] = 0, 0
            s = torch.rand(x.shape, device=x.device)
            mask_mask = (s<0.8) * all_mask
            rand_mask = ((s>0.8) * (s<0.9)) * all_mask
            
            x[mask_mask] = enc.tokenizer.mask_token_id
            x[rand_mask] = torch.randint(999, enc.tokenizer.vocab_size, size = x[rand_mask].shape, dtype=x.dtype,  device=x.device)

            return x, all_mask.float()

        if os.path.isfile(load_path):
            checkpoint = torch.load(load_path,map_location='cpu')
            enc.load_state_dict(checkpoint["enc_model_state_dict"])
            dec.load_state_dict(checkpoint["dec_model_state_dict"])
            # optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            # start_epoch = checkpoint['epoch'] -1
            best_loss = checkpoint['loss']
            print('load from checkpoint')
        else:
            print('from scratch')
            start_epoch = 0
            best_loss = 7

        # Define checkpoint frequency (e.g., save every 5 epochs)
        checkpoint_freq = 1
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)
        s_time =time.time()
        snow = datetime.datetime.now().strftime("%m_%d_%H_%M").strip()
        ma_loss=10
       
        for epoch in range(0,config['train_config']['max_epoch']):

            bar = tqdm(dataloader)
            count=0
            for train_x in bar:
                count+=1
                train_x = torch.cat([torch.ones([len(train_x),1], dtype=torch.long)*cls, train_x, torch.ones([len(train_x),1], dtype=torch.long)*eos], dim=1)#(B,256)
                train_x = train_x.to(device)
                targets = train_x
                bar_x, bar_mask=Masking(train_x, 0.3)
                tilde_x, tilde_mask=Masking(train_x, 0.3, bar_mask)
                bar_x = {"input_ids": bar_x}
                tilde_x= {"input_ids": tilde_x}
                enc_logits, a, b=enc.forward(bar_x)
                dec_logits=dec.forward(tilde_x, b=b)
                
                # print(targets.shape) #(B, N)
                enc_loss=-torch.log_softmax(enc_logits, dim=-1)[torch.arange(targets.shape[0])[:,None], torch.arange(targets.shape[1])[None,:], targets] #(B,N)
                dec_loss=-torch.log_softmax(dec_logits, dim=-1)[torch.arange(targets.shape[0])[:,None], torch.arange(targets.shape[1])[None,:], targets] #(B,N)
                enc_loss = (enc_loss*bar_mask).sum()/(bar_mask.sum()+1e-4)
                dec_loss = (dec_loss*tilde_mask).sum()/(tilde_mask.sum()+1e-4)
                loss = enc_loss+dec_loss
                loss = loss.mean()
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                enc_pred = enc_logits.max(dim=-1).indices
                enc_acc=((enc_pred==train_x).float()*bar_mask).sum()/(bar_mask.sum())
                dec_pred = dec_logits.max(dim=-1).indices
                dec_acc=((dec_pred==train_x).float()*bar_mask).sum()/(bar_mask.sum())

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
        data = dataset.DocumentDatasets()

        num_samples = config['data_config']['num_doc']
        random_sequence = torch.randperm(len(data))
        random_nnn = random_sequence[:num_samples]
        random_nnn=sorted(random_nnn)
        data=data[random_nnn]
        print(data.shape)

        lex_MAE_retriver=lex_retriever()
        lex_MAE_retriver.to(device)
        load_path = 'app/save/LEX_MAE_retriever838.pt'
        lex_MAE_retriver.model.load_state_dict(torch.load(load_path, map_location='cpu')['enc_model_state_dict'])
        print('load weight from',load_path)
        
        feature = lex_MAE_retriver.get_feature(data, 32)
        # feature = top_k_sparse(feature, 256)
        torch.save(feature,f'app/data/vecs_reduced_{num_samples}.pt')

        print('saved vecs_reduced.pt')

        torch.save(data,f'app/data/data_reduced_{num_samples}.pt')

        print('saved data_reduced.pt')
        # vecs=torch.load('/home/devil/workspace/nlg_progress/backend/app/data/key_feature.pt')
        # print(vecs.shape)
        # vecs=vecs[random_nnn]
        # torch.save(vecs,'/home/devil/workspace/nlg_progress/backend/app/data/vecs_reduced.pt')
        # print('saved')
    elif sys.argv[1]=="doc_build":
        cluster_config=config["cluster_config"]
        File_name="200000"
        data=torch.load('/home/devil/workspace/nlg_progress/backend/app/data/vecs_reduced_200000.pt') ## shape:(N,d)

        ## Trian
        print(data.shape)
        print(data[:5])
        cluster = cluster_builder(k=cluster_config["k"])
        cluster_ids_x, centers=cluster.train(data, epoch=10, bs = cluster_config['bs'], tol=cluster_config['tol'], lr=cluster_config['lr'])
        cluster.build()
        name = cluster.save()
        cluster.load(name)
    elif sys.argv[1]=="test":
        cluster_config=config["cluster_config"]
        cluster = cluster_builder(k=cluster_config["k"])
        cluster.load('03_31_20_22')

        lex_MAE_retriver=lex_retriever()
        lex_MAE_retriver.to('cpu')
        lex_MAE_retriver.model.load_state_dict(torch.load('app/save/LEX_MAE_retriever838.pt', map_location='cpu')['enc_model_state_dict'])

        data=torch.load('app/data/data_reduced_200000.pt') ## shape:(N,d)
        retriever = doc_retriever(model = lex_MAE_retriver, data = data, cluster=cluster)
        # retriever.to('cuda')


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
        


