import sys
sys.path.append("../../")

# Load model directly
import torch

from transformers import AutoTokenizer, AutoModel
import logging

# from app.lib import mongodb
# from app.lib import dataset
from nlg_progress.backend.app.lib import not_use_llama
# import llama
from torch.utils.data import DataLoader
from tqdm import tqdm


def cos_sim(a:torch.Tensor, b:torch.Tensor):
    return (a @ b.T)/(torch.norm(a,dim=1)[:,None]@torch.norm(b,dim=1)[None,:])


def check_Qmark(text:str):
    # Reduce sensitivity to question marks
    text=text.replace('？','?')
    while '??' in text:
        text=text.replace('??','?')
    if '?' not in text:
        text+='?'
    return text


class DOC_Retriever(torch.nn.Module):
    def __init__(self):
        super().__init__()

        self.model=Contriever()
        self.model.eval()

    @torch.inference_mode()
    def get_feature(self, texts)->torch.Tensor:
        '''
        texts: text list with shape:(N)\\
        return: tensor with shape:(N, 768)
        '''
        feature_list=[]
        text_list=[]

        dataloader = DataLoader(texts, batch_size=256, shuffle=False)
        for texts in (bar:=tqdm(dataloader,ncols=0)):
            bs=len(texts)

            feature  = self.model(texts)#(bs, d)
            feature_list.append(feature)
            text_list.extend(texts)


        feature_list=torch.cat(feature_list)


        return  feature_list#, text_list

    def build_index(self, texts:list[str]):
        self.Q  = texts#list(filter(lambda x:type(x)==str, texts))

        texts=[]
        for q in self.Q:
            text=check_Qmark(q)
            texts.append(text)
        self.feature= self.get_feature(texts)


    @torch.inference_mode()
    def retrieve(self, query:str, k=5, threshold=0.2):
        '''
        return k retrieved id and similarity
        '''

        query_feature = self.model(query)
        if len(query_feature.shape)==1:
            query_feature=query_feature[None,:]
        #cosine similarity
        sim = cos_sim(query_feature, self.feature)[0]

        #top-k vector and index
        v, id = torch.topk(sim, k, dim=0, largest=True)
        # scale to [0,100]
        scale=lambda x: max(0, min(100, x*100))
        return [{'question':self.Q[idx], 'score':scale(sim.item())} for idx, sim in zip(id[v>threshold], v[v>threshold])]


class Contriever(torch.nn.Module):
    def __init__(self):
        super(Contriever, self).__init__()
        self.model = AutoModel.from_pretrained("facebook/contriever")
        self.tokenizer = AutoTokenizer.from_pretrained("facebook/contriever")

    def forward(self, x):
        x=self.tokenizer(x, return_tensors='pt', padding=True ,truncation=True).to('cuda')
        y=self.model(**x)
        y=self.mean_pooling( y[0], x['attention_mask'])
        return y
    def mean_pooling(self,token_embeddings, mask):
        token_embeddings = token_embeddings.masked_fill(~mask[..., None].bool(), 0.)
        sentence_embeddings = token_embeddings.sum(dim=1) / mask.sum(dim=1)[..., None]
        return sentence_embeddings


def query_questions(file_path: str):
    try:
        with open(file_path, 'r') as file:
            group = []
            for line in file:
                group.append(line.strip())

        return group

    except FileNotFoundError:
        # Handle the case when the file is not found.
        logging.error(f"File not found: {file_path}")
    except Exception as e:
        # Handle the case when the file is not found.
        logging.error(f"Error occurred: {e}")



def compare_text(text_to_compare: str):

    logging.info(f'query question: {text_to_compare}')
    candidate =R.retrieve(query = check_Qmark(text_to_compare))
    if len(candidate)>0:
        # print(candidate)
        question = candidate[0]['question']
        score=candidate[0]['score']
        # result = mongodb.query_mongodb_question(question)
        print(f'find related document! score:{score}')
        result=not_use_llama.response('knowledge: '+question+'question: '+text_to_compare)
        if result is not None:
            logging.info(f'similar text: {question}')
            return  {'message' : result}
            return  {'message' : result['answer'],
                     'score' : candidate[0]['score']}
        else:
            logging.error("No matching text found in db.")
            return {'message': f'No matching text found in db. '+question, 'score': 0}

    return {'message': not_use_llama.response(text_to_compare)}

#file_path ='/home/devil/workspace/nlg_progress/backend/app/data/question_list.txt'

file_path = 'app/data/question_list.txt'
question_array = query_questions(file_path)

#initial a retriever
R = DOC_Retriever()

# use gpu(if available)
# only use cpu is fine.
device = 'cuda' if torch.cuda.is_available() else 'cpu'

R.to(device)
#pre-compute the vector of all questions
R.build_index(question_array)


if __name__=='__main__':
    answer=compare_text("when are hops added to the brewing process?")
    print(answer)
    # model = Contriever()

    # model.to('cuda')
    # # x=["when are hops added to the brewing process?","After mashing , the beer wort is boiled with hops ( and other flavourings if used ) in a large tank known as a \" copper \" or brew kettle – though historically the mash vessel was used and is still in some small breweries . The boiling process is where chemical reactions take place , including sterilization of the wort to remove unwanted bacteria , releasing of hop flavours , bitterness and aroma compounds through isomerization , stopping of enzymatic processes , precipitation of proteins , and concentration of the wort . Finally , the vapours produced during the boil volatilise off - flavours , including dimethyl sulfide precursors . The boil is conducted so that it is even and intense – a continuous \" rolling boil \" . The boil on average lasts between 45 and 90 minutes , depending on its intensity , the hop addition schedule , and volume of water the brewer expects to evaporate . At the end of the boil , solid particles in the hopped wort are separated out , usually in a vessel called a \" whirlpool \"."]
    # x = [
    # "Where was Marie Curie born?",
    # "Maria Sklodowska, later known as Marie Curie, was born on November 7, 1867.",
    # "Born in Paris on 15 May 1859, Pierre Curie was the son of Eugène Curie, a doctor of French Catholic origin from Alsace.",
    # "Marie Curie, was born in a shit.",
    # "Marie Curie",
    # ]


    # embeddings = model(x)


    # print(embeddings.shape)

    # print((embeddings@embeddings.T)[0])
