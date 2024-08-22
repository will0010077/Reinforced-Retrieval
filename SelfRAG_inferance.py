from transformers import AutoTokenizer, AutoModelForCausalLM
from vllm import LLM, SamplingParams
from RL.utils import LLMEnv_test
from DatasetLoader.collate_func import collate
from DatasetLoader.dataset import NQADataset
from DocBuilder.LexMAE import lex_retriever
from metric.reward import metric

from  tqdm import tqdm
import torch
import re
def format_prompt(input, response, paragraph=None):
    prompt = f"### Instruction:\nPlease provide a very short answer in no more than three words.\n\n### Input:\n{input}\n\n### Response:\n{response}"
    # prompt = f"### Instruction:\nPlease provide a very long answer in more than 50 words.\n\n### Input:\n{input}\n\n### Response:\n{response}"
    if paragraph is not None:
        prompt += "[Retrieval]<paragraph>{0}</paragraph>".format(paragraph)
    return prompt
class selfEnv(LLMEnv_test):
    def get_next_response(self, indices):
        d_t = self.ret.tokenizer.batch_decode([self.d_t[i] for i in indices], skip_special_tokens=True)
        all_pred = ["" for _ in range(self.batch_size)]
        done = [False]*self.batch_size
        ret = [False]*self.batch_size
        while not all(done):
            messages = [format_prompt(self.x[i], all_pred[i], d_t[i] if ret[i] else None) for  i in indices]
            ret = [False]*self.batch_size
            responses = self.LM.generate(messages, sampling_params, use_tqdm=False)
            preds = [pred.outputs[0].text for pred in responses]
            for i in range(self.batch_size):
                if "[Retrieval]" in preds[i]:
                    preds[i] = preds[i].split("[Retrieval]")[0]
                    ret[i] = True
                if not done[i]:
                    all_pred[i] += preds[i]
                if eos_token in preds[i]:
                    done[i]=True
   
        return all_pred
    def get_basic_response(self, x, y, d_t):
        return "not implemented!!!"
    
device = 'cuda:0'
model = LLM("selfrag/selfrag_llama2_7b", dtype="half")
sampling_params = SamplingParams(temperature=0.0, top_p=1.0, max_tokens=1024, skip_special_tokens=False, stop = "[Retrieval]", include_stop_str_in_output = True)
eos_token = model.get_tokenizer().eos_token
print('Initilize retriever')
lex_MAE_retriver=lex_retriever()
lex_MAE_retriver.to(device)
lex_MAE_retriver.eval()
lex_MAE_retriver.model.load_state_dict(torch.load('save/LEX_MAE_retriever904.pt', map_location='cpu')['enc_model_state_dict'], assign=False)


print('Loading dataset...')
data_path='data/TV_test.jsonl'
num_testing=200
dataset=NQADataset(data_path=data_path, use_doc=True, use_short=True, use_long=False, num_samples = num_testing+32)
bs = 1
env = selfEnv(dataset, model, lex_MAE_retriver, 3, collate(), batch_size = bs, shuffle = False, step_size = 256, eos_id=0)

query_1 = "who is playing the halftime show at super bowl 2016"
query_2 = "who won the 2017 sports personality of the year"
queries = [query_1, query_2]

# for a query that doesn't require retrieval
preds = model.generate([format_prompt(query, "") for query in queries], sampling_params)
for pred in preds:
  print("Model prediction: {0}".format(pred.outputs[0].text))


metric_c = metric()
q_list=[]
a_list=[]
true_list=[]
print("Starting reset...")
f = open("SelfRAG_result.txt", "a")

for i in tqdm(range(num_testing//bs)):
	[env.reset(j) for j in range(bs)]
	preds = env.get_next_response(range(bs))

	for j in range(bs):
		q_list.append(env.x[j])
		a_list.append(re.sub("","", re.sub("(\[.*\]|</s>|\.)","",preds[j])))
		true_list.append(env.ground_truth[j])


 # normalize
a_list = [a.lower() for a in a_list]
true_list = [t.lower() if isinstance(t, str) else [e.lower() for e in t] for t in true_list]

if isinstance(true_list[0],list):
    maching = [(a_list[i] in true_list[i]) or (any([true_list[i][j] in a_list[i] for j in range(len(true_list[i])) ]) ) for i in range(len(a_list))]
    print(f"Exact match1: {sum(maching)/len(maching)}")
    true_list = [t[0] for t in true_list]
bert = metric_c.Bert_score(a_list, true_list )
R_1, R_2, R_L = metric_c.ROUGE_score(a_list, true_list )
bleu = metric_c.BLEU_1_score(a_list, true_list)

for j in range(len(q_list)):
	f.write(
f'''Prompt: {q_list[j]}\nGround truth: {true_list[j]}
[{bleu[j]:.3f}, {R_1[j]:.3f}, {R_2[j]:.3f}, {R_L[j]:.3f}, {bert[j]:.3f}] Response: {a_list[j]}
''' +"="*80+"\n")
	
f.write(f"BLEU_1: {sum(bleu)/len(bleu)*100:05.2f}\n")
f.write(f"ROUGE-1: {sum(R_1)/len(R_1)*100:05.2f}\n")
f.write(f"ROUGE-2: {sum(R_2)/len(R_2)*100:05.2f}\n")
f.write(f"ROUGE-L: {sum(R_L)/len(R_L)*100:05.2f}\n")
f.write(f"BERT: {sum(bert)/len(bert)*100:05.2f}\n")