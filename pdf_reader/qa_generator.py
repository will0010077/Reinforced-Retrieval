from transformers import AutoTokenizer, AutoModelForCausalLM
from torch.utils.data import  DataLoader
import torch,json,re,os
from tqdm import tqdm
from LLM_API import LLAMA3_response,LLAMA3_message

device = 'cuda' if torch.cuda.is_available() else 'cpu'
model_id = "meta-llama/Meta-Llama-3-8B-Instruct"

tokenizer = AutoTokenizer.from_pretrained(model_id,token="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",padding_side='left')
model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        token="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"
    )
model.to(device)
tokenizer.pad_token = tokenizer.eos_token

def llama_parser(file_name,doc,article:list)->list:
    # 定义正则表达式模式
    # qa_pattern_1 = re.compile(r'\*\*Question \d+\*\*\n(.*prompt={}?)\n\*\*Answer\*\*: (.*?)\n\*\*Confidence\*\*: (0\.\d+)', re.DOTALL)
    # qa_pattern_2 = re.compile(r'{Question: (.*?)\nAnswer: (.*?)\nConfidence: (0\.\d+)}',re.DOTALL)
    # # 使用正则表达式提取（问题，答案，置信度）对
    # matches = qa_pattern_1.findall(article)
    qa_confidence_pairs=[]
    Fail_count=0
    for i in article:
        try:
            match_result=json.loads(i)
        # 将匹配结果转换为所需格式的列表
            qa_confidence_pairs+=[{"File_name":file_name,"Document":doc,"Question":vv["Question"].strip(), "Answer":vv["Answer"].strip(), "Confidence":float(vv["Confidence"])} for vv in match_result]
        except:
            Fail_count+=1
            print(f"Fail Count {Fail_count}")

    return qa_confidence_pairs


def prompter(Document:str,question_count:int)->str:
    response_format="{Question : [Your Question here],Answer: [Your Answer here],Confidence: [Your Confidence here]}"
    confidence_define_prompt="Note: The confidence indicates how likely you think your Answer is true and correct,from 0.00 (worst) to 1.00 (best)"
    return {'system_prompt':"You are a long form Question and Answer generator,Please Base on the given Document ask the Question and provide the answer to the question as much as possible, also provide the confidence Score to the answer in json","user_prompt":f"Please provide {question_count} pairs of very long Question and Answer as much as possible, also provide the confidence Score to the answer in json \nOnly give me the reply according to response format, don't give me any other words. \nDocument:{Document}\n\n response format:[{response_format}*{question_count}]","assistant_prompt":f'{confidence_define_prompt}'}

def chunk_document(batch_data):
    TOKEN_LIMIT = 256

    # 使用tokenizer将文章转换为tokens
    # tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    tokens = tokenizer(batch_data['Document'], return_tensors='pt')['input_ids'][0]

    # 将tokens切分成固定大小的块
    chunks = [tokens[i:i + TOKEN_LIMIT] for i in range(0, len(tokens), TOKEN_LIMIT)]

    # 将每个块转换回字符串
    chunks_text = [tokenizer.decode(chunk, skip_special_tokens=True) for chunk in chunks]

    # # 打印每个块
    # for i, chunk in enumerate(chunks_text):
    #     print(f"Chunk {i+1}:\n{chunk}\n")
    print(f"{batch_data['File_name']} Document Chunk : {len(chunks_text)}")
    return chunks_text


def collet_fn(batch):
    res=[LLAMA3_message(prompter(i,2)) for i in batch]

    return res,batch

def main(file_path):

    Save_path=f'smart_factory_qapairs2.jsonl'

    with open(file_path,encoding='unicode_escape') as f:
            data=json.load(f)

    if os.path.exists(Save_path):
        with open(Save_path,encoding='unicode_escape') as f:
            result=json.load(f)
        file_name_list=[i['File_name'] for i in result]
    else:
        result=[]
        file_name_list=[]

    for v in (p:=tqdm(data)):

        if v['File_name'] in file_name_list:
            continue
        doc=chunk_document(v)
        File_Loader=DataLoader(doc,batch_size=5,shuffle=True,collate_fn=collet_fn)

        for message,document in (q:=tqdm(File_Loader)):

            q.set_postfix_str(f"Start to generate")
            for times in range(5):
                qa_pairs_in_text=LLAMA3_response(model,tokenizer,message)
                qa_pairs=llama_parser(v['File_name'],document,qa_pairs_in_text)
                q.set_postfix_str(f"QA generator success-> Parser Success {len(qa_pairs)}")
                if qa_pairs:
                    q.set_description_str(f"Parser {len(qa_pairs)}")
                    result+=qa_pairs
                else:
                    q.set_description_str(f"Parser Fail")

        with open(Save_path,'a+',encoding='unicode_escape') as f:
            json.dump(result,f)

        p.set_description_str(f"Write Success")

if __name__=="__main__":
    torch.cuda.empty_cache()
    main('smart_factory.jsonl')
    # with open('smart_factory_qapairs.jsonl',encoding='unicode_escape') as f:
    #         data=json.load(f)

    # print(len(data))