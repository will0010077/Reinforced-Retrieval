from openai import OpenAI
import openai,time,os,threading as th,json
import yaml,torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import multiprocessing as mp
class openai_GPT:
    def __init__(self,model,api_key):
        self.model_name=model
        self.api_key=api_key
        self.openai_client = OpenAI(
            api_key=self.api_key,)
        self.gpt_lock=th.Lock()
        self.APIValidation=False
        self.complete_tokens=0
        self.prompt_tokens=0
        self.re_gen_times=10

    def  ChatGPT_reply(self,system_prompt='',user_prompt='',input_text='',temperature=0,max_tokens=4096,pb=None,lock=None,assit_prompt=""):
        if input_text:
            for _ in range(self.re_gen_times):
                try:
                    response=self.openai_client.chat.completions.create(
                    model=self.model_name,
                    messages= [
                        {"role": "system", "content":system_prompt},
                        {"role": "user", "content":f"{user_prompt}\n {input_text}"},
                        {"role": "assistant", "content": f"{assit_prompt}"}
                        ],

                    temperature=temperature,
                    max_tokens=max_tokens,
                    response_format={ "type": "json_object" }
                    )
                    if dict(response).get('choices',None) is not None:
                        self.APIValidation=True
                        try:
                            claim_text= json.loads(response.choices[0].message.content)
                            return claim_text,response.usage.completion_tokens,response.usage.prompt_tokens
                        except json.JSONDecodeError as e:
                            logger.error(f"{th.current_thread().name}JSON decoding failed: {e} {user_prompt} : {input_text} {response}")
                            continue
                        except Exception as e:
                            logger.error(f"{th.current_thread().name}Unexpected error: {e} {user_prompt} : {input_text} {response}")
                            continue

                except openai.APIStatusError as e:
                    logger.error(f"{th.current_thread().name} code : {e.status_code}_{e}")
                    continue
        else:
            logger.debug("Text input empty, please check your input text")
            return 1

    ## TODO: Add more models LLAMA2 , BREEZE , LLAMA3

## LLAMA2
def LlamaChatCompletion(model_name, prompt, max_tokens):

    model_name = "daryl149/llama-2-7b-chat-hf"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name).to("cuda")

    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to("cuda")
    outputs = model.generate(input_ids=input_ids,
                            max_new_tokens=max_tokens,return_dict_in_generate=True, output_scores=True, output_hidden_states=True)

    tokenizer.batch_decode(outputs, skip_special_tokens=True)
    # pdb.set_trace()
    return outputs

## LLAMA3

def LLAMA3_message(prompt):
    messages = [
        {"role": "system", "content": f"{prompt['system_prompt']},{prompt['assistant_prompt']}"},
        {"role": "user", "content": prompt["user_prompt"]},
    ]
    return messages

def LLAMA3_response(model,tokenizer,messages_batch)->str: ## Change to batch
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # f'''<|begin_of_text|><|start_header_id|>{prompt["system_prompt"]}<|end_header_id|>

    #     {{ system_prompt }}<|eot_id|><|start_header_id|>{prompt["user_prompt"]}<|end_header_id|>

    #     {{ user_msg_1 }}<|eot_id|><|start_header_id|>{prompt["assistant_prompt"]}<|end_header_id|>

    #     {{ model_answer_1 }}<|eot_id|>'''

      # 确保所有输入都是一个批次

    batch_input_ids = []
    for messages in messages_batch:
        input_ids = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            return_tensors="pt",
            padding=True,
        )
        # print(input_ids.squeeze(0).shape)
        batch_input_ids.append(input_ids)

    token=tokenizer(batch_input_ids, padding=True, return_tensors="pt",max_length=4096,truncation=True).to(device)

    terminators = [
        tokenizer.eos_token_id,
        tokenizer.convert_tokens_to_ids("<|eot_id|>")
    ]

    outputs = model.generate(
        **token,
        max_new_tokens=4096,
        eos_token_id=terminators,
        pad_token_id=tokenizer.eos_token_id,
        do_sample=True,
        temperature=0.6,
        top_p=0.9,
    )
    # 解析生成的响应
    responses = []
    for i, output in enumerate(outputs):
        response = output[token.input_ids[i].shape[-1]:]
        res = tokenizer.decode(response, skip_special_tokens=True)
        responses.append(res)
    return responses

class API:
    def __init__(self,api_name,api_key,parser,system_prompt='',user_prompt='',input_text='',assit_prompt=''):
        '''
        This is API response Function
        Need input api_name and api_key
        Then system_prompt for system character
        user_promt+input_text for input
        assis_prompt is the assistent character
        return
        1. claim text (response text):str
        2. complete tokens : output tokens
        3. prompt token : input tokens
        '''
        self.api_name=api_name
        self.api_key=api_key
        self.parser=parser
        self.re_gen_times=5
        self.system_prompt=system_prompt
        self.user_prompt=user_prompt
        self.input_text=input_text
        self.assit_prompt=assit_prompt
        self.api=OpenAI(api_key=api_key)

    def ans_parser(self,result):
        final_result={}
        if self.parser=="similarity":
            simi=result.get("similarity",None)
            if simi is not None:
                final_result["similarity"]=simi
            else:
                logger.info(f"{self.api_name} {self.parser} Fail : {result}")
                return final_result
        elif self.parser=="confidence":
            ans=result.get("Answer",None)
            conf=result.get("Confidence",None)
            explain=result.get("Explanation",None)
            if ans is not None and conf is not None:
                final_result["Answer"]=ans
                final_result["Confidence"]=conf
                final_result["Explanation"]=explain
            else:
                logger.info(f"{self.api_name} {self.parser} Fail : {result}")
                return final_result

        elif self.parser=="multi_step_confidence":
            FFinal=result.get("Final Answer and Overall Confidence",None)
            if FFinal is not None:
                multi_result={}
                for k,v in result.items():
                    if f"Step" in k:
                        multi_result[k]=v
                    elif f"Confidence" in k and k!="Final Answer and Overall Confidence":
                        multi_result[k]=v

                final_result["Step_result"]=multi_result
                final_result["Confidence"]=FFinal.get("Confidence",None)
                final_result["Answer"]=FFinal.get("Final Answer",None)
            else:
                logger.info(f"{self.api_name} {self.parser} Fail : {result}")
                return final_result

        return final_result
    def generate(self):
        if self.api_name=="gpt-3.5-turbo-0125":
            self.api_key=self.api_key['openai']['api_key']

            for _ in range(self.re_gen_times):
                result,indi_complete_tokens,indi_Prompt_tokens=openai_GPT(self.api_name,self.api_key).ChatGPT_reply(system_prompt=self.system_prompt,user_prompt=self.user_prompt,input_text=self.input_text,assit_prompt=self.assit_prompt)
                final_res=self.ans_parser(result)
                if final_res:
                    return final_res,indi_complete_tokens,indi_Prompt_tokens
                else:
                    continue
            else:
                logger.error(f"{mp.current_process().name} generate fail exit() : {result}")
                return {},0,0

        elif self.api_name=="llama2":
            pass



if __name__=="__main__":
    pass

