
import pynvml
import torch,os,glob,re,json
import fitz
from io import BytesIO
from PIL import Image
from transformers import LlamaTokenizer
from transformers import AutoTokenizer, AutoModelForCausalLM


def replace_newlines(text):
    # First replace all occurrences of "/n" with a unique placeholder
    placeholder = "__TEMP_NEWLINE__"
    text = text.replace(". \n", placeholder)

    # Use regex to replace "\n" that are not preceded by "./"
    text = re.sub(r'\n', ' ', text)

    # Finally, replace the placeholder back with "/n"
    text = text.replace(placeholder, ".\n")

    return text


def pdf_reader(pdf_path):

        pdf_file_path=os.path.join(pdf_path)
        # project_name=pdf_path.split("\\")[-1]
        doc=fitz.open(pdf_file_path)
        text=''
        toc=doc.get_toc()
        # image_id=[]
        # img_list_total=set()
        for pg in range(len(doc)):
            page=doc.load_page(pg)
            text+=page.get_text("text")
            # img_list = doc.get_page_images(pg,full=True)
            # img_list_total|=set(img_list)
        # text = text.encode('utf-8').decode('utf-8')
        text = replace_newlines(text)

        doc.close()

        return text


if __name__=="__main__":
    datapath='2024_LLMRAG_smartfactory/'
    data_list=glob.glob(f'{datapath}*.pdf')
    res=[]
    for i in data_list:
        text= (pdf_reader(i))
        decoded_string = text.encode('unicode_escape').decode('unicode_escape')
        # print(decoded_string)
        res.append({
            "File_name":i,
            "Document":decoded_string,
            })
        print(i)

    with open(f'smart_factory.jsonl','w+',encoding='unicode_escape') as f:
        f.writelines(json.dumps(res,indent=4))

    with open(f'smart_factory.jsonl','r',encoding='unicode_escape') as f:
        data=json.load(f)

    print([i['Document']for i in data if i['File_name']=="2024_LLMRAG_smartfactory/TSMC_2021_sustainabilityReport_chinese_c-all.pdf"])
