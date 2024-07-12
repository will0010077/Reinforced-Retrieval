import json
import re

# 打開文件並讀取所有內容
with open('smart_factory_qapairs2.jsonl', 'r', encoding='unicode_escape') as f:
    content = f.read()

# 假設文件內容是一個 JSON 數組
# 去掉開頭的 '[' 和結尾的 ']'，然後分割成多個 JSON 字符串
json_strings = re.findall(r'\{.*?\}', content)

# 初始化一個列表來存儲解析出的 JSON 對象
data = []

# 集合來存儲已遇到的 (File_name, Question, Answer) 組合
seen = set()

# 解析每個 JSON 字符串並附加到列表中，過濾重複的 QA 對
for json_str in json_strings:
    try:
        json_obj = json.loads(json_str)
        qa_pair = (json_obj["File_name"],json_obj["Document"], json_obj["Question"], json_obj["Answer"])
        if qa_pair not in seen:
            seen.add(qa_pair)
            data.append(json_obj)
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON: {e}")
        continue

# 打印解析出的唯一 JSON 對象的數量
print(len(data))

# 將唯一的 JSON 對象寫入新的 JSON Lines 文件中
with open('unique_smart_factory_qapairs.jsonl', 'w', encoding='utf-8') as outfile:
    for json_obj in data:
        outfile.write(json.dumps(json_obj) + '\n')