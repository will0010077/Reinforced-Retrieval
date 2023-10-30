import openai
import json
import logging

from app.lib import mongodb

def merge_chunks(array, chunk_size: int):
    merged_list = []

    # Iterate over the array in chunks of chunk_size
    for i in range(0, len(array), chunk_size):
        merged_element = '\n'.join(array[i:i + chunk_size])
        merged_list.append(merged_element)

    return merged_list

def query_and_merge_questions(file_path: str, chunk_size: int):
    try:
        with open(file_path, 'r') as file:
            group = []
            for line in file:
                group.append(line.strip()) 

        return merge_chunks(group, chunk_size)

    except FileNotFoundError:
        # Handle the case when the file is not found.
        logging.error(f"File not found: {file_path}")
    except Exception as e:
        # Handle the case when the file is not found.
        logging.error(f"Error occurred: {e}")


# Return the stripped line from the file, which represents the API key
def get_api_key():
    with open("app/data/key.txt", "r") as key_file:
        api_key = key_file.readline().strip()

    return api_key

def compare_text(text_to_compare: str):
    openai.api_key = get_api_key()

    logging.info(f'query question: {text_to_compare}')

    # Initialize the highest similarity score and the similar text.
    highest_similarity_score = 0
    most_similar_text = ""

    prompt = '''
    以上問題與下列哪個相似，請以餘弦相似度來計算，回答格式如下(以 json 格式回答，僅回答最相似的文字，不需要其他格式)
    {
        "Text_to_Compare": str
        "Similarity_Text": str
        "Similarity": float
    }

    '''
    file_path = 'app/data/question_list.txt'
    question_array = query_and_merge_questions(file_path, 500)

    # Compare each text element with the text to be compared.
    for text in question_array:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo-16k",
            messages=[
                {
                    "role": "user", 
                    "content": 'Text to compare: '+ text_to_compare + "\n" + prompt + text
                }
            ]
        )

        response_message = response['choices'][0]["message"]["content"]
        parsed_data = json.loads(response_message)

        # Update the highest similarity score and the similar text.
        if float(parsed_data["Similarity"]) > highest_similarity_score:
            if parsed_data["Text_to_Compare"] == text_to_compare:
                highest_similarity_score = float(parsed_data["Similarity"])
                most_similar_text = parsed_data['Similarity_Text']

    if(highest_similarity_score>0.5):
        question = most_similar_text
        logging.info(f'similar text: {question}')
        result = mongodb.query_mongodb_question(question)
        if result is None:
            logging.error("No matching text found in db.")

            response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo-16k",
            messages=[
                {
                    "role": "user", 
                    "content": text_to_compare + '請用繁體中文回答'
                }
            ]
            )

            response_message = response['choices'][0]["message"]["content"]

            return {'message': f'我不清楚你問甚麼，但我可以幫你向我的前輩詢問：\n {response_message}'}
        else:
            logging.info(f'similar text: {question}')
            return {'message': result['answer']}
    else:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo-16k",
            messages=[
                {
                    "role": "user", 
                    "content": text_to_compare + '請用繁體中文回答'
                }
            ]
        )

        response_message = response['choices'][0]["message"]["content"]

        return {'message': f'我不清楚你問甚麼，但我可以幫你向我的前輩詢問：\n {response_message}'}

if __name__ == "__main__":
    question = input('me > ')
    answer = compare_text(question)
    print(f'Anser > {answer}' )

