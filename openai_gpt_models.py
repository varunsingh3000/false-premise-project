import yaml
import os 
from openai import OpenAI

with open('params.yaml', 'r') as file:
    config = yaml.safe_load(file)

MODEL = config['MODEL']
TEMPERATURE = config['TEMPERATURE']
CANDIDATE_TEMPERATURE = config['CANDIDATE_TEMPERATURE']
QUERY_PROMPT_PATH = config['QUERY_PROMPT_PATH']
MATCH_CRITERIA = config['MATCH_CRITERIA'] 
MAX_CANDIDATE_RESPONSES = config['MAX_CANDIDATE_RESPONSES'] 

# call the openai api with either gpt 3.5 or gpt 4 latest model
def perform_gpt_response(client,prompt_var_list,temperature,prompt_path):
    
    with open(prompt_path, 'r') as file:
        file_content = file.read()

    message = [
        {
            "role": "system",
            "content": file_content.format(*prompt_var_list) 
        }
    ]
    chat_completion = client.chat.completions.create(
        messages=message,
        model=MODEL,
        temperature=temperature
        )
    
    # print(chat_completion.choices[0].message)
    # print("The token usage: ", chat_completion.usage)
    return chat_completion.choices[0].message.content.strip()

# function to call the LLM to generate multiple responses which will then be compared with the original response
# using their core concepts
def perform_qa_task(client,query,external_evidence,WORKFLOW_RUN_COUNT):
    
    prompt_var_list = [query]
    final_response = perform_gpt_response(client,prompt_var_list,CANDIDATE_TEMPERATURE,QUERY_PROMPT_PATH)
    print("Final response is: ", final_response)
    return final_response


def start_openai_api_model_response(query,external_evidence,WORKFLOW_RUN_COUNT):
    print("OpenAI model response process starts: ", query)
    client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY")) # defaults to os.environ.get("OPENAI_API_KEY")
    result = perform_qa_task(client,query,external_evidence,WORKFLOW_RUN_COUNT)
    # print("OpenAI model response process ends")
    if result is None:
        print("Error: Result is None")
    else:
        final_response = result
        return final_response
