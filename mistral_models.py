import yaml
import os
from mistralai.client import MistralClient
from mistralai.models.chat_completion import ChatMessage

from utils.utils import process_response
from utils.utils import check_dict_keys_condition

with open('params.yaml', 'r') as file:
    config = yaml.safe_load(file)

MODEL = config['MODEL']
TEMPERATURE = config['TEMPERATURE']
CANDIDATE_TEMPERATURE = config['CANDIDATE_TEMPERATURE']
QUERY_PROMPT_PATH = config['QUERY_PROMPT_PATH']
MATCH_CRITERIA = config['MATCH_CRITERIA'] 
MAX_CANDIDATE_RESPONSES = config['MAX_CANDIDATE_RESPONSES'] 


def perform_mistral_response(client,prompt_var_list,temperature,prompt_path):
    # variable1 and variable2 in general are the first and second input passed to the prompt
    # this can be query and external evidence or original response and candidate response
    #Read the prompts from txt files
    with open(prompt_path, 'r') as file:
        file_content = file.read()
    
    message = [
        ChatMessage(role="user", content=file_content.format(*prompt_var_list))
    ]

    chat_completion = client.chat(
        model=MODEL,
        temperature=temperature,
        messages=message
    )

    # print(chat_completion.choices[0].message)
    # print("The token usage: ", chat_completion.usage)
    return chat_completion.choices[0].message.content.strip()

def perform_qa_task(client,query,external_evidence,WORKFLOW_RUN_COUNT):
    
    prompt_var_list = [query]
    final_response = perform_mistral_response(client,prompt_var_list,CANDIDATE_TEMPERATURE,QUERY_PROMPT_PATH)
    print("Final response is: ", final_response)
    return final_response

def start_mistral_api_model_response(query,external_evidence,WORKFLOW_RUN_COUNT):
    print("Mistral model response process starts ", query)
    client = MistralClient(api_key=os.environ["MISTRAL_API_KEY"])
    result = perform_qa_task(client,query,external_evidence,WORKFLOW_RUN_COUNT)
    if result is None:
        print("Error: Result is None")
    else:
        final_response = result
        return final_response
