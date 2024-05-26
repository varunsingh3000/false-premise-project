import yaml
import boto3
import json

from utils.utils import process_response
from utils.utils import check_dict_keys_condition

with open('params.yaml', 'r') as file:
    config = yaml.safe_load(file)

MODEL = config['MODEL']
TEMPERATURE = config['TEMPERATURE']
CANDIDATE_TEMPERATURE = config['CANDIDATE_TEMPERATURE']
QUERY_PROMPT_PATH = config['LLAMA_QUERY_PROMPT_PATH']
MATCH_CRITERIA = config['MATCH_CRITERIA'] 
MAX_CANDIDATE_RESPONSES = config['MAX_CANDIDATE_RESPONSES'] 

# call the meta llama api
def perform_llama_response(client,prompt_var_list,temperature,prompt_path):
    
    with open(prompt_path, 'r') as file:
        file_content = file.read()
    
    message = json.dumps({
    "prompt": file_content.format(*prompt_var_list),   #file_content.format(variable1,variable2)
    "temperature": temperature
    })

    accept = 'application/json'
    contentType = 'application/json'

    response = client.invoke_model(body=message, modelId=MODEL, accept=accept, contentType=contentType)

    chat_completion = json.loads(response.get("body").read())

    # print(chat_completion.get("generation"))
    # print("The token usage: ", chat_completion.usage)
    return chat_completion.get("generation").strip()

def perform_qa_task(client,query,external_evidence,WORKFLOW_RUN_COUNT):
    
    prompt_var_list = [query]
    final_response = perform_llama_response(client,prompt_var_list,CANDIDATE_TEMPERATURE,QUERY_PROMPT_PATH)
    print("Final response is: ", final_response)
    return final_response

def start_meta_api_model_response(query,external_evidence,WORKFLOW_RUN_COUNT):
    print("Meta Llama2 model response process starts ", query)
    client = boto3.client(service_name='bedrock-runtime', region_name='us-east-1')
    
    result = perform_qa_task(client,query,external_evidence,WORKFLOW_RUN_COUNT)
    # print("Meta Llama2 model response process ends")
    if result is None:
        print("Error: Result is None")
    else:
        final_response = result
        return final_response
