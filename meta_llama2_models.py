# Calling the OpenAI GPT models 3.5 and 4

import yaml
import os 
import boto3
import json
from rake_nltk import Rake

from utils.utils import process_response
from utils.utils import extract_value_from_single_key
from utils.utils import check_dict_keys_condition

with open('params.yaml', 'r') as file:
    config = yaml.safe_load(file)

MODEL = config['MODEL']
TEMPERATURE = config['TEMPERATURE']
CANDIDATE_TEMPERATURE = config['CANDIDATE_TEMPERATURE']
QUERY_PROMPT_PATH = config['LLAMA_QUERY_PROMPT_PATH']
BACKWARD_REASONING_PROMPT_PATH = config['LLAMA_BACKWARD_REASONING_PROMPT_PATH']

# call the meta llama api
def perform_llama_response(client,prompt_var_list,temperature,prompt_path):
    
    with open(prompt_path, 'r') as file:
        file_content = file.read()
    
    message = json.dumps({
    "prompt": file_content.format(*prompt_var_list), 
    "temperature": temperature,
    "max_gen_len": 600
    })

    accept = 'application/json'
    contentType = 'application/json'

    response = client.invoke_model(body=message, modelId=MODEL, accept=accept, contentType=contentType)

    chat_completion = json.loads(response.get("body").read())

    return chat_completion.get("generation").strip()

# performs adversarial attack hallucination detection process by asking the user the question
def perform_adversarial_attack(client,query,external_evidence,final_response_ans,final_response_exp):
    # print("Adversarial attack process starts")
    forward_reasoning_list = []
    backward_reasoning_list = []
    fwd_main_answers_list = []
    bck_main_answers_list = []
    all_responses_list = []
    
    fwd_main_answers_list.append(final_response_ans) #first response i.e. original response
    forward_reasoning_list.append(f"{final_response_ans}\n{final_response_exp}")
    fwd_main_answers_list.append(final_response_exp) #final response i.e. last response that is to be used as the final answer

    #backward reasoning
    
    fwd_extracted_final_response = final_response_ans[:]
    fwd_extracted_final_resp_exp = final_response_exp[:]
    # if len(fwd_extracted_final_response.split()) < 5:
    fwd_extracted_final_response = fwd_extracted_final_response + " " + fwd_extracted_final_resp_exp

    external_evidence = json.dumps(external_evidence, indent=4)
    prompt_var_list = [external_evidence, fwd_extracted_final_response]
    back_reasoning_response = perform_llama_response(client,prompt_var_list,TEMPERATURE,BACKWARD_REASONING_PROMPT_PATH)
    # print(back_reasoning_response)
    bck_main_answers_list.append(back_reasoning_response)
    backward_reasoning_list.append(back_reasoning_response)
    
    # print(len(backward_reasoning_list))

    return forward_reasoning_list, backward_reasoning_list, fwd_main_answers_list, bck_main_answers_list, all_responses_list


def start_meta_api_model_response(query,external_evidence):
    print("Meta Llama2 model response process starts",query)
    client = boto3.client(service_name='bedrock-runtime', region_name='us-east-1')
    prompt_var_list = [query, external_evidence]
    chat_completion = perform_llama_response(client,prompt_var_list,TEMPERATURE,QUERY_PROMPT_PATH)
    og_response_dict = process_response(chat_completion)
    # print(og_response_dict)
    if not check_dict_keys_condition(og_response_dict):
        og_response_dict['Answer:'] = next(iter(og_response_dict.items()))[1]
        og_response_dict['Explanation:'] = ""
    forward_reasoning_list, backward_reasoning_list, fwd_main_answers_list, bck_main_answers_list, \
                                    all_responses_list = perform_adversarial_attack(client,query,
                                    external_evidence,og_response_dict['Answer:'],og_response_dict['Explanation:'])
    # print("Meta Llama2 model response process ends")
    return og_response_dict, forward_reasoning_list, backward_reasoning_list, fwd_main_answers_list, \
                                                        bck_main_answers_list, all_responses_list
