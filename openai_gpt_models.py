# Calling the OpenAI GPT models 3.5 and 4

import yaml
import os 
import json
from openai import OpenAI

from utils.utils import process_response
from utils.utils import extract_value_from_single_key
from utils.utils import check_dict_keys_condition


with open('params.yaml', 'r') as file:
    config = yaml.safe_load(file)

MODEL = config['MODEL']
TEMPERATURE = config['TEMPERATURE']
CANDIDATE_TEMPERATURE = config['CANDIDATE_TEMPERATURE']
QUERY_PROMPT_PATH = config['QUERY_PROMPT_PATH']
BACKWARD_REASONING_RESP_PROMPT_PATH = config['BACKWARD_REASONING_RESP_PROMPT_PATH']
BACKWARD_REASONING_QUERY_PROMPT_PATH = config['BACKWARD_REASONING_QUERY_PROMPT_PATH']

# call the openai api with either gpt 3.5 or gpt 4 latest model
def perform_gpt_response(client,prompt_var_list,temperature,prompt_path):
    # all the variables needed for the prompt are in the prompt_var_list
    # this can be query and external evidence or original response and candidate response
    #Read the prompts from txt files
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
    
    return chat_completion.choices[0].message.content.strip()


def perform_adversarial_attack(client,query,external_evidence,final_response_ans,final_response_exp):
    # print("Adversarial attack process starts")
    fwd_main_answers_list = []
    bck_main_answers_list = []

    fwd_main_answers_list.append(final_response_ans) #first response i.e. original response
    fwd_main_answers_list.append(final_response_exp) #final response i.e. last response that is to be used as the final answer

    #backward reasoning
    fwd_extracted_final_response = final_response_ans[:]
    fwd_extracted_final_resp_exp = final_response_exp[:]
    if len(fwd_extracted_final_response.split()) < 5:
        fwd_extracted_final_response = fwd_extracted_final_response + " " + fwd_extracted_final_resp_exp

    external_evidence = json.dumps(external_evidence, indent=4)
    # print(external_evidence)
    prompt_var_list = [external_evidence, fwd_extracted_final_response]
    back_reasoning_response_query = perform_gpt_response(client,prompt_var_list,CANDIDATE_TEMPERATURE,BACKWARD_REASONING_QUERY_PROMPT_PATH)
    back_reasoning_response = perform_gpt_response(client,prompt_var_list,CANDIDATE_TEMPERATURE,BACKWARD_REASONING_RESP_PROMPT_PATH)
    bck_main_answers_list.append(back_reasoning_response_query)
    bck_main_answers_list.append(back_reasoning_response)

    return fwd_main_answers_list, bck_main_answers_list

def start_openai_api_model_response(query,external_evidence):
    print("OpenAI model response process starts",query)
    client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY")) # defaults to os.environ.get("OPENAI_API_KEY")
    prompt_var_list = [query, external_evidence]
    chat_completion = perform_gpt_response(client,prompt_var_list,TEMPERATURE,QUERY_PROMPT_PATH)
    og_response_dict = process_response(chat_completion)
    # print(og_response_dict)
    if not check_dict_keys_condition(og_response_dict):
        og_response_dict['Answer:'] = next(iter(og_response_dict.items()))[1]
        og_response_dict['Explanation:'] = ""
    fwd_main_answers_list, bck_main_answers_list = perform_adversarial_attack(client,query,external_evidence,
                                       og_response_dict['Answer:'],og_response_dict['Explanation:'])
    # print("Mistral model response process ends")
    return og_response_dict, fwd_main_answers_list, bck_main_answers_list