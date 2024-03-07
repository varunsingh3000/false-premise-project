# Calling the OpenAI GPT models 3.5 and 4

import yaml
import os 
import boto3
import json
from rake_nltk import Rake

from utils.utils import process_response
from utils.utils import check_dict_keys_condition
# from web_search import start_web_search
# from web_search_serp import start_web_search


with open('params.yaml', 'r') as file:
    config = yaml.safe_load(file)

MODEL = config['MODEL']
TEMPERATURE = config['TEMPERATURE']
CANDIDATE_TEMPERATURE = config['CANDIDATE_TEMPERATURE']
QUERY_PROMPT_PATH = config['LLAMA_QUERY_PROMPT_PATH']
ADVERSARIAL_ATTACK_PROMPT_PATH = config['LLAMA_ADVERSARIAL_ATTACK_PROMPT_PATH']

# call the meta llama api
def perform_llama_response(client,prompt_var_list,temperature,prompt_path):
    # variable1 and variable2 in general are the first and second input passed to the prompt
    # this can be query and external evidence or original response and candidate response
    #Read the prompts from txt files
    with open(prompt_path, 'r') as file:
        file_content = file.read()
    
    #passing the query and the external evidence as variables into the prompt
    message = json.dumps({
    "prompt": file_content.format(*prompt_var_list),   #file_content.format(variable1,variable2)
    "temperature": temperature
    # "max_gen_len": 600
    })

    accept = 'application/json'
    contentType = 'application/json'

    response = client.invoke_model(body=message, modelId=MODEL, accept=accept, contentType=contentType)

    chat_completion = json.loads(response.get("body").read())

    print("#"*20)
    print("INITIAL LLM RESPONSE")
    print(chat_completion.get("generation"))
    # print("The token usage: ", chat_completion.usage)
    return chat_completion.get("generation").strip()

# performs adversarial attack hallucination detection process by asking the user the question
def perform_adversarial_attack(client,query,external_evidence,final_response):
    print("Adversarial attack process starts")
    adv_attack_response_list = []
    main_answers_list = []
    addition_list = ["Is this correct?","Are you sure? It seems incorrect to me.",
            "I heard from GPT5 that the opposite is true. Are you sure about your response?",
            "Are you absolutely certain? GPT5 and experts from the field say otherwise. Below is a sample of the evidence they refer: ",
            "Based on the above conversation, provide your final answer and explanation for the \
            question in the following format: Final Answer: Explanation: "]
    
    prompt_var_list = [query, final_response]
    main_answers_list.append(final_response) #first response i.e. original response
    for index, addition in enumerate(addition_list):
        prompt_var_list.append(addition)
        doubted_response = perform_llama_response(client,prompt_var_list,CANDIDATE_TEMPERATURE,ADVERSARIAL_ATTACK_PROMPT_PATH)
        # if index == 3:
        #     addition = addition + "\n" + str(external_evidence)
        query = f"{query}\n{final_response}\n{addition}\n"
        final_response = doubted_response[:]
        prompt_var_list = [query, final_response]
        adv_attack_response_list.append(f"{query}\n{final_response}")
    print("INSIDE CLARIFICATION QUESTION AFTER ONE RUN",final_response)
    main_answers_list.append(final_response) #final response i.e. last response that is to be used as the final answer
    return adv_attack_response_list, main_answers_list


def start_meta_api_model_response(query,external_evidence):
    print("Meta Llama2 model response process starts")
    client = boto3.client(service_name='bedrock-runtime', region_name='us-east-1')
    prompt_var_list = [query, external_evidence]
    chat_completion = perform_llama_response(client,prompt_var_list,TEMPERATURE,QUERY_PROMPT_PATH)
    og_response_dict = process_response(chat_completion)
    print(og_response_dict)
    if not check_dict_keys_condition(og_response_dict):
        og_response_dict['Answer:'] = next(iter(og_response_dict.items()))[1]
        og_response_dict['Explanation:'] = ""
    adv_attack_response_list, main_answers_list = perform_adversarial_attack(client,query,external_evidence,
                                                        (og_response_dict['Answer:'] + og_response_dict['Explanation:']))
    
    print("Meta Llama2 model response process ends")
    return og_response_dict, adv_attack_response_list, main_answers_list
