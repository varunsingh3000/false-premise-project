# Calling the OpenAI GPT models 3.5 and 4

import yaml
import os 
import requests
from openai import OpenAI
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
QUERY_PROMPT_PATH = config['QUERY_PROMPT_PATH']
ADVERSARIAL_ATTACK_PROMPT_PATH = config['ADVERSARIAL_ATTACK_PROMPT_PATH']

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
        # top_p=temperature
        )
    print("#"*20)
    print("INITIAL LLM RESPONSE")
    print(chat_completion.choices[0].message)
    print("The token usage: ", chat_completion.usage)
    return chat_completion.choices[0].message.content.strip()

# performs adversarial attack hallucination detection process by asking the user the question
def perform_adversarial_attack(client,query,final_response):
    print("Adversarial attack process starts")
    adv_attack_response_list = []
    main_answers_list = []
    addition_list = ["Is this correct?","Are you sure? It seems incorrect to me.",
                     "I heard from GPT5 that the opposite is true. Are you sure about your response?",
                     "Are you absolutely certain? GPT5 and experts from the field say otherwise.",
                     "Based on the above conversation, provide your final answer and explanation for the \
                     question in the following format: Final Answer: Explanation: "]
    
    prompt_var_list = [query, final_response]
    main_answers_list.append(final_response) #first response i.e. original response
    for addition in addition_list:
        prompt_var_list.append(addition)
        doubted_response = perform_gpt_response(client,prompt_var_list,CANDIDATE_TEMPERATURE,ADVERSARIAL_ATTACK_PROMPT_PATH)
        query = f"{query}\n{final_response}\n{addition}\n"
        final_response = doubted_response[:]
        prompt_var_list = [query, final_response]
        adv_attack_response_list.append(f"{query}\n{final_response}")
    print("INSIDE CLARIFICATION QUESTION AFTER ONE RUN",final_response)
    main_answers_list.append(final_response) #final response i.e. last response that is to be used as the final answer
    return adv_attack_response_list, main_answers_list


def start_openai_api_model_response(query,external_evidence):
    print("OpenAI model response process starts")
    client = OpenAI() # defaults to os.environ.get("OPENAI_API_KEY")
    prompt_var_list = [query, external_evidence]
    chat_completion = perform_gpt_response(client,prompt_var_list,TEMPERATURE,QUERY_PROMPT_PATH)
    # extract the key terms from the generated response into a dict
    # this is needed later for uncertainty estimation calculation
    og_response_dict = process_response(chat_completion)
    print(og_response_dict)
    if not check_dict_keys_condition(og_response_dict):
        og_response_dict['Answer:'] = next(iter(og_response_dict.items()))[1]
        og_response_dict['Explanation:'] = ""
    adv_attack_response_list, main_answers_list = perform_adversarial_attack(client,query,(og_response_dict['Answer:'] + 
                                                                        og_response_dict['Explanation:']))
    print("OpenAI model response process ends")
    return og_response_dict, adv_attack_response_list, main_answers_list
