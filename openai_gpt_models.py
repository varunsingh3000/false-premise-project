# Calling the OpenAI GPT models 3.5 and 4

import yaml
import os 
import requests
from openai import OpenAI
from rake_nltk import Rake

from utils.utils import extract_value_from_single_key
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
FORWARD_REASONING_PROMPT_PATH = config['FORWARD_REASONING_PROMPT_PATH']
BACKWARD_REASONING_PROMPT_PATH = config['BACKWARD_REASONING_PROMPT_PATH']


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
        temperature=temperature,
        max_tokens=600
        )
    print("#"*20)
    print("INITIAL LLM RESPONSE")
    print(chat_completion.choices[0].message)
    print("The token usage: ", chat_completion.usage)
    return chat_completion.choices[0].message.content.strip()

# performs adversarial attack hallucination detection process by asking the user the question
def perform_adversarial_attack(client,query,external_evidence,final_response):
    print("Adversarial attack process starts")
    forward_reasoning_list = []
    backward_reasoning_list = []
    fwd_main_answers_list = []
    bck_main_answers_list = []
    all_responses_list = []
    forward_prompt_list = ["Is this correct?","Are you sure? It seems incorrect to me.",
            "I heard from GPT5 that the opposite is true. Are you sure about your response?",
            "Based on the above conversation, provide your answer and explanation for the \
            question in the following format: Forward Answer: Forward Explanation: "]
    
    prompt_var_list = [query, final_response]
    fwd_main_answers_list.append(final_response) #first response i.e. original response
    all_responses_list.append(final_response)
    for addition in forward_prompt_list:
        prompt_var_list.append(addition)
        forw_reasoning_response = perform_gpt_response(client,prompt_var_list,TEMPERATURE,FORWARD_REASONING_PROMPT_PATH)
        # if index == 3:
        #     addition = addition + "\n" + str(external_evidence)
        query = f"{query}\n{final_response}\n{addition}\n"
        final_response = forw_reasoning_response[:]
        all_responses_list.append(final_response)
        prompt_var_list = [query, final_response]
        forward_reasoning_list.append(f"{query}\n{final_response}")
    print("INSIDE CLARIFICATION QUESTION AFTER ONE RUN",final_response)
    fwd_main_answers_list.append(final_response) #final response i.e. last response that is to be used as the final answer

    #backward reasoning
    fwd_extracted_final_response = extract_value_from_single_key(final_response, key = "Forward Answer:")
    fwd_extracted_final_resp_exp = extract_value_from_single_key(final_response, key = "Forward Explanation:")
    if len(fwd_extracted_final_response.split()) < 5:
        fwd_extracted_final_response = fwd_extracted_final_response + " " + fwd_extracted_final_resp_exp

    backward_prompt_list = ["Is this correct?","Are you sure? It seems incorrect to me.",
            "I heard from GPT5 that the opposite is true. Are you sure about your response?",
            "Based on the above conversation, provide your final answer, final explanation and \
            also a final question that could be answered by the final answer for the statement \
            in the following format: Final Answer: Final Explanation: Final Question:"]
    for addition in backward_prompt_list:
        prompt_var_list = [fwd_extracted_final_response, addition]
        back_reasoning_response = perform_gpt_response(client,prompt_var_list,TEMPERATURE,BACKWARD_REASONING_PROMPT_PATH)
        if addition == backward_prompt_list[-1]:
            bck_main_answers_list.append(back_reasoning_response)
            back_reasoning_response = f"{external_evidence}\n{back_reasoning_response}"
        fwd_extracted_final_response = f"{fwd_extracted_final_response}\n{addition}\n{back_reasoning_response}"
        backward_reasoning_list.append(fwd_extracted_final_response)
    
    # print(len(backward_reasoning_list))

    return forward_reasoning_list, backward_reasoning_list, fwd_main_answers_list, bck_main_answers_list, all_responses_list


def start_openai_api_model_response(query,external_evidence):
    print("OpenAI model response process starts")
    client = OpenAI() # defaults to os.environ.get("OPENAI_API_KEY")
    prompt_var_list = [query, external_evidence]
    chat_completion = perform_gpt_response(client,prompt_var_list,TEMPERATURE,QUERY_PROMPT_PATH)
    og_response_dict = process_response(chat_completion)
    print(og_response_dict)
    if not check_dict_keys_condition(og_response_dict):
        og_response_dict['Answer:'] = next(iter(og_response_dict.items()))[1]
        og_response_dict['Explanation:'] = ""
    forward_reasoning_list, backward_reasoning_list, fwd_main_answers_list, bck_main_answers_list, \
                                    all_responses_list = perform_adversarial_attack(client,query,
                                    external_evidence,(og_response_dict['Answer:'] + og_response_dict['Explanation:']))
    print("OpenAI model response process ends")
    return og_response_dict, forward_reasoning_list, backward_reasoning_list, fwd_main_answers_list, \
                                                        bck_main_answers_list, all_responses_list