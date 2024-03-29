# Calling the OpenAI GPT models 3.5 and 4

import yaml
import os
from rake_nltk import Rake 
from mistralai.client import MistralClient
from mistralai.models.chat_completion import ChatMessage

from utils.utils import process_response
from utils.utils import extract_value_from_single_key
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

# call the mistral api with either tiny or small model
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
        messages=message,
        max_tokens=600
    )
    # print("#"*20)
    # print("INITIAL LLM RESPONSE")
    # print(chat_completion.choices[0].message)
    # print("The token usage: ", chat_completion.usage)
    return chat_completion.choices[0].message.content.strip()


# performs adversarial attack hallucination detection process by asking the user the question
def perform_adversarial_attack(client,query,external_evidence,final_response_ans,final_response_exp):
    # print("Adversarial attack process starts")
    forward_reasoning_list = []
    backward_reasoning_list = []
    fwd_main_answers_list = []
    bck_main_answers_list = []
    all_responses_list = []
    forward_prompt_list = ["Is this correct? Please provide your final answer and explanation for the \
            question in the following format: Forward Answer: Forward Explanation: "]
    
    # prompt_var_list = [query, final_response]
    fwd_main_answers_list.append(final_response_ans) #first response i.e. original response
    # all_responses_list.append(final_response_ans)

    forward_reasoning_list.append(f"{final_response_ans}\n{final_response_exp}")
    # print("INSIDE CLARIFICATION QUESTION AFTER ONE RUN",final_response)
    fwd_main_answers_list.append(final_response_exp) #final response i.e. last response that is to be used as the final answer

    #backward reasoning
    
    fwd_extracted_final_response = final_response_ans[:]
    fwd_extracted_final_resp_exp = final_response_exp[:]
    # if len(fwd_extracted_final_response.split()) < 5:
    fwd_extracted_final_response = fwd_extracted_final_response + " " + fwd_extracted_final_resp_exp

    
    prompt_var_list = [external_evidence, fwd_extracted_final_response]
    back_reasoning_response = perform_mistral_response(client,prompt_var_list,TEMPERATURE,BACKWARD_REASONING_PROMPT_PATH)
    bck_main_answers_list.append(back_reasoning_response)
    backward_reasoning_list.append(back_reasoning_response)
    
    # print(len(backward_reasoning_list))

    return forward_reasoning_list, backward_reasoning_list, fwd_main_answers_list, bck_main_answers_list, all_responses_list


def start_mistral_api_model_response(query,external_evidence):
    print("Mistral model response process starts ",query)
    client = MistralClient(api_key=os.environ["MISTRAL_API_KEY"])
    prompt_var_list = [query, external_evidence]
    chat_completion = perform_mistral_response(client,prompt_var_list,TEMPERATURE,QUERY_PROMPT_PATH)
    # extract the key terms from the generated response into a dict
    # this is needed later for uncertainty estimation calculation
    og_response_dict = process_response(chat_completion)
    # print(og_response_dict)
    if not check_dict_keys_condition(og_response_dict):
        og_response_dict['Answer:'] = next(iter(og_response_dict.items()))[1]
        og_response_dict['Explanation:'] = ""
    forward_reasoning_list, backward_reasoning_list, fwd_main_answers_list, bck_main_answers_list, \
                                    all_responses_list = perform_adversarial_attack(client,query,
                                    external_evidence,og_response_dict['Answer:'],og_response_dict['Explanation:'])
    # print("Mistral model response process ends")
    return og_response_dict, forward_reasoning_list, backward_reasoning_list, fwd_main_answers_list, \
                                                        bck_main_answers_list, all_responses_list

