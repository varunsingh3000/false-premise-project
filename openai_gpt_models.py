# Calling the OpenAI GPT models 3.5 and 4

import json
import os 
import requests
from openai import OpenAI

from utils.utils import uncertainty_confidence_cal

WORKFLOW_RUN_COUNT = 0

# call the openai api with either gpt 3.5 or gpt 4 latest model
def perform_gpt_response(client,query,model,temperature,prompt_path,external_evidence):
    #Read the prompts from txt files
    with open(prompt_path, 'r') as file:
        file_content = file.read()
    
    #passing the query and the external evidence as variables into the prompt
    message = [
        {
            "role": "system",
            "content": file_content.format(query,external_evidence) 
        }
    ]

    chat_completion = client.chat.completions.create(
        messages=message,
        model=model,
        temperature=temperature
        )

    print("#"*20)
    # print("INITIAL LLM RESPONSE")
    print(chat_completion.choices[0].message)
    print("The token usage: ", chat_completion.usage)

    return chat_completion

# function to parse through the api response and extract certain keywords in a dict
def process_response(chat_completion):
    #use the chat_completion object to retrieve the textual LLM response
    text = chat_completion.choices[0].message.content

    # Remove all newline characters ("\n")
    text_without_newlines = text.replace('\n', '')

    # Define key terms to split the text into sections
    key_terms = ['Explanation:', 'Answer:', 'Confidence Level:', 'Source:', 'Core Concept:', 'Premise of the Question:']

    # Initialize an empty dictionary to store key-value pairs
    response_dict = {}

    # Splitting the text into sections based on key terms
    for i in range(len(key_terms) - 1):
        term = key_terms[i]
        next_term = key_terms[i + 1]
        if term in text_without_newlines and next_term in text_without_newlines:
            split_text = text_without_newlines.split(term, 1)[1].split(next_term, 1)
            response_dict[term.strip()] = split_text[0].strip() if len(split_text) > 1 else ''

    # For the last key term
    last_term = key_terms[-1]
    if last_term in text_without_newlines:
        split_text = text_without_newlines.split(last_term, 1)
        response_dict[last_term.strip()] = split_text[1].strip() if len(split_text) > 1 else ''

    # print(response_dict)
    # print(response_dict.keys())
    return response_dict


def perform_display_potential_hal_message():
    # This function will just display the
    print("There is a high likelihood that the response generated is inaccurate, we request you carefully check the \
          response before using it")
    exit(1)

# performs clarification process by asking the user the question
def perform_clarification_ques(initial_core_concept,uncertainty_prompt_path,model,temperature,
                               prompt_path,num_runs,external_evidence):
    print("Clarification Question process starts")
    cq_user_ques = f"What would you like to know about '{initial_core_concept}'"
    cq_user_ans = "I would like to know about Mars and it's moons"

    start_openai_api_model_response(cq_user_ans,prompt_path,uncertainty_prompt_path,model,
                                    temperature,num_runs,external_evidence)
        # print(modified_chat_completion.choices[0].message)
        # print("The token usage: ", modified_chat_completion.usage)

# function to call the LLM to generate multiple responses which will then be compared with the original response
# using their core concepts
def perform_uncertainty_estimation(response_dict,client,query,temperature,model,prompt_path,
                                   uncertainty_prompt_path,num_runs,external_evidence):
    print("Uncertainty Estimation process starts")
    global WORKFLOW_RUN_COUNT
    new_temperature = 0.7
    # responses_dict = {}
    initial_core_concept = response_dict['Core Concept:']

    match_count = 0
    match_check = num_runs//2
    confi_list = []
    max_confi_value = 0

    for i in range(num_runs):
        # candidate responses are generated
        chat_completion_resp_obj = perform_gpt_response(client,query,model,new_temperature,prompt_path,external_evidence)
        response_dict = process_response(chat_completion_resp_obj)
        print("Candidate response {}: {}".format(i,response_dict))
        # responses_dict["resp{}".format(i+1)] = response_dict #just saving all of the objs for now, not actually using it
        # concepts are passed instead of query and external evidence since the function basically just needs to call the api
        chat_completion_uncertainty_resp_obj = perform_gpt_response(client,initial_core_concept,model,temperature,
                                               uncertainty_prompt_path,response_dict['Core Concept:'])
        uncertainty_response = chat_completion_uncertainty_resp_obj.choices[0].message.content
        print("Uncertainty estimation response {}: {}".format(i,uncertainty_response))
        # checking if the candidate response agrees with the original response
        if uncertainty_response.upper() == "YES":
            confi_value = int(response_dict['Confidence Level:'][:-1]) if response_dict['Confidence Level:'][:-1].isdigit() else 0
            if confi_value > max_confi_value:
                max_confi_value = confi_value
                potential_final_response = response_dict.copy()
            confi_list.append(confi_value)
            match_count += 1
        
        # checking whether sufficent candidate responses agree with the original response
        if match_count > match_check:
            final_confidence_value = uncertainty_confidence_cal(confi_list)
            potential_final_response['Confidence Level:'] = f"{final_confidence_value}%"
            final_response = potential_final_response.copy()
            print("The final response is : ",final_response)
            return final_response 
    
    if match_count <= match_check:
        if WORKFLOW_RUN_COUNT == 0 and WORKFLOW_RUN_COUNT < 1:
            print("It seems that the LLM is uncertain about it's response. User clarification input is requested")
            WORKFLOW_RUN_COUNT += 1
            perform_clarification_ques(initial_core_concept,uncertainty_prompt_path,model,
                                       temperature,prompt_path,num_runs,external_evidence)
            
        elif WORKFLOW_RUN_COUNT > 0:
            perform_display_potential_hal_message()


def start_openai_api_model_response(query,prompt_path,uncertainty_prompt_path,model,temperature,num_runs,external_evidence):
    print("OpenAI model response process starts")
    client = OpenAI() # defaults to os.environ.get("OPENAI_API_KEY")
    # call the LLM and generate a response
    #this func will basically perform all the operations needed for one run of the LLM workflow
    chat_completion = perform_gpt_response(client,query,model,temperature,prompt_path,external_evidence)
    # extract the key terms from the generated response into a dict
    # this is needed later for uncertainty estimation calculation
    response_dict = process_response(chat_completion)
    final_response = perform_uncertainty_estimation(response_dict,client,query,temperature,model,
                                                    prompt_path,uncertainty_prompt_path,num_runs,external_evidence)
    
    print("OpenAI model response process ends")

