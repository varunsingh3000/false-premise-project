# Calling the OpenAI GPT models 3.5 and 4

import yaml
import os 
import requests
from openai import OpenAI

from utils.utils import uncertainty_confidence_cal
from utils.utils import matching_condition_check
from utils.utils import check_dict_keys_condition
from web_search import start_web_search


with open('params.yaml', 'r') as file:
    config = yaml.safe_load(file)

MODEL = config['MODEL']
TEMPERATURE = config['TEMPERATURE']
CANDIDATE_TEMPERATURE = config['CANDIDATE_TEMPERATURE']
QUERY_PROMPT_PATH = config['QUERY_PROMPT_PATH']
UNCERTAINTY_PROMPT_PATH = config['UNCERTAINTY_PROMPT_PATH']
# can be "Half" or "Full", Half would mean as soon as half the number of candidate responses are the same as the original response
# stop the workflow and consider it to be successful 
MATCH_CRITERIA = config['MATCH_CRITERIA'] 
# no of candidate responses to generate after the original response
MAX_CANDIDATE_RESPONSES = config['MAX_CANDIDATE_RESPONSES'] 
# no of times you want the workflow to run after the first time, so 1 means in total twice
MAX_WORKFLOW_RUN_COUNT = config['MAX_WORKFLOW_RUN_COUNT'] 

# call the openai api with either gpt 3.5 or gpt 4 latest model
def perform_gpt_response(client,variable1,temperature,prompt_path,variable2):
    # variable1 and variable2 in general are the first and second input passed to the prompt
    # this can be query and external evidence or original response and candidate response
    #Read the prompts from txt files
    with open(prompt_path, 'r') as file:
        file_content = file.read()
    
    #passing the query and the external evidence as variables into the prompt
    message = [
        {
            "role": "system",
            "content": file_content.format(variable1,variable2) 
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
    # print("The token usage: ", chat_completion.usage)
    return chat_completion

# function to parse through the api response and extract certain keywords in a dict
def process_response(chat_completion):
    #use the chat_completion object to retrieve the textual LLM response
    text = chat_completion.choices[0].message.content

    # Remove all newline characters ("\n")
    text_without_newlines = text.replace('\n', '')

    # Define key terms to split the text into sections
    key_terms = ['Explanation:', 'Answer:', 'Confidence Level:', 'Source:', 'Core Concept:', 'Premise of the Question:']
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

    return response_dict

# performs clarification process by asking the user the question
def perform_clarification_ques(responses_dict,initial_core_concept,WORKFLOW_RUN_COUNT):
    print("Clarification Question process starts")
    # cq_user_ques = f"What would you like to know about '{initial_core_concept}'"
    # cq_user_ans = f"I would like to know about '{initial_core_concept}'"
    cq_user_ans = input(f"What would you like to know about '{initial_core_concept}'")

    external_evidence = start_web_search(cq_user_ans)
    result = start_openai_api_model_response(cq_user_ans,WORKFLOW_RUN_COUNT,external_evidence)
    print("LENGTH OF RESULT : ", len(result))
    # print("RESULT: ", result)
    responses_dict_new, final_response, final_confidence_value = result

    responses_dict.update(responses_dict_new)
    print("INSIDE CALRIFICATION QUESTION AFTER ONE RUN")
    return responses_dict, final_response, final_confidence_value

# function to call the LLM to generate multiple responses which will then be compared with the original response
# using their core concepts
def perform_uncertainty_estimation(og_response_dict,client,query,WORKFLOW_RUN_COUNT,external_evidence):
    print("Uncertainty Estimation process starts")
    if check_dict_keys_condition(og_response_dict):
        responses_dict = {}
        responses_dict.update({WORKFLOW_RUN_COUNT:[]})
        # original response is added to the first index of the responses_dict, the rest will be candidate responses
        responses_dict[WORKFLOW_RUN_COUNT].append(og_response_dict)
        # external evidence is added to the second index of the responses_dict, the rest will be candidate responses
        responses_dict[WORKFLOW_RUN_COUNT].append(external_evidence)
        intial_explanation = og_response_dict['Explanation:']
        initial_core_concept = og_response_dict['Core Concept:']

        match_count = 0
        # match_check = MAX_CANDIDATE_RESPONSES//2
        confi_list = []
        confi_match_list = []
        max_confi_value = 0

        for i in range(MAX_CANDIDATE_RESPONSES):
            # candidate responses are generated
            chat_completion_resp_obj = perform_gpt_response(client,query,CANDIDATE_TEMPERATURE,QUERY_PROMPT_PATH,external_evidence)
            response_dict = process_response(chat_completion_resp_obj)
            # if all keys are not present in the candidate response dict then skip the current iteration
            # the code logic ahead will not work without all keys and there would be too many conditions to make things work
            if not check_dict_keys_condition(response_dict):
                print("It seems the candidate response {} was missing some keys in the response dict {} so the current \
                      iteration of the candidate response generation has been skipped. The next iteration \
                      will continue.".format(i,response_dict))
                continue
            print("Candidate response {}: {}".format(i,response_dict))
            responses_dict[WORKFLOW_RUN_COUNT].append(response_dict)
            # concepts are passed instead of query and external evidence since the function basically just needs to call the api
            chat_completion_uncertainty_resp_obj = perform_gpt_response(client,intial_explanation,TEMPERATURE,
                                                UNCERTAINTY_PROMPT_PATH,response_dict['Explanation:'])
            uncertainty_response = chat_completion_uncertainty_resp_obj.choices[0].message.content
            print("Uncertainty estimation response {}: {}".format(i,uncertainty_response))
            response_dict.update({"Certainty_Estimation":uncertainty_response})
            confi_value = int(response_dict['Confidence Level:'][:-1]) if response_dict['Confidence Level:'][:-1].isdigit() else 0
            confi_list.append(confi_value)
            # checking if the candidate response agrees with the original response
            if uncertainty_response.upper() == "YES":
                #Max confidence value itself is not used but this condition is used to identify the response with the highest confidence
                #and that response will be chosen as the potential final response
                if confi_value > max_confi_value: 
                    max_confi_value = confi_value
                    potential_final_response = response_dict.copy()
                confi_match_list.append(confi_value)
                match_count += 1
            if uncertainty_response.upper() == "NO":
                confi_value = 0
                confi_match_list.append(confi_value)
            # checking whether sufficent candidate responses agree with the original response
            if matching_condition_check(match_count,MAX_CANDIDATE_RESPONSES,MATCH_CRITERIA): 
                final_confidence_value = uncertainty_confidence_cal(confi_match_list,confi_list)
                potential_final_response['Confidence Level:'] = f"{final_confidence_value}%"
                final_response = potential_final_response.copy()
                return responses_dict, final_response, final_confidence_value
    else:
        print("It seems all the keys in the original response were not available so the current workflow \
              iteration has been skipped and a repitation of the workflow with user input will be done. \
              {}".format(og_response_dict))
        responses_dict = {}
        initial_core_concept = query
    # if the workflow has run till here then that means the matching condition was not satisfied and the workflow needs to repeat
    if WORKFLOW_RUN_COUNT < MAX_WORKFLOW_RUN_COUNT:
        print("It seems that the LLM is uncertain about it's response. User clarification input is requested")
        WORKFLOW_RUN_COUNT += 1
        responses_dict, final_response, final_confidence_value = perform_clarification_ques(responses_dict,
                                                initial_core_concept,WORKFLOW_RUN_COUNT)
        return responses_dict, final_response, final_confidence_value
    elif WORKFLOW_RUN_COUNT == MAX_WORKFLOW_RUN_COUNT:
        print("There is a high likelihood that the response generated is inaccurate, we request you carefully check the \
                response before using it")
        final_confidence_value = -1
        final_response = "Workflow did not succeed"
        return responses_dict, final_response, final_confidence_value


def start_openai_api_model_response(query,WORKFLOW_RUN_COUNT,external_evidence):
    print("OpenAI model response process starts")
    client = OpenAI() # defaults to os.environ.get("OPENAI_API_KEY")
    # call the LLM and generate a response
    #this func will basically perform all the operations needed for one run of the LLM workflow
    chat_completion = perform_gpt_response(client,query,TEMPERATURE,QUERY_PROMPT_PATH,external_evidence)
    # extract the key terms from the generated response into a dict
    # this is needed later for uncertainty estimation calculation
    og_response_dict = process_response(chat_completion)
    result = perform_uncertainty_estimation(og_response_dict,client,query,WORKFLOW_RUN_COUNT,external_evidence)
    print("OpenAI model response process ends")
    if result is None:
        print("Error: Result is None")
    else:
        responses_dict, final_response, final_confidence_value = result
        return responses_dict, final_response, final_confidence_value
