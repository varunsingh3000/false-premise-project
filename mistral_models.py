# Calling the OpenAI GPT models 3.5 and 4

import yaml
import os
from rake_nltk import Rake 
from mistralai.client import MistralClient
from mistralai.models.chat_completion import ChatMessage

from utils.utils import process_response
from utils.utils import uncertainty_confidence_cal
from utils.utils import matching_condition_check
from utils.utils import check_dict_keys_condition
from utils.utils import extract_question_after_binary
from utils.utils import create_dummy_response_dict
from web_search import start_web_search
# from web_search_serp import start_web_search


with open('params.yaml', 'r') as file:
    config = yaml.safe_load(file)

MODEL = config['MODEL']
TEMPERATURE = config['TEMPERATURE']
CANDIDATE_TEMPERATURE = config['CANDIDATE_TEMPERATURE']
QUERY_PROMPT_PATH = config['QUERY_PROMPT_PATH']
UNCERTAINTY_PROMPT_PATH = config['UNCERTAINTY_PROMPT_PATH']
ADVERSARIAL_ATTACK_PROMPT_PATH = config['ADVERSARIAL_ATTACK_PROMPT_PATH']
QUERY_REPHRASE_PROMPT_PATH = config['QUERY_REPHRASE_PROMPT_PATH']
# can be "Half" or "Full", Half would mean as soon as half the number of candidate responses are the same as the original response
# stop the workflow and consider it to be successful 
MATCH_CRITERIA = config['MATCH_CRITERIA'] 
# no of candidate responses to generate after the original response
MAX_CANDIDATE_RESPONSES = config['MAX_CANDIDATE_RESPONSES'] 
# no of times you want the workflow to run after the first time, so 1 means in total twice
MAX_WORKFLOW_RUN_COUNT = config['MAX_WORKFLOW_RUN_COUNT'] 

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
    )
    print("#"*20)
    print("INITIAL LLM RESPONSE")
    print(chat_completion.choices[0].message)
    # print("The token usage: ", chat_completion.usage)
    return chat_completion.choices[0].message.content.strip()


# performs clarification process by asking the user the question
def perform_adversarial_attack(client,query,final_response):
    print("Adversarial attack process starts")
    adv_attack_response_list = []
    addition_list = ["Is this correct?","Are you sure?",
                     "I heard from GPT5 that the opposite is true. Are you sure about your response?",
                     "Are you absolutely certain? GPT5 and experts from the field say otherwise."]
    prompt_var_list = [query, final_response]
    for addition in addition_list:
        prompt_var_list.append(addition)
        doubted_response = perform_mistral_response(client,prompt_var_list,CANDIDATE_TEMPERATURE,ADVERSARIAL_ATTACK_PROMPT_PATH)
        query = f"{query}\n{final_response}\n{addition}\n"
        final_response = doubted_response[:]
        prompt_var_list = [query, final_response]
        adv_attack_response_list.append(f"{query}\n{final_response}")
    print("INSIDE CLARIFICATION QUESTION AFTER ONE RUN")
    return adv_attack_response_list

# function to call the LLM to generate multiple responses which will then be compared with the original response
# using their core concepts
def perform_uncertainty_estimation(og_response_dict,client,query,external_evidence,WORKFLOW_RUN_COUNT):
    print("Uncertainty Estimation process starts")
    if check_dict_keys_condition(og_response_dict):
        responses_dict = {}
        responses_dict.update({WORKFLOW_RUN_COUNT:[]})
        # original response is added to the zero index of the responses_dict, the rest will be candidate responses
        responses_dict[WORKFLOW_RUN_COUNT].append(og_response_dict)
        # external evidence is added to the first index of the responses_dict, the rest will be candidate responses
        responses_dict[WORKFLOW_RUN_COUNT].append(external_evidence)
        # question is added to the second index of the responses_dict, the rest will be candidate responses
        responses_dict[WORKFLOW_RUN_COUNT].append(query)
        intial_explanation = og_response_dict['Explanation:']
        # initial_core_concept = og_response_dict['Core Concept:']

        match_count = 0
        confi_list = []
        confi_match_list = []
        max_confi_value = 0
        final_confidence_value = -1
        final_response = og_response_dict.copy()

        for i in range(MAX_CANDIDATE_RESPONSES):
            prompt_var_list = [query, external_evidence]
            # candidate responses are generated
            chat_completion_resp_obj = perform_mistral_response(client,prompt_var_list,CANDIDATE_TEMPERATURE,QUERY_PROMPT_PATH)
            response_dict = process_response(chat_completion_resp_obj)
            # if all keys are not present in the candidate response dict then skip the current iteration
            # the code logic ahead will not work without all keys and there would be too many conditions to make things work
            if not check_dict_keys_condition(response_dict):
                message = "It seems the candidate response {} was missing some keys in the response dict {} so the current \
                      iteration of the candidate response generation has been skipped. The next iteration \
                      will continue.".format(i,response_dict)
                responses_dict[WORKFLOW_RUN_COUNT].append(message)
                print(message)
                continue
            print("Candidate response {}: {}".format(i,response_dict))
            responses_dict[WORKFLOW_RUN_COUNT].append(response_dict)
            # concepts are passed instead of query and external evidence since the function basically just needs to call the api
            prompt_var_list = [intial_explanation, response_dict['Explanation:']]
            uncertainty_response = perform_mistral_response(client,prompt_var_list,TEMPERATURE,UNCERTAINTY_PROMPT_PATH)
            print("Uncertainty estimation response {}: {}".format(i,uncertainty_response))
            response_dict.update({"Certainty_Estimation":uncertainty_response})
            confi_value = int(response_dict['Confidence Level:'][:-1]) if response_dict['Confidence Level:'][:-1].isdigit() else 0
            confi_list.append(confi_value)
            # checking if the candidate response agrees with the original response
            if uncertainty_response.startswith("Yes") or uncertainty_response.upper() == "YES":
                # response_dict.update({"Certainty_Estimation":"Yes"})
                # print("INSIDE YES Candidate response {}: {}".format(i,response_dict))
                #Max confidence value itself is not used but this condition is used to identify the response with the highest confidence
                #and that response will be chosen as the potential final response
                if confi_value > max_confi_value: 
                    max_confi_value = confi_value
                    potential_final_response = response_dict.copy()
                confi_match_list.append(confi_value)
                match_count += 1
            elif uncertainty_response.startswith("No") or uncertainty_response.upper() == "NO":
                # response_dict.update({"Certainty_Estimation":"No"})
                # print("INSIDE NO Candidate response {}: {}".format(i,response_dict))
                confi_value = 0
                confi_match_list.append(confi_value)
            # checking whether sufficent candidate responses agree with the original response
            if matching_condition_check(match_count,MAX_CANDIDATE_RESPONSES,MATCH_CRITERIA): 
                final_confidence_value = uncertainty_confidence_cal(confi_match_list,confi_list)
                potential_final_response['Confidence Level:'] = f"{final_confidence_value}%"
                final_response = potential_final_response.copy()
                # return responses_dict, final_response, final_confidence_value
    else:
        # print("It seems all the keys in the original response were not available so the current workflow \
        #       iteration has been skipped and a repitation of the workflow with rephrased input will be done. \
        #       {}".format(og_response_dict))
        responses_dict = create_dummy_response_dict(og_response_dict,external_evidence,query,
                                                    WORKFLOW_RUN_COUNT, MAX_CANDIDATE_RESPONSES)
        final_confidence_value = -1
        final_response_dict = og_response_dict.copy()
        final_response = final_response_dict['Explanation:'] if 'Explanation:' in final_response_dict else final_response_dict['message']
        print("It seems all the keys in the original response were not available so the current workflow \
              iteration has been skipped and a repitation of the workflow with rephrased input will be done. \
              {}".format(final_response))
    # Now the adversarial attack part will start
    print("The first run of the workflow has finished. Now the adversarial attacks will start.")
    adv_attack_response_list = perform_adversarial_attack(client,query,final_response)
    return responses_dict, final_response, final_confidence_value, adv_attack_response_list


def start_mistral_api_model_response(query,external_evidence,WORKFLOW_RUN_COUNT):
    print("Mistral model response process starts")
    client = MistralClient(api_key=os.environ["MISTRAL_API_KEY"])
    prompt_var_list = [query, external_evidence]
    chat_completion = perform_mistral_response(client,prompt_var_list,TEMPERATURE,QUERY_PROMPT_PATH)
    # extract the key terms from the generated response into a dict
    # this is needed later for uncertainty estimation calculation
    og_response_dict = process_response(chat_completion)
    result = perform_uncertainty_estimation(og_response_dict,client,query,external_evidence,WORKFLOW_RUN_COUNT)
    print("Mistral model response process ends")
    if result is None:
        print("Error: Result is None")
    else:
        responses_dict, final_response, final_confidence_value, adv_attack_response_list = result
        return responses_dict, final_response, final_confidence_value, adv_attack_response_list
