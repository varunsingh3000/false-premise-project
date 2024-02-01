#This is the main file from where the execution of the workflow starts

import yaml
import pandas as pd

from utils.dataset import start_dataset_processing
from web_search import start_web_search
# from web_search_serp import start_web_search
from openai_gpt_models import start_openai_api_model_response
# from openai_gpt_workflow_test import start_openai_api_model_response
from mistral_models import start_mistral_api_model_response
from meta_llama2_models import start_meta_api_model_response

with open('params.yaml', 'r') as file:
    config = yaml.safe_load(file)

# Retrieve parameters 
MODEL = config['MODEL']
TEMPERATURE = config['TEMPERATURE']
QUERY_PROMPT_PATH = config['QUERY_PROMPT_PATH']
UNCERTAINTY_PROMPT_PATH = config['UNCERTAINTY_PROMPT_PATH']
RESULT_SAVE_PATH = config['RESULT_SAVE_PATH']
DATASET_NAME = config['DATASET_NAME']
#defines the max no of candidate responses to generate, the actual no of responses could vary depending on the matching condition
MAX_CANDIDATE_RESPONSES = config['MAX_CANDIDATE_RESPONSES']
# WORKFLOW_RUN_COUNT variable keeps track of the amount of times the workflow has run, this has to be less than MAX_WORKFLOW_RUN_COUNT
# If it exceeds the count then the workflow is terminated. It is necessary that this variable is defined in main.py since for
# every question the WORKFLOW_RUN_COUNT is reset to 0 before being passed in the loop.
WORKFLOW_RUN_COUNT = config['WORKFLOW_RUN_COUNT']

# Func to start workflow for a query
def start_workflow(query,MODEL,WORKFLOW_RUN_COUNT):
    external_evidence = start_web_search(query)
    if MODEL in ["gpt-3.5-turbo-1106", "gpt-4-turbo-preview"]:
        result = start_openai_api_model_response(query,external_evidence,WORKFLOW_RUN_COUNT)
    elif MODEL in ["mistral-tiny", "mistral-small", "mistral-medium"]:
        result = start_mistral_api_model_response(query,external_evidence,WORKFLOW_RUN_COUNT)
    elif MODEL in ["meta.llama2-13b-chat-v1", "meta.llama2-70b-chat-v1"]:
        result = start_meta_api_model_response(query,external_evidence,WORKFLOW_RUN_COUNT)
    else:
        print("Please enter a valid MODEL id in the next attempt for the workflow to execute")
    responses_dict, final_response, final_confidence_value = result
    return responses_dict, final_response, final_confidence_value
 

def start_complete_workflow():
    # processing specific to freshqa dataset
    ques_id_list, query_list, ans_list = start_dataset_processing(DATASET_NAME)

    # list variables initialised to save results later to a dataframe
    ques_no_list = []
    workflow_run_count = []
    question_list = []
    true_ans_list = []
    og_response_list = []
    finalans_list = []
    finalconfi_list = []
    evidence_list = []
    candidate_responses_list = []

    for ques_id,query,true_ans in zip(ques_id_list,query_list,ans_list):
    
        print("NEW QUERY HAS STARTED"*4)
        responses_dict, final_response, final_confidence_value = start_workflow(query,MODEL,WORKFLOW_RUN_COUNT)
        print("RESPONSES DICT : ", responses_dict)
        print("QUERY HAS FINISHED"*4)    

        num_keys = len(responses_dict)
        temp_candidate_response_list = []
        temp_indi_resp_list=[]
        for key_value in responses_dict:
            ques_no_list.append(ques_id)
            true_ans_list.append(true_ans)
            workflow_run_count.append(key_value)
            og_response_list.append(responses_dict[key_value][0])
            evidence_list.append(responses_dict[key_value][1])
            question_list.append(responses_dict[key_value][2])
            if num_keys > 1 and key_value != num_keys - 1:
                finalans_list.append("Final response could not be determined in this run of the workflow")
                finalconfi_list.append("Final confidence value could not be determined in this run of the workflow") 
            else:
                finalans_list.append(final_response)
                finalconfi_list.append(final_confidence_value)   
            # index 0, 1 and 2 are skipped because that is the original response, external evidence and question respectively
            # for loop iteration for the last element is skipped because that is the question
            for candidate_resp in responses_dict[key_value][3:]:  
                temp_indi_resp_list.append(candidate_resp)
            temp_candidate_response_list.append(temp_indi_resp_list)
            temp_indi_resp_list = []
        candidate_responses_list.append(temp_candidate_response_list)
    print("CANDIDATE RESPONSES LIST :" , candidate_responses_list)


    data_dict = {
        "ques_id":ques_no_list,
        "workflow_run_count":workflow_run_count,
        "question":question_list,
        "true_ans":true_ans_list,
        "original_response": og_response_list,
        "final_ans":finalans_list,
        "final_confi":finalconfi_list,
        "evidence": evidence_list
    }

    # loop to generate keys for candidate responses depending on the num of runs i.e. candidate responses
    for i in range(MAX_CANDIDATE_RESPONSES):
        key_name = f"candidate_response{i}"
        data_dict[key_name] = [] 

    # loop to append the values of the candidate responses to the list created in the above loop
    # for the responses of all the candidate responses generated via the worflow for all number of runs
    for candidate_response_list in candidate_responses_list:
        for individual_resp_list in candidate_response_list:
            for indi_response_ind in range(MAX_CANDIDATE_RESPONSES):
                if indi_response_ind < len(individual_resp_list):
                    candidate_response = individual_resp_list[indi_response_ind]
                else:
                    candidate_response = ""  # Assign an empty string if the index is out of bounds
                data_dict[f"candidate_response{indi_response_ind}"].append(candidate_response)

    # Find the maximum length among the lists
    max_length = max(len(lst) for lst in data_dict.values() if isinstance(lst, list))

    # Pad shorter lists with default value (e.g., None) to make them of equal lengths
    for key, value in data_dict.items():
        if isinstance(value, list):
            if len(value) < max_length:
                data_dict[key].extend([""] * (max_length - len(value)))

    print("$"*100)
    df = pd.DataFrame(data_dict)
    print(df.head())
    print("$"*100)

    df.to_csv(RESULT_SAVE_PATH + MODEL + ".csv",index=False)  # Set index=False to not write row indices

start_complete_workflow()

#############################################################

# import argparse

# def start_workflow(query, model, temperature, query_prompt_path, uncertainty_prompt_path, num_runs):
#     external_evidence = start_web_search(query)
#     start_openai_api_model_response(query, query_prompt_path, uncertainty_prompt_path, model, temperature, num_runs, external_evidence)

# def main():
#     parser = argparse.ArgumentParser(description="Run workflow with specified parameters")
    
#     parser.add_argument('--query', type=str, default="Why does Mars have three moons?", help='Query for the workflow')
#     parser.add_argument('--model', type=str, default="gpt-3.5-turbo-1106", help='Model name')
#     parser.add_argument('--temperature', type=float, default=0.0, help='Temperature value')
#     parser.add_argument('--query-prompt-path', type=str, default=r"C:\GAMES_SETUP\Thesis\Code\Prompts\GPT3.5\initial_prompt_w_cot_single.txt", help='Path to query prompt')
#     parser.add_argument('--uncertainty-prompt-path', type=str, default=r"C:\GAMES_SETUP\Thesis\Code\Prompts\GPT3.5\uncertainty_estimation_confidence.txt", help='Path to uncertainty prompt')
#     parser.add_argument('--num-runs', type=int, default=3, help='Number of runs')
    
#     args = parser.parse_args()

#     start_time = time.time()

#     start_workflow(
#         args.query,
#         args.model,
#         args.temperature,
#         args.query_prompt_path,
#         args.uncertainty_prompt_path,
#         args.num_runs
#     )

#     end_time = time.time()
#     print(f"Total time taken: {end_time - start_time} seconds")

# if __name__ == "__main__":
#     main()



