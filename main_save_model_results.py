#This is the main file from where the execution of the workflow starts

import json
import yaml
import pandas as pd

from utils.dataset import start_dataset_processing
from utils.utils import generate_evidence_batch
from utils.utils import auto_evaluation
# from web_search import start_web_search
# from web_search_serp import start_web_search
from openai_gpt_models import start_openai_api_model_response
from mistral_models import start_mistral_api_model_response
from meta_llama2_models import start_meta_api_model_response

with open('params.yaml', 'r') as file:
    config = yaml.safe_load(file)

# Retrieve parameters 
MODEL = config['MODEL']
TEMPERATURE = config['TEMPERATURE']
EVIDENCE_BATCH_SAVE_PATH = config['EVIDENCE_BATCH_SAVE_PATH']
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
def start_workflow(query,external_evidence,MODEL,WORKFLOW_RUN_COUNT):
    # external_evidence = start_web_search(query)
    if MODEL in ["gpt-3.5-turbo", "gpt-3.5-turbo-0125","gpt-3.5-turbo-1106", "gpt-4-turbo-preview"]:
        result = start_openai_api_model_response(query,external_evidence,WORKFLOW_RUN_COUNT)
    elif MODEL in ["mistral-tiny", "mistral-small", "mistral-medium", "mistral-medium-latest"]:
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

    # list variables initialised to save QA results later to a dataframe
    ques_no_list = []
    workflow_run_count = []
    question_list = []
    true_ans_list = []
    og_response_list = []
    final_ans_dict_list = []
    finalconfi_list = []
    evidence_list = []
    candidate_responses_list = []
    final_response_list = []

    
    #generate_evidence_batch is used to save evidence results in a batch
    # if the function has been called before and results are already save then comment the function call
    # generate_evidence_batch(ques_id_list,query_list)

    with open(EVIDENCE_BATCH_SAVE_PATH, 'r') as json_file:
        evidence_batch_list = json.load(json_file)
        # print(evidence_batch_list)
        # exit(1)
        
    for ques_id,query,true_ans,external_evidence in zip(ques_id_list,query_list,ans_list,evidence_batch_list):
    
        # print("NEW QUERY HAS STARTED"*4)
        responses_dict, final_response, final_confidence_value = start_workflow(query,
                                                            external_evidence,MODEL,WORKFLOW_RUN_COUNT)
        # print("RESPONSES DICT : ", responses_dict)
        # print("QUERY HAS FINISHED"*4)    

        temp_candidate_response_list = []
        temp_indi_resp_list=[]
        for key_value in responses_dict:
            ques_no_list.append(ques_id)
            true_ans_list.append(true_ans)
            workflow_run_count.append(key_value)
            og_response_list.append(responses_dict[key_value][0])
            evidence_list.append(responses_dict[key_value][1])
            question_list.append(responses_dict[key_value][2])
            final_ans_dict_list.append(final_response)
            finalconfi_list.append(final_confidence_value) 

            for candidate_resp in responses_dict[key_value][3:]:  
                temp_indi_resp_list.append(candidate_resp)
            temp_candidate_response_list.append(temp_indi_resp_list)
            temp_indi_resp_list = []
        candidate_responses_list.append(temp_candidate_response_list)
    
        #appending results for accuracy
        final_resp_text = ""
        if 'Answer:' in final_response:
            final_resp_text += final_response['Answer:']
        if 'Explanation:' in final_response:
            if final_resp_text:  # If Answer: key exists and is not empty, add a space before Explanation:
                final_resp_text += " "
            final_resp_text += final_response['Explanation:']
        
        final_response_list.append(final_resp_text)

    qa_data_dict = {
        "ques_id":ques_no_list,
        "workflow_run_count":workflow_run_count,
        "question":question_list,
        "true_ans":true_ans_list,
        "final_answer":final_response_list,
        "original_response": og_response_list,
        "final_ans_dict":final_ans_dict_list,
        "final_confi":finalconfi_list,
        "evidence": evidence_list
    }

    # loop to generate keys for candidate responses depending on the num of runs i.e. candidate responses
    for i in range(MAX_CANDIDATE_RESPONSES):
        key_name = f"candidate_response{i}"
        qa_data_dict[key_name] = [] 

    # loop to append the values of the candidate responses to the list created in the above loop
    # for the responses of all the candidate responses generated via the worflow for all number of runs
    for candidate_response_list in candidate_responses_list:
        for individual_resp_list in candidate_response_list:
            for indi_response_ind in range(MAX_CANDIDATE_RESPONSES):
                if indi_response_ind < len(individual_resp_list):
                    candidate_response = individual_resp_list[indi_response_ind]
                else:
                    candidate_response = ""  # Assign an empty string if the index is out of bounds
                qa_data_dict[f"candidate_response{indi_response_ind}"].append(candidate_response)

    # Find the maximum length among the lists
    max_length = max(len(lst) for lst in qa_data_dict.values() if isinstance(lst, list))

    # Pad shorter lists with default value (e.g., None) to make them of equal lengths
    for key, value in qa_data_dict.items():
        if isinstance(value, list):
            if len(value) < max_length:
                qa_data_dict[key].extend([""] * (max_length - len(value)))

    print("$"*100)
    df = pd.DataFrame(qa_data_dict)
    print(df.head())
    print("$"*100)

    df.to_excel(RESULT_SAVE_PATH + MODEL + "alltest.xlsx",index=False)  # Set index=False to not write row indices


def start_evaluation():
    #list variable to save automatic evaluation results
    accuracy_result_list = []
    final_accuracy_comment_list = []
    path = RESULT_SAVE_PATH + MODEL + "alltest.xlsx"
    # path = "C:\GAMES_SETUP\Thesis\Code\Results\evidence_test_gpt-3.5-turbo-1106alltest.xlsx"
    df = pd.read_excel(path)
    query_list = df["question"].tolist()
    true_ans_list = df["true_ans"].tolist()
    final_answer_list = df["final_answer"].tolist()
    for query,true_ans,final_resp_text in zip(query_list,true_ans_list,final_answer_list):
        extracted_gt_ans_resp1, accuracy_comment, accuracy = auto_evaluation(query,true_ans,final_resp_text)
        accuracy_result_list.append(accuracy)
        final_accuracy_comment_list.append(accuracy_comment)

    df["accuracy"] = accuracy_result_list
    df["accuracy_comment"] = final_accuracy_comment_list

    df.to_excel(RESULT_SAVE_PATH + MODEL + "alltest_eval.xlsx",index=False)

start_complete_workflow()
start_evaluation()