#This is the main file from where the execution of the workflow starts

import json
import yaml
import pandas as pd

from utils.dataset import start_dataset_processing
from utils.utils import generate_evidence_batch
from utils.utils import modify_evidence_batch_dict
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
RESULT_SAVE_PATH = config['RESULT_SAVE_PATH']
DATASET_NAME = config['DATASET_NAME']
MAX_CANDIDATE_RESPONSES = config['MAX_CANDIDATE_RESPONSES']
WORKFLOW_RUN_COUNT = config['WORKFLOW_RUN_COUNT']

# Func to start workflow for a query
def start_workflow(query,external_evidence,MODEL,WORKFLOW_RUN_COUNT):
    # external_evidence = start_web_search(query)
    if MODEL in ["gpt-3.5-turbo", "gpt-3.5-turbo-0125","gpt-3.5-turbo-1106", "gpt-4-turbo-preview"]:
        result = start_openai_api_model_response(query,external_evidence,WORKFLOW_RUN_COUNT)
    elif MODEL in ["mistral-tiny", "mistral-small","mistral-small-latest", "mistral-medium", "mistral-medium-latest"]:
        result = start_mistral_api_model_response(query,external_evidence,WORKFLOW_RUN_COUNT)
    elif MODEL in ["meta.llama2-13b-chat-v1", "meta.llama2-70b-chat-v1"]:
        result = start_meta_api_model_response(query,external_evidence,WORKFLOW_RUN_COUNT)
    else:
        print("Please enter a valid MODEL id in the next attempt for the workflow to execute")
    final_response = result
    return final_response
 

def start_complete_workflow():
    # processing specific to freshqa dataset

    dataset_elements = start_dataset_processing(DATASET_NAME)
    if len(dataset_elements) == 6:
        ques_id_list, query_list, ans_list, effective_year_list, \
        num_hops_list, fact_type_list = dataset_elements
    elif len(dataset_elements) == 3:
        ques_id_list, query_list, ans_list = dataset_elements
    
    # list variables initialised to save QA results later to a dataframe
    ques_no_list = []
    question_list = []
    true_ans_list = []
    final_response_list = []

    #generate_evidence_batch is used to save evidence results in a batch
    # if the function has been called before and results are already saved then comment the function call
    # generate_evidence_batch(ques_id_list,query_list)

    with open(EVIDENCE_BATCH_SAVE_PATH, 'r') as json_file:
        evidence_batch_list = json.load(json_file)
        # print(evidence_batch_list)
        # exit(1)
    for i, d in enumerate(evidence_batch_list[:]):
        evidence_batch_list[i] = modify_evidence_batch_dict(d)
    
    # print(evidence_batch_list[0])

    for ques_id,query,true_ans,external_evidence in zip(ques_id_list,query_list,ans_list,evidence_batch_list):
    
        
        final_response = start_workflow(query,external_evidence,MODEL,WORKFLOW_RUN_COUNT)
        ques_no_list.append(ques_id)
        true_ans_list.append(true_ans)
        question_list.append(query)   
        final_response_list.append(final_response)

    if DATASET_NAME == "freshqa":
        qa_data_dict = {
            "ques_id":ques_no_list,
            "question":question_list,
            "true_ans":true_ans_list,
            "final_answer":final_response_list,
            "effective_year":effective_year_list,
            "num_hops":num_hops_list,
            "fact_type":fact_type_list
        }
    elif DATASET_NAME == "QAQA":
        qa_data_dict = {
            "ques_id":ques_no_list,
            "question":question_list,
            "true_ans":true_ans_list,
            "final_answer":final_response_list
        }

    print("$"*100)
    df = pd.DataFrame(qa_data_dict)
    print(df.head())
    print("$"*100)

    df.to_excel(RESULT_SAVE_PATH + MODEL + "alltest.xlsx",index=False)  # Set index=False to not write row indices


def start_evaluation():
    #list variable to save automatic evaluation results
    accuracy_result_list = []
    path = RESULT_SAVE_PATH + MODEL + "alltest.xlsx"
    df = pd.read_excel(path)
    query_list = df["question"].tolist()
    true_ans_list = df["true_ans"].tolist()
    final_answer_list = df["final_answer"].tolist()
    for query,true_ans,final_resp_text in zip(query_list,true_ans_list,final_answer_list):
        accuracy = auto_evaluation(query,true_ans,final_resp_text)
        accuracy_result_list.append(accuracy)

    df["accuracy"] = accuracy_result_list
    df.to_excel(RESULT_SAVE_PATH + MODEL + "alltest_eval.xlsx",index=False)

start_complete_workflow()
start_evaluation()