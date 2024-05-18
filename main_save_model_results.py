#This is the main file from where the execution of the workflow starts

import json
import yaml
import pandas as pd

from utils.dataset import start_dataset_processing
from utils.utils import generate_evidence_batch
from utils.utils import modify_evidence_batch_dict
from utils.utils import extract_value_from_single_key
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
EVIDENCE_BATCH_SAVE_PATH = config['EVIDENCE_BATCH_SAVE_PATH']
RESULT_SAVE_PATH = config['RESULT_SAVE_PATH']
DATASET_NAME = config['DATASET_NAME']
WORKFLOW_RUN_COUNT = config['WORKFLOW_RUN_COUNT']


# Func to start workflow for a query
def start_workflow(query,external_evidence,MODEL):
    # external_evidence = start_web_search(query)
    if MODEL in ["gpt-3.5-turbo-0125", "gpt-3.5-turbo-1106", "gpt-4-turbo-preview"]:
        result = start_openai_api_model_response(query,external_evidence)
    elif MODEL in ["mistral-small-latest", "mistral-small", "mistral-medium", "mistral-medium-latest", "mistral-large-latest"]:
        result = start_mistral_api_model_response(query,external_evidence)
    elif MODEL in ["meta.llama2-13b-chat-v1", "meta.llama2-70b-chat-v1"]:
        result = start_meta_api_model_response(query,external_evidence)
    else:
        print("Please enter a valid MODEL id in the next attempt for the workflow to execute")
    og_response_dict, fwd_main_answers_list, bck_main_answers_list = result
    return og_response_dict, fwd_main_answers_list, bck_main_answers_list
 

def start_complete_workflow():
    # processing specific to freshqa dataset
    ques_id_list, query_list, ans_list = start_dataset_processing(DATASET_NAME)

    # list variables to save QA results later to a dataframe
    ques_no_list = []
    question_list = []
    true_ans_list = []
    evidence_list = []
    original_response_list = []

    # list variables to save reasoning results later to a dataframe
    fwd_final_response_list = []
    fwd_final_resp_exp_list = []

    bck_final_response_list = []
    bck_final_resp_exp_list = []
    bck_final_question_list = []

    #generate_evidence_batch is used to save evidence results in a batch
    # if the function has been called before and results are already save then comment the function call
    # generate_evidence_batch(ques_id_list, query_list) 

    with open(EVIDENCE_BATCH_SAVE_PATH, 'r') as json_file:
        evidence_batch_list = json.load(json_file)
        # print(evidence_batch_list)
        # exit(1)
    
    for i, d in enumerate(evidence_batch_list[:]):
        evidence_batch_list[i] = modify_evidence_batch_dict(d)

                
    for ques_id,query,true_ans,external_evidence in zip(ques_id_list,query_list,ans_list,evidence_batch_list):
    
        print("NEW QUERY HAS STARTED"*4)
        og_response_dict, fwd_main_answers_list, bck_main_answers_list = start_workflow(query,external_evidence,MODEL)

        #appending results for qa
        ques_no_list.append(ques_id)
        true_ans_list.append(true_ans)
        evidence_list.append(external_evidence)
        question_list.append(query)
        original_response_list.append(og_response_dict)

        #appending results for forward reasoning
        fwd_final_response_list.append(fwd_main_answers_list[0])
        fwd_final_resp_exp_list.append(fwd_main_answers_list[1])

        #appending results for backward attack
        bck_extracted_final_question = extract_value_from_single_key(bck_main_answers_list[0], key = "Final Question:")
        bck_extracted_final_response = extract_value_from_single_key(bck_main_answers_list[1], key = "Final Answer:")
        bck_extracted_final_resp_exp = extract_value_from_single_key(bck_main_answers_list[1], key = "Final Explanation:")
        
        bck_final_response_list.append(bck_extracted_final_response)
        bck_final_resp_exp_list.append(bck_extracted_final_resp_exp)
        bck_final_question_list.append(bck_extracted_final_question)

    qa_data_dict = {
        "ques_id":ques_no_list,
        "question":question_list,
        "true_ans":true_ans_list,
        "fwd_final_ans":fwd_final_response_list,
        "fwd_final_ans_exp":fwd_final_resp_exp_list,
        "bck_final_ans":bck_final_response_list,
        "bck_final_ans_exp":bck_final_resp_exp_list,
        "bck_final_question":bck_final_question_list,
        "original_response": original_response_list,
        "evidence": evidence_list
    }

    print("$"*100)
    df = pd.DataFrame(qa_data_dict)
    print(df.head())
    print("$"*100)

    df.to_excel(RESULT_SAVE_PATH + MODEL + "21stApr_for_back_reasoningabd.xlsx",index=False)  # Set index=False to not write row indices

    adv_attack_data_dict = {
        "ques_id":ques_no_list,
        "question":question_list,
        "true_ans":true_ans_list,
        "fwd_final_ans":fwd_final_response_list,
        "fwd_final_ans_exp":fwd_final_resp_exp_list,
        "bck_final_ans":bck_final_response_list,
        "bck_final_ans_exp":bck_final_resp_exp_list,
        "bck_final_question":bck_final_question_list
    }

    df1 = pd.DataFrame(adv_attack_data_dict)

    # Initialize an empty dictionary to store the structured data
    structured_data = {}

    # Iterate through the DataFrame
    for _, row in df1.iterrows():
        ques_id = row['ques_id']
        if ques_id not in structured_data:
            structured_data[ques_id] = {
                'question': row['question'],
                'true_ans': row['true_ans'],
                'fwd_final_ans': row['fwd_final_ans'],
                'fwd_final_ans_exp': row['fwd_final_ans_exp'],
                'bck_final_ans': row['bck_final_ans'],
                'bck_final_ans_exp': row['bck_final_ans_exp'],
                'bck_final_question': row['bck_final_question']
            }

    # Convert the structured data dictionary to JSON format
    json_data = json.dumps(structured_data, indent=4)
    # Write the dictionary to a JSON file
    with open(RESULT_SAVE_PATH + MODEL + "21stApr_for_back_reasoningabd.json", 'w') as json_file:
        json_file.write(json_data)

def start_evaluation():
    #list variable to save automatic evaluation results
    final_accuracy_list = []
    same_answer_list = []
    same_question_list = []
    final_accuracy_comment_list = []
    path = RESULT_SAVE_PATH + MODEL + "30thMar_for_back_reasoningabd.xlsx"
    # path = "C:\GAMES_SETUP\Thesis\Code\Results\evidence_test_gpt-3.5-turbo-1106alltest.xlsx"
    df = pd.read_excel(path)#[160:175]
    query_list = df["question"].tolist()
    bck_extracted_final_question_list = df["bck_final_question"].tolist()
    true_ans_list = df["true_ans"].tolist()
    fwd_extracted_final_response_list = df["fwd_final_ans"].tolist()
    fwd_extracted_final_response_exp_list = df["fwd_final_ans_exp"].tolist()
    bwd_extracted_final_response_list = df["bck_final_ans"].tolist()
    bwd_extracted_final_response_exp_list = df["bck_final_ans_exp"].tolist()
    for query,bck_extracted_final_question,true_ans,fwd_extracted_final_response, \
        bck_extracted_final_response,fwd_extracted_final_resp_exp,bck_extracted_final_resp_exp \
        in zip(query_list,bck_extracted_final_question_list,true_ans_list,fwd_extracted_final_response_list,
               bwd_extracted_final_response_list,fwd_extracted_final_response_exp_list,bwd_extracted_final_response_exp_list):
        same_ques_resp, same_ans_resp, accuracy, comment = auto_evaluation(query,bck_extracted_final_question,true_ans,
                                                fwd_extracted_final_response,bck_extracted_final_response,
                                                fwd_extracted_final_resp_exp,bck_extracted_final_resp_exp)
        same_question_list.append(same_ques_resp)
        same_answer_list.append(same_ans_resp)
        final_accuracy_list.append(accuracy)
        final_accuracy_comment_list.append(comment)

    df["same_question"] = same_question_list
    df["same_answer"] = same_answer_list
    df["final_accuracy"] = final_accuracy_list
    df["final_accuracy_comment"] = final_accuracy_comment_list

    df.to_excel(RESULT_SAVE_PATH + MODEL + "21stApr_alltest_evalabd.xlsx",index=False)    


# start_complete_workflow()
start_evaluation()



