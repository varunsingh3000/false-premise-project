#This is the main file from where the execution of the workflow starts

import json
import yaml
import pandas as pd

from utils.dataset import start_dataset_processing
from utils.utils import generate_evidence_batch
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
#defines the max no of candidate responses to generate, the actual no of responses could vary depending on the matching condition
MAX_CANDIDATE_RESPONSES = config['MAX_CANDIDATE_RESPONSES']
WORKFLOW_RUN_COUNT = config['WORKFLOW_RUN_COUNT']


# Func to start workflow for a query
def start_workflow(query,external_evidence,MODEL):
    # external_evidence = start_web_search(query)
    if MODEL in ["gpt-3.5-turbo", "gpt-3.5-turbo-1106", "gpt-4-turbo-preview"]:
        result = start_openai_api_model_response(query,external_evidence)
    elif MODEL in ["mistral-tiny", "mistral-small", "mistral-medium", "mistral-large-latest"]:
        result = start_mistral_api_model_response(query,external_evidence)
    elif MODEL in ["meta.llama2-13b-chat-v1", "meta.llama2-70b-chat-v1"]:
        result = start_meta_api_model_response(query,external_evidence)
    else:
        print("Please enter a valid MODEL id in the next attempt for the workflow to execute")
    og_response_dict, adv_attack_response_list, main_answers_list, all_responses_list = result
    return og_response_dict, adv_attack_response_list, main_answers_list, all_responses_list
 

def start_complete_workflow():
    # processing specific to freshqa dataset
    ques_id_list, query_list, ans_list = start_dataset_processing(DATASET_NAME)

    # list variables to save QA results later to a dataframe
    ques_no_list = []
    question_list = []
    true_ans_list = []
    evidence_list = []
    original_response_list = []

    # list variables to save adversarial attack results later to a dataframe
    adv_attack_resp1_list = []
    adv_attack_resp2_list = []
    adv_attack_resp3_list = []
    adv_attack_resp4_list = []
    adv_attack_resp5_list = []
    adv_original_response_list = []
    adv_final_response_list = []
    adv_final_resp_exp_list = []

    #list variable to save automatic evaluation results
    accuracy_result_list = []
    # beliefans_list = []
    og_resp_veracity_list = []
    resp_change_list = []
    #generate_evidence_batch is used to save evidence results in a batch
    # if the function has been called before and results are already save then comment the function call
    # generate_evidence_batch(ques_id_list, query_list) 

    with open(EVIDENCE_BATCH_SAVE_PATH, 'r') as json_file:
        evidence_batch_list = json.load(json_file)
                
    for ques_id,query,true_ans,external_evidence in zip(ques_id_list,query_list,ans_list,evidence_batch_list):
    
        print("NEW QUERY HAS STARTED"*4)
        og_response_dict, adv_attack_response_list, main_answers_list, all_responses_list = start_workflow(query,
                                                                    external_evidence,MODEL)

        #appending results for qa
        ques_no_list.append(ques_id)
        true_ans_list.append(true_ans)
        evidence_list.append(external_evidence)
        question_list.append(query)
        original_response_list.append(og_response_dict)

        #appending results for adversarial attack
        adv_attack_resp1_list.append(adv_attack_response_list[0])
        adv_attack_resp2_list.append(adv_attack_response_list[1])
        adv_attack_resp3_list.append(adv_attack_response_list[2])
        adv_attack_resp4_list.append(adv_attack_response_list[3])
        adv_attack_resp5_list.append(adv_attack_response_list[4])
        adv_original_response_list.append(main_answers_list[0])
        extracted_final_response = extract_value_from_single_key(main_answers_list[1], key = "Final Answer:")
        extracted_final_resp_exp = extract_value_from_single_key(main_answers_list[1], key = "Explanation:")
        adv_final_response_list.append(extracted_final_response)
        adv_final_resp_exp_list.append(extracted_final_resp_exp)

        #appending results for accuracy
        all_responses_list.append(true_ans) # adding ground truth response for comparison
        accuracy_response, task1, task2 = auto_evaluation(true_ans,extracted_final_response + " " + extracted_final_resp_exp,all_responses_list)
        accuracy_result_list.append(accuracy_response)
        # beliefans_list.append(beliefans)
        og_resp_veracity_list.append(task1)
        resp_change_list.append(task2)

    print("ADVERSARIAL ATTACK LIST: ", adv_attack_response_list)

    qa_data_dict = {
        "ques_id":ques_no_list,
        "question":question_list,
        "true_ans":true_ans_list,
        "final_ans":adv_final_response_list,
        "final_ans_exp":adv_final_resp_exp_list,
        "accuracy":accuracy_result_list,
        # "beliefans":beliefans_list,
        "og_resp_veracity":og_resp_veracity_list,
        "resp_change_label":resp_change_list,
        "original_response": original_response_list,
        "evidence": evidence_list
    }

    print("$"*100)
    df = pd.DataFrame(qa_data_dict)
    print(df.head())
    print("$"*100)

    df.to_excel(RESULT_SAVE_PATH + MODEL + "belieftest_all.xlsx",index=False)  # Set index=False to not write row indices

    adv_attack_data_dict = {
        "ques_id":ques_no_list,
        "question":question_list,
        "true_ans":true_ans_list,
        "first_ans":adv_original_response_list,
        "final_ans":adv_final_response_list,
        "final_ans_exp":adv_final_resp_exp_list,
        "adv_attack1": adv_attack_resp1_list,
        "adv_attack2": adv_attack_resp2_list,
        "adv_attack3": adv_attack_resp3_list,
        "adv_attack4": adv_attack_resp4_list,
        "adv_attack5": adv_attack_resp5_list
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
                'first_ans': row['first_ans'],
                'final_ans': row['final_ans'],
                'final_ans_exp': row['final_ans_exp'],
                'adv_attacks': [row['adv_attack1'], row['adv_attack2'], row['adv_attack3'], row['adv_attack4'], row['adv_attack5']]
            }

    # Convert the structured data dictionary to JSON format
    json_data = json.dumps(structured_data, indent=4)
    # Write the dictionary to a JSON file
    with open(RESULT_SAVE_PATH + MODEL + "belieftest_all.json", 'w') as json_file:
        json_file.write(json_data)

    # df1 = pd.DataFrame(adv_attack_data_dict)
    # print(df1.head())
    # df1.to_csv(RESULT_SAVE_PATH + MODEL + "adv_attack.csv",index=False)  # Set index=False to not write row indices

start_complete_workflow()





