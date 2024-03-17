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
#defines the max no of candidate responses to generate, the actual no of responses could vary depending on the matching condition
MAX_CANDIDATE_RESPONSES = config['MAX_CANDIDATE_RESPONSES']
WORKFLOW_RUN_COUNT = config['WORKFLOW_RUN_COUNT']


# Func to start workflow for a query
def start_workflow(query,external_evidence,MODEL):
    # external_evidence = start_web_search(query)
    if MODEL in ["gpt-3.5-turbo-0125", "gpt-3.5-turbo-1106", "gpt-4-turbo-preview"]:
        result = start_openai_api_model_response(query,external_evidence)
    elif MODEL in ["mistral-tiny", "mistral-small", "mistral-medium", "mistral-medium-latest", "mistral-large-latest"]:
        result = start_mistral_api_model_response(query,external_evidence)
    elif MODEL in ["meta.llama2-13b-chat-v1", "meta.llama2-70b-chat-v1"]:
        result = start_meta_api_model_response(query,external_evidence)
    else:
        print("Please enter a valid MODEL id in the next attempt for the workflow to execute")
    og_response_dict, forward_reasoning_list, backward_reasoning_list, fwd_main_answers_list, \
                bck_main_answers_list, all_responses_list = result
    return og_response_dict, forward_reasoning_list, backward_reasoning_list, fwd_main_answers_list, \
                bck_main_answers_list, all_responses_list
 

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
    fwd_reasoning_resp1_list = []
    fwd_reasoning_resp2_list = []
    fwd_reasoning_resp3_list = []
    fwd_reasoning_resp4_list = []
    fwd_original_response_list = []
    fwd_final_response_list = []
    fwd_final_resp_exp_list = []

    bck_reasoning_resp1_list = []
    bck_reasoning_resp2_list = []
    bck_reasoning_resp3_list = []
    bck_reasoning_resp4_list = []
    bck_final_response_list = []
    bck_final_resp_exp_list = []
    bck_final_question_list = []

    #list variable to save automatic evaluation results
    final_accuracy_list = []
    same_answer_list = []
    same_question_list = []
    final_accuracy_comment_list = []
    #generate_evidence_batch is used to save evidence results in a batch
    # if the function has been called before and results are already save then comment the function call
    # generate_evidence_batch(ques_id_list, query_list) 

    with open(EVIDENCE_BATCH_SAVE_PATH, 'r') as json_file:
        evidence_batch_list = json.load(json_file)
        # print(evidence_batch_list)
        # exit(1)
    
    for i, d in enumerate(evidence_batch_list):
        evidence_batch_list[i] = modify_evidence_batch_dict(d)

                
    for ques_id,query,true_ans,external_evidence in zip(ques_id_list,query_list,ans_list,evidence_batch_list):
    
        print("NEW QUERY HAS STARTED"*4)
        og_response_dict, forward_reasoning_list, backward_reasoning_list, fwd_main_answers_list, \
                            bck_main_answers_list, all_responses_list = start_workflow(query,external_evidence,MODEL)

        #appending results for qa
        ques_no_list.append(ques_id)
        true_ans_list.append(true_ans)
        evidence_list.append(external_evidence)
        question_list.append(query)
        original_response_list.append(og_response_dict)

        #appending results for forward reasoning
        fwd_reasoning_resp1_list.append(forward_reasoning_list[0])
        # fwd_reasoning_resp2_list.append(forward_reasoning_list[1])
        # fwd_reasoning_resp3_list.append(forward_reasoning_list[2])
        # fwd_reasoning_resp4_list.append(forward_reasoning_list[3])
        fwd_original_response_list.append(fwd_main_answers_list[0])
        fwd_extracted_final_response = extract_value_from_single_key(fwd_main_answers_list[1], key = "Forward Answer:")
        fwd_extracted_final_resp_exp = extract_value_from_single_key(fwd_main_answers_list[1], key = "Forward Explanation:")
        fwd_final_response_list.append(fwd_extracted_final_response)
        fwd_final_resp_exp_list.append(fwd_extracted_final_resp_exp)

        #appending results for backward attack
        bck_reasoning_resp1_list.append(backward_reasoning_list[0])
        # bck_reasoning_resp2_list.append(backward_reasoning_list[1])
        # bck_reasoning_resp3_list.append(backward_reasoning_list[2])
        # bck_reasoning_resp4_list.append(backward_reasoning_list[3])
        bck_extracted_final_response = extract_value_from_single_key(bck_main_answers_list[0], key = "Final Answer:")
        bck_extracted_final_resp_exp = extract_value_from_single_key(bck_main_answers_list[0], key = "Final Explanation:")
        bck_extracted_final_question = extract_value_from_single_key(bck_main_answers_list[0], key = "Final Question:")
        bck_final_response_list.append(bck_extracted_final_response)
        bck_final_resp_exp_list.append(bck_extracted_final_resp_exp)
        bck_final_question_list.append(bck_extracted_final_question)

        #auto evaluation
        same_ques_resp, same_ans_resp, accuracy, comment = auto_evaluation(query,bck_extracted_final_question,true_ans,
                                                fwd_extracted_final_response,bck_extracted_final_response,
                                                fwd_extracted_final_resp_exp,bck_extracted_final_resp_exp)
        same_question_list.append(same_ques_resp)
        same_answer_list.append(same_ans_resp)
        final_accuracy_list.append(accuracy)
        final_accuracy_comment_list.append(comment)


    print("ADVERSARIAL ATTACK LIST: ", forward_reasoning_list)

    qa_data_dict = {
        "ques_id":ques_no_list,
        "question":question_list,
        "true_ans":true_ans_list,
        "fwd_final_ans":fwd_final_response_list,
        "fwd_final_ans_exp":fwd_final_resp_exp_list,
        "bck_final_ans":bck_final_response_list,
        "bck_final_ans_exp":bck_final_resp_exp_list,
        "bck_final_question":bck_final_question_list,
        "same_question":same_question_list,
        "same_answer":same_answer_list,
        "final_accuracy":final_accuracy_list,
        "final_accuracy_comment":final_accuracy_comment_list,
        "original_response": original_response_list,
        "evidence": evidence_list
    }

    print("$"*100)
    df = pd.DataFrame(qa_data_dict)
    print(df.head())
    print("$"*100)

    df.to_excel(RESULT_SAVE_PATH + MODEL + "16thMar_for_back_reasoning.xlsx",index=False)  # Set index=False to not write row indices

    adv_attack_data_dict = {
        "ques_id":ques_no_list,
        "question":question_list,
        "true_ans":true_ans_list,
        "first_ans":fwd_original_response_list,
        "fwd_final_ans":fwd_final_response_list,
        "fwd_final_ans_exp":fwd_final_resp_exp_list,
        "bck_final_ans":bck_final_response_list,
        "bck_final_ans_exp":bck_final_resp_exp_list,
        "fwd_resp_1": fwd_reasoning_resp1_list,
        # "fwd_resp_2": fwd_reasoning_resp2_list,
        # "fwd_resp_3": fwd_reasoning_resp3_list,
        # "fwd_resp_4": fwd_reasoning_resp4_list,
        "bck_resp_1": bck_reasoning_resp1_list
        # "bck_resp_2": bck_reasoning_resp2_list,
        # "bck_resp_3": bck_reasoning_resp3_list,
        # "bck_resp_4": bck_reasoning_resp4_list,
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
                'fwd_final_ans': row['fwd_final_ans'],
                'fwd_final_ans_exp': row['fwd_final_ans_exp'],
                'bck_final_ans': row['bck_final_ans'],
                'bck_final_ans_exp': row['bck_final_ans_exp'],
                'forw_reasoning': [row['fwd_resp_1']],
                'back_reasoning': [row['bck_resp_1']]
                # 'forw_reasoning': [row['fwd_resp_1'], row['fwd_resp_2'], row['fwd_resp_3'], row['fwd_resp_4']],
                # 'back_reasoning': [row['bck_resp_1'], row['bck_resp_2'], row['bck_resp_3'], row['bck_resp_4']]
            }

    # Convert the structured data dictionary to JSON format
    json_data = json.dumps(structured_data, indent=4)
    # Write the dictionary to a JSON file
    with open(RESULT_SAVE_PATH + MODEL + "16thMar_for_back_reasoning.json", 'w') as json_file:
        json_file.write(json_data)

    # df1 = pd.DataFrame(adv_attack_data_dict)
    # print(df1.head())
    # df1.to_csv(RESULT_SAVE_PATH + MODEL + "adv_attack.csv",index=False)  # Set index=False to not write row indices

start_complete_workflow()





