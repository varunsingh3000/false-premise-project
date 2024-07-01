#This is the main file from where the execution of the workflow starts

import json
import pandas as pd
import argparse

from utils.dataset import start_dataset_processing
from utils.utils import generate_evidence_batch
from utils.utils import modify_evidence_batch_dict
from utils.utils import extract_value_from_single_key
from utils.utils import auto_evaluation
from web_search_serp import start_web_search
from openai_gpt_models import start_openai_api_model_response
from mistral_models import start_mistral_api_model_response
from llama2 import start_meta_api_model_response


# Func to start workflow for a query
def start_workflow(query,external_evidence,MODEL):
    if MODEL in ["gpt-3.5-turbo-1106", "gpt-4-turbo-preview"]:
        result = start_openai_api_model_response(query,external_evidence)
    elif MODEL in ["mistral-small-latest"]:
        result = start_mistral_api_model_response(query,external_evidence)
    elif MODEL in ["meta.llama2-70b-chat-v1"]:
        result = start_meta_api_model_response(query,external_evidence)
    else:
        print("Please enter a valid MODEL id in the next attempt for the workflow to execute")
    og_response_dict, fwd_main_answers_list, bck_main_answers_list = result
    return og_response_dict, fwd_main_answers_list, bck_main_answers_list
 

def start_complete_workflow(args):
    dataset_elements = start_dataset_processing(args.DATASET_NAME)
    if len(dataset_elements) == 7:
        ques_id_list, query_list, ans_list, effective_year_list, \
        num_hops_list, fact_type_list, premise_list = dataset_elements
    elif len(dataset_elements) == 4:
        ques_id_list, query_list, ans_list, premise_list = dataset_elements

    # list variables to save QA results later to a dataframe
    ques_no_list = []
    question_list = []
    true_ans_list = []
    original_response_list = []

    # list variables to save reasoning results later to a dataframe
    fwd_final_response_list = []
    fwd_final_resp_exp_list = []

    bck_final_response_list = []
    bck_final_resp_exp_list = []
    bck_final_question_list = []

    if args.EVIDENCE_BATCH_GENERATE:
    #generate_evidence_batch is used to save evidence results in a batch
    # if the function has been called before and results are already save then comment the function call
    # OR make sure the default value of args.EVIDENCE_BATCH_GENERATE is not true
        # generate_evidence_batch(ques_id_list, query_list)
        print("test batch generate")

    if args.EVIDENCE_BATCH_USE:
        with open(args.EVIDENCE_BATCH_SAVE_PATH, 'r') as json_file:
            evidence_batch_list = json.load(json_file)
        
        for i, d in enumerate(evidence_batch_list[:]):
            evidence_batch_list[i] = modify_evidence_batch_dict(d)
    else:
        # this is just used to ensure the loop below works with minimal changes
        evidence_batch_list = [0] * len(ques_id_list)
                
    for ques_id,query,true_ans,external_evidence in zip(ques_id_list,query_list,ans_list,evidence_batch_list):
        
        if not args.EVIDENCE_BATCH_USE:
        # this evidence will be used when the batch evidence is not being used
            external_evidence = start_web_search(ques_id,query)
        print(external_evidence)
        # print("NEW QUERY HAS STARTED"*4)
        og_response_dict, fwd_main_answers_list, bck_main_answers_list = start_workflow(query,external_evidence,args.MODEL)

        #appending results for qa
        ques_no_list.append(ques_id)
        true_ans_list.append(true_ans)
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

    if args.DATASET_NAME == "freshqa":
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
            "effective_year":effective_year_list,
            "num_hops":num_hops_list,
            "fact_type":fact_type_list,
            "premise":premise_list
        }
    elif args.DATASET_NAME == "QAQA":
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
            "all_assumptions_valid": premise_list
        }

    print("$"*100)
    df = pd.DataFrame(qa_data_dict)
    print(df.head())
    print("$"*100)
    
    df.to_excel(args.RESULT_SAVE_PATH + args.MODEL + "back_reasoningabd.xlsx",index=False)  # Set index=False to not write row indices


def start_evaluation(args):
    #list variable to save automatic evaluation results
    final_accuracy_list = []
    same_question_list = []
    path = args.RESULT_SAVE_PATH + args.MODEL + "back_reasoningabd.xlsx"
    # path = "C:\GAMES_SETUP\Thesis\Code\Results\evidence_test_meta.llama2-70b-chat-v121stApr_for_back_reasoningabd.xlsx"
    df = pd.read_excel(path)
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
        same_ques_resp, accuracy = auto_evaluation(query,bck_extracted_final_question,true_ans,
                                                fwd_extracted_final_response,bck_extracted_final_response,
                                                fwd_extracted_final_resp_exp,bck_extracted_final_resp_exp)
        same_question_list.append(same_ques_resp)
        final_accuracy_list.append(accuracy)

    df["same_question"] = same_question_list
    df["final_accuracy"] = final_accuracy_list

    print(df.head())
    df.to_excel(args.RESULT_SAVE_PATH + args.MODEL + "evalabd.xlsx",index=False)    


def create_parser():
    parser = argparse.ArgumentParser(description="Script parameters")
    
    parser.add_argument('--MODEL', type=str, choices=["gpt-3.5-turbo-1106", "mistral-small-latest", "meta.llama2-70b-chat-v1"],
                        default="gpt-3.5-turbo-1106", help='Model to use')
    parser.add_argument('--EVAL_MODEL', type=str, default="gpt-4-turbo-preview", help='Evaluation model to use')
    parser.add_argument('--TEMPERATURE', type=float, default=0.0, help='Temperature setting for the model')
    parser.add_argument('--CANDIDATE_TEMPERATURE', type=float, default=1.0, help='Candidate temperature setting for the model')
    parser.add_argument('--DATASET_NAME', type=str, choices=["freshqa", "QAQA"], default="freshqa", help='Dataset name to use')
    parser.add_argument('--DATASET_PATH', type=str, default="Data\\", help='Path to the dataset')
    parser.add_argument('--EVIDENCE_BATCH_GENERATE', type=int, default=0, help='Boolean to decide whether evidence is generated in batch or indiviudally for each question. Zero means by default batch will not be generated.')
    parser.add_argument('--EVIDENCE_BATCH_USE', type=int, default=1, help='Boolean to decide whether batch evidence has to be used. One means by default batch will be used.')
    parser.add_argument('--EVIDENCE_BATCH_SAVE_PATH', type=str, default="Web_Search_Response\\evidence_results_batch_serp_all_freshqa.json", help='Path to save evidence batch results')
    parser.add_argument('--QUERY_PROMPT_PATH', type=str, default="Prompts\\minimal_response.txt", help='Path to the query prompt file')
    parser.add_argument('--BACKWARD_REASONING_RESP_PROMPT_PATH', type=str, default="Prompts\\backward_reasoning_resp.txt", help='Path to the backward reasoning response prompt file')
    parser.add_argument('--BACKWARD_REASONING_QUERY_PROMPT_PATH', type=str, default="Prompts\\backward_reasoning_query.txt", help='Path to the backward reasoning query prompt file')
    parser.add_argument('--LLAMA_QUERY_PROMPT_PATH', type=str, default="Prompts\\minimal_response_meta.txt", help='Path to the LLAMA query prompt file')
    parser.add_argument('--LLAMA_BACKWARD_REASONING_RESP_PROMPT_PATH', type=str, default="Prompts\\backward_reasoning_resp_meta.txt", help='Path to the LLAMA backward reasoning response prompt file')
    parser.add_argument('--LLAMA_BACKWARD_REASONING_QUERY_PROMPT_PATH', type=str, default="Prompts\\backward_reasoning_query_meta.txt", help='Path to the LLAMA backward reasoning query prompt file')
    parser.add_argument('--AUTO_EVALUATION_PROMPT_PATH', type=str, default="Prompts\\eval_response_comp.txt", help='Path to the auto evaluation prompt file')
    parser.add_argument('--RESULT_SAVE_PATH', type=str, default="Results\\", help='Path to save results')
    parser.add_argument('--MAX_CANDIDATE_RESPONSES', type=int, default=3, help='Maximum candidate responses')

    return parser

def main():
    
    parser = create_parser()
    
    # Parse the command-line arguments
    args = parser.parse_args()
    
    # Print the parsed arguments (or use them as needed)
    print(args)
    start_complete_workflow(args)
    start_evaluation(args)


if __name__ == "__main__":
    main()