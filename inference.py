#This is the main file from where the execution of the workflow starts

import json
import pandas as pd
import argparse
from importlib import import_module

from utils.dataset import start_dataset_processing
from utils.result_processing import start_result_processing
from utils.utils import generate_evidence_batch
from utils.utils import modify_evidence_batch_dict
from utils.utils import extract_value_from_single_key
from utils.utils import auto_evaluation

from web_search_serp import start_web_search


def start_fp_detc(args,query,external_evidence):
    # make sure that the names of the files in methods matches with the ones given here for import
    model_name = "gpt"
    if "mistral" in args.MODEL_API:
        model_name = "mistral"
    elif "llama2" in args.MODEL_API:
        model_name = "llama2"
    # Import the appropriate module dynamically based on the method
    fp_detc_module = import_module(f"Methods.{args.METHOD}.{model_name}")
    # Call the method-specific function from the imported module
    result = fp_detc_module.start_model_response(args,query,external_evidence)
    return result
 

def start_workflow(args):

    if args.METHOD == "FPDAR":
        # list variables to save QA results later to a dataframe
        original_response_list = []

        fwd_final_response_list = []
        fwd_final_resp_exp_list = []

        bck_final_response_list = []
        bck_final_resp_exp_list = []
        bck_final_question_list = []

        full_response_save_path = (
            f"{args.RESULT_SAVE_PATH}{args.DATASET_NAME}_{args.METHOD}_"
            f"{args.SIMILARITY_THRESHOLD}_{args.MODEL_API}"
        )


    elif args.METHOD == "SC":
        # list variables to save QA results later to a dataframe
        final_ans_dict_list = []
        candidate_responses_list = []
        final_response_list = []

        full_response_save_path = (
            f"{args.RESULT_SAVE_PATH}{args.DATASET_NAME}_{args.METHOD}_"
            f"{args.CANDIDATE_TEMPERATURE}_{args.MODEL_API}"
        )

    elif args.METHOD == "FourShot":
        # list variables to save QA results later to a dataframe
        final_response_list = []
        # fourshot method has different prompts without evidence, so those are assigned here
        args.QUERY_PROMPT_PATH = f"{args.PROMPT_PATH}{args.METHOD}\\resp_generation.txt"
        args.LLAMA_QUERY_PROMPT_PATH = f"{args.PROMPT_PATH}{args.METHOD}\\resp_generation_meta.txt"
        args.EVIDENCE_ALLOWED = 0

    # retrieve the datasets fields in proper formats
    dataset_elements = start_dataset_processing(args)
    ques_id_list, query_list = dataset_elements[:2]


    if args.EVIDENCE_BATCH_GENERATE and args.EVIDENCE_ALLOWED:
    #generate_evidence_batch is used to save evidence results in a batch
    # if the function has been called before and results are already save then comment the function call
    # OR make sure the default value of args.EVIDENCE_BATCH_GENERATE is not true
        # generate_evidence_batch(args.EVIDENCE_BATCH_SAVE_PATH,ques_id_list, query_list)
        print("test batch generate")

    if args.EVIDENCE_BATCH_USE and args.EVIDENCE_ALLOWED:
        with open(args.EVIDENCE_BATCH_SAVE_PATH, 'r') as json_file:
            evidence_batch_list = json.load(json_file)
        
        for i, d in enumerate(evidence_batch_list[:]):
            evidence_batch_list[i] = modify_evidence_batch_dict(d)
    else:
        # this is just used to ensure the loop below works with minimal changes
        evidence_batch_list = [0] * len(ques_id_list)
                
    # fp detection loop for each individual questions starts here
    for ques_id,query,external_evidence in zip(ques_id_list,query_list,evidence_batch_list):
        
        if not args.EVIDENCE_BATCH_USE and args.EVIDENCE_ALLOWED:
        # this evidence will be used when the batch evidence is not being used
            external_evidence = start_web_search(ques_id,query)
        # print("NEW QUERY HAS STARTED"*4)

        if args.METHOD == "FPDAR":
            og_response_dict, fwd_main_answers_list, bck_main_answers_list = start_fp_detc(args,query,external_evidence)

            #appending results for qa
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

            combined_result_list = [original_response_list,fwd_final_response_list,fwd_final_resp_exp_list,
                                    bck_final_response_list,bck_final_resp_exp_list,
                                    bck_final_question_list,dataset_elements]

        elif args.METHOD == "SC":
            responses_dict, final_response = start_fp_detc(args,query,external_evidence)

            temp_candidate_response_list = []
            temp_indi_resp_list=[]
            for key_value in responses_dict:
                final_ans_dict_list.append(final_response)

                for candidate_resp in responses_dict[key_value][3:]:  
                    temp_indi_resp_list.append(candidate_resp)
                temp_candidate_response_list.append(temp_indi_resp_list)
                temp_indi_resp_list = []
            candidate_responses_list.append(temp_candidate_response_list)

            final_response_list.append(final_response)

            combined_result_list = [candidate_responses_list,final_response_list,dataset_elements]
        
        elif args.METHOD == "FourShot":
            final_response = start_fp_detc(args,query,external_evidence)
            final_response_list.append(final_response)
            combined_result_list = [final_response_list,dataset_elements]

    qa_data_dict = start_result_processing(args,combined_result_list)
    
    print("$"*100)
    df = pd.DataFrame(qa_data_dict)
    print(df.head())
    print("$"*100)
    
    
    df.to_csv(full_response_save_path + "_responses.csv",index=False)


def start_evaluation(args):
    #list variable to save automatic evaluation results
    final_accuracy_list = []
    same_question_list = []
    path = args.RESULT_SAVE_PATH + args.MODEL_API + "back_reasoningabd.csv"
    # path = "C:\GAMES_SETUP\Thesis\Code\Results\evidence_test_meta.llama2-70b-chat-v121stApr_for_back_reasoningabd.xlsx"
    df = pd.read_csv(path)
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
    df.to_csv(args.RESULT_SAVE_PATH + args.MODEL_API + "evalabd.csv",index=False)


def create_parser():
    parser = argparse.ArgumentParser(description="Script parameters")
    
    parser.add_argument('--METHOD', type=str, choices=["FPDAR", "SC", "FourShot"],
                        default="FPDAR", help='Method to use. This will decide which prompts are used going further.')
    parser.add_argument('--MODEL_API', type=str, choices=["gpt-3.5-turbo-1106", "mistral-small-latest", "meta.llama2-70b-chat-v1"],
                        default="gpt-3.5-turbo-1106", help='Model API to use')
    parser.add_argument('--EVAL_MODEL', type=str, default="gpt-4-turbo-preview", help='Evaluation model to use')
    parser.add_argument('--DATASET_NAME', type=str, choices=["freshqa", "QAQA"], default="freshqa", help='Dataset name to use')
    parser.add_argument('--DATASET_PATH', type=str, default="Data\\", help='Path to the dataset')
    parser.add_argument('--PROMPT_PATH', type=str, default="Prompts\\", help='Path to the prompts')
    parser.add_argument('--SIMILARITY_THRESHOLD', type=float, default=0.7, help='Similarity threshold value for FPDAR FP Detection')
    parser.add_argument('--TEMPERATURE', type=float, default=0.0, help='Temperature setting for the model')
    parser.add_argument('--CANDIDATE_TEMPERATURE', type=float, default=1.0, help='Candidate temperature setting for the model')
    parser.add_argument('--MAX_CANDIDATE_RESPONSES', type=int, default=3, help='Maximum candidate responses to be generated for SC method')
    parser.add_argument('--EVIDENCE_ALLOWED', type=int, default=1, help='Boolean to decide to whether evidence should be used')
    parser.add_argument('--EVIDENCE_BATCH_GENERATE', type=int, default=0, help='Boolean to decide whether evidence is generated in batch or indiviudally for each question. Zero means by default batch will not be generated.')
    parser.add_argument('--EVIDENCE_BATCH_USE', type=int, default=1, help='Boolean to decide whether batch evidence has to be used. One means by default batch will be used.')
    parser.add_argument('--EVIDENCE_BATCH_SAVE_PATH', type=str, default="Web_Search_Response\\evidence_results_batch_serp_all_freshqa.json", help='Path to save evidence batch results')
    parser.add_argument('--QUERY_PROMPT_PATH', type=str, default="Prompts\\Common\\resp_generation.txt", help='Path to the query prompt file')
    parser.add_argument('--BACKWARD_REASONING_RESP_PROMPT_PATH', type=str, default="Prompts\\FPDAR\\backward_reasoning_resp.txt", help='Path to the backward reasoning response prompt file')
    parser.add_argument('--BACKWARD_REASONING_QUERY_PROMPT_PATH', type=str, default="Prompts\\FPDAR\\backward_reasoning_query.txt", help='Path to the backward reasoning query prompt file')
    parser.add_argument('--LLAMA_QUERY_PROMPT_PATH', type=str, default="Prompts\\Common\\resp_generation_meta.txt", help='Path to the LLAMA query prompt file')
    parser.add_argument('--LLAMA_BACKWARD_REASONING_RESP_PROMPT_PATH', type=str, default="Prompts\\FPDAR\\backward_reasoning_resp_meta.txt", help='Path to the LLAMA backward reasoning response prompt file')
    parser.add_argument('--LLAMA_BACKWARD_REASONING_QUERY_PROMPT_PATH', type=str, default="Prompts\\FPDAR\\backward_reasoning_query_meta.txt", help='Path to the LLAMA backward reasoning query prompt file')
    parser.add_argument('--AUTO_EVALUATION_PROMPT_PATH', type=str, default="Prompts\\Common\\response_evaluation.txt", help='Path to the auto evaluation prompt file')
    parser.add_argument('--RESULT_SAVE_PATH', type=str, default="Results\\", help='Path to save results')
    
    return parser

def main():
    parser = create_parser()
    # Parse the command-line arguments
    args = parser.parse_args()
    # Print the parsed arguments (or use them as needed)
    print(args)
    # Start the main fp detection workflow and prepare the responses to be processed for evaluation
    start_workflow(args)
    # start_evaluation(args)


if __name__ == "__main__":
    main()