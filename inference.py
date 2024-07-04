import json
import pandas as pd
from importlib import import_module

from utils.dataset import start_dataset_processing
from utils.response_processing import start_response_processing
from utils.utils import generate_evidence_batch
from utils.utils import modify_evidence_batch_dict
from utils.utils import extract_value_from_single_key
from utils.web_search_serp import start_web_search


def start_fp_detc(args,query,external_evidence):
    # make sure that the names of the files in methods matches with the ones given here for import
    model_name = "gpt"
    if "mistral" in args.MODEL_API:
        model_name = "mistral"
        args.CANDIDATE_TEMPERATURE = 1.0
    elif "llama2" in args.MODEL_API:
        model_name = "llama2"
        args.CANDIDATE_TEMPERATURE = 1.0
    # Import the appropriate module dynamically based on the method
    fp_detc_module = import_module(f"Methods.{args.METHOD}.{model_name}")
    # Call the method-specific function from the imported module
    result = fp_detc_module.start_model_response(args,query,external_evidence)
    return result
 

def start_workflow(args):
    # list variables to save QA results later to a dataframe
    original_response_list = []

    fwd_final_response_list = []
    fwd_final_resp_exp_list = []

    bck_final_response_list = []
    bck_final_resp_exp_list = []
    bck_final_question_list = []

    full_response_save_path = (
        f"{args.RESULT_SAVE_PATH}{args.DATASET_NAME}_{args.ABLATION}_"
        f"{args.SIMILARITY_THRESHOLD}_{args.MODEL_API}"
    )

    if args.ABLATION == "Ab_Deduction":
        args.BACKWARD_REASONING_RESP_PROMPT_PATH = f"{args.PROMPT_PATH}{args.ABLATION}\\backward_reasoning_resp.txt"
        args.LLAMA_BACKWARD_REASONING_RESP_PROMPT_PATH = f"{args.PROMPT_PATH}{args.ABLATION}\\backward_reasoning_resp_meta.txt"

    if args.ABLATION == "Ab_ExtraInputQ" or args.ABLATION == "Ab_ExtraInputQ`":
        args.BACKWARD_REASONING_RESP_PROMPT_PATH = f"{args.PROMPT_PATH}{args.ABLATION}\\backward_reasoning_resp.txt"
        args.LLAMA_BACKWARD_REASONING_RESP_PROMPT_PATH = f"{args.PROMPT_PATH}{args.ABLATION}\\backward_reasoning_resp_meta.txt"


    # retrieve the datasets fields in proper formats
    dataset_elements = start_dataset_processing(args)
    ques_id_list, query_list = dataset_elements[:2]


    if args.EVIDENCE_BATCH_GENERATE and args.EVIDENCE_ALLOWED:
    #generate_evidence_batch is used to save evidence results in a batch
    # if the function has been called before and results are already save then comment the function call
    # OR make sure the default value of args.EVIDENCE_BATCH_GENERATE is not true
        generate_evidence_batch(args.EVIDENCE_BATCH_SAVE_PATH,ques_id_list, query_list)
        # print("test batch generate")

    if args.EVIDENCE_BATCH_USE and args.EVIDENCE_ALLOWED:
        with open(args.EVIDENCE_BATCH_SAVE_PATH, 'r') as json_file:
            evidence_batch_list = json.load(json_file)
        
        for i, d in enumerate(evidence_batch_list[:]):
            evidence_batch_list[i] = modify_evidence_batch_dict(d)
    else:
        # this is just used to ensure the loop below works with minimal changes
        # the dummy evidence_batch_list is used for fourshot method
        evidence_batch_list = [0] * len(ques_id_list)


    # fp detection loop for each individual questions starts here
    for ques_id,query,external_evidence in zip(ques_id_list,query_list,evidence_batch_list):
        
        if not args.EVIDENCE_BATCH_USE and args.EVIDENCE_ALLOWED:
        # this evidence will be used when the batch evidence is not being used
            initial_external_evidence = start_web_search(ques_id,query)
            external_evidence = modify_evidence_batch_dict(initial_external_evidence)
        # print("NEW QUERY HAS STARTED"*4)

        
        og_response_dict, fwd_main_answers_list, bck_main_answers_list = start_fp_detc(args,query,external_evidence)

        #appending results for qa
        original_response_list.append(og_response_dict)

        #appending results for forward reasoning
        fwd_final_response_list.append(fwd_main_answers_list[0])
        fwd_final_resp_exp_list.append(fwd_main_answers_list[1])

        #appending results for backward attack
        bck_extracted_final_question = extract_value_from_single_key(bck_main_answers_list[0], key = "Final Question:")

        if args.ABLATION == "Ab_AnswerQ`X`":
            bck_extracted_final_response = extract_value_from_single_key(bck_main_answers_list[1], key = "Answer:")
            bck_extracted_final_resp_exp = ""
        else:
            bck_extracted_final_response = extract_value_from_single_key(bck_main_answers_list[1], key = "Final Answer:")
            bck_extracted_final_resp_exp = extract_value_from_single_key(bck_main_answers_list[1], key = "Final Explanation:")
        
        bck_final_response_list.append(bck_extracted_final_response)
        bck_final_resp_exp_list.append(bck_extracted_final_resp_exp)
        bck_final_question_list.append(bck_extracted_final_question)

        combined_result_list = [original_response_list,fwd_final_response_list,fwd_final_resp_exp_list,
                                bck_final_response_list,bck_final_resp_exp_list,
                                bck_final_question_list,dataset_elements]


    qa_data_dict = start_response_processing(args,combined_result_list)
    
    print("$"*100)
    df = pd.DataFrame(qa_data_dict)
    print(df.head())
    print("$"*100)
    
    
    df.to_csv(full_response_save_path + "_responses.csv",index=False)