import numpy as np
import pandas as pd



def processing_fpdar(combined_result_list,qa_data_dict):
    # Unpack the combined_result_list into individual lists
    (
        original_response_list,fwd_final_response_list,fwd_final_resp_exp_list, 
        bck_final_response_list, bck_final_resp_exp_list,bck_final_question_list,
    ) = combined_result_list[:-1]
    
    # New keys and values to add
    result_dict = {
            "fwd_final_ans":fwd_final_response_list,
            "fwd_final_ans_exp":fwd_final_resp_exp_list,
            "bck_final_ans":bck_final_response_list,
            "bck_final_ans_exp":bck_final_resp_exp_list,
            "bck_final_question":bck_final_question_list,
            "original_response": original_response_list
        }
    # Add new keys and values to the existing dictionary
    qa_data_dict.update(result_dict)

    return qa_data_dict

def processing_sc(MAX_CANDIDATE_RESPONSES,combined_result_list,qa_data_dict):

    # Unpack the combined_result_list into individual lists
    (candidate_responses_list,final_response_list) = combined_result_list[:-1]

    result_dict = {
            "final_answer":final_response_list
        }
    # Add new keys and values to the existing dictionary
    qa_data_dict.update(result_dict)

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
    
    return qa_data_dict

def processing_fourshot(combined_result_list,qa_data_dict):
    pass

def start_result_processing(args,combined_result_list):

    if args.DATASET_NAME == "freshqa":
        ques_id_list, query_list, ans_list, effective_year_list, \
            num_hops_list, fact_type_list, premise_list = combined_result_list[-1]
        
        qa_data_dict = {
                "ques_id":ques_id_list,
                "question":query_list,
                "true_ans":ans_list,
                "effective_year":effective_year_list,
                "num_hops":num_hops_list,
                "fact_type":fact_type_list,
                "premise":premise_list
            }
        
    elif args.DATASET_NAME == "QAQA":
        ques_id_list, query_list, ans_list, premise_list = combined_result_list[-1]
        
        qa_data_dict = {
                "ques_id":ques_id_list,
                "question":query_list,
                "true_ans":ans_list,
                "all_assumptions_valid": premise_list
            }

    if args.METHOD == "FPDAR":
        qa_data_dict = processing_fpdar(combined_result_list,qa_data_dict)
    elif args.METHOD == "SC":
        qa_data_dict = processing_sc(args.MAX_CANDIDATE_RESPONSES,combined_result_list,qa_data_dict)
    elif args.METHOD == "FourShot":
        qa_data_dict = processing_fourshot(combined_result_list,qa_data_dict)

    return qa_data_dict