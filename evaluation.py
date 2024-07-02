import pandas as pd

from utils.evaluation_processing import auto_evaluation

def start_evaluation(args):
    # getting the necessary fields from the generated response csv according to method
    accuracy_list = []
    
    if args.METHOD == "FPDAR":
        full_response_save_path = (
            f"{args.RESULT_SAVE_PATH}{args.DATASET_NAME}_{args.METHOD}_"
            f"{args.SIMILARITY_THRESHOLD}_{args.MODEL_API}"
        )

        df = pd.read_csv(full_response_save_path + "_responses.csv")
        similarity_list = []
        fwd_extracted_final_response_list = df["fwd_final_ans"].tolist()
        fwd_extracted_final_response_exp_list = df["fwd_final_ans_exp"].tolist()
        bwd_extracted_final_response_list = df["bck_final_ans"].tolist()
        bwd_extracted_final_response_exp_list = df["bck_final_ans_exp"].tolist()
        bck_extracted_final_question_list = df["bck_final_question"].tolist()

    elif args.METHOD == "SC" or args.METHOD == "FourShot":
        full_response_save_path = (
            f"{args.RESULT_SAVE_PATH}{args.DATASET_NAME}_{args.METHOD}_"
            f"{args.CANDIDATE_TEMPERATURE}_{args.MODEL_API}"
        )

        df = pd.read_csv(full_response_save_path + "_responses.csv")
        final_answer_list = df["final_answer"].tolist()
    
    query_list = df["question"].tolist()
    true_ans_list = df["true_ans"].tolist()
    
    # calling the evaluation function according to method
    if args.METHOD == "FPDAR":
        for query,bck_extracted_final_question,true_ans,fwd_extracted_final_response, \
            bck_extracted_final_response,fwd_extracted_final_resp_exp,bck_extracted_final_resp_exp \
            in zip(query_list,bck_extracted_final_question_list,true_ans_list,fwd_extracted_final_response_list,
                bwd_extracted_final_response_list,fwd_extracted_final_response_exp_list,bwd_extracted_final_response_exp_list):
            
            # fwd_extracted_final_response,fwd_extracted_final_resp_exp,bck_extracted_final_response,
            # bck_extracted_final_resp_exp = map(str,(fwd_extracted_final_response,
            #     fwd_extracted_final_resp_exp,bck_extracted_final_response,bck_extracted_final_resp_exp))

            similarity, accuracy = auto_evaluation(args,query,bck_extracted_final_question,true_ans,
                                                    fwd_extracted_final_response,bck_extracted_final_response,
                                                    fwd_extracted_final_resp_exp,bck_extracted_final_resp_exp)
            similarity_list.append(similarity)
            accuracy_list.append(accuracy)
        
        df["same_question"] = similarity_list

    elif args.METHOD == "SC" or args.METHOD == "FourShot":
        for query,true_ans,final_resp_text in zip(query_list,true_ans_list,final_answer_list):
            accuracy = auto_evaluation(args,query,true_ans,final_resp_text)
            accuracy_list.append(accuracy)

    
    df["final_accuracy"] = accuracy_list

    print(df.head())

    df.to_csv(full_response_save_path + "_eval.csv",index=False)

