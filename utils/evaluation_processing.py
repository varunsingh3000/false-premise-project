import argparse
from multipledispatch import dispatch
from sentence_transformers import SentenceTransformer
from scipy.spatial.distance import cosine

from utils.utils import extract_value_from_single_key
from utils.utils import perform_gpt_response
from utils.utils import perform_llama_response
from utils.utils import perform_mistral_response

# dispatch is used to implement func overload based on the number of arguments passed to auto_evaluation
@dispatch(argparse.Namespace, object, object, object, object, object, object, object)
def auto_evaluation(args,query,bck_extracted_final_question,true_ans,fwd_extracted_final_response,bck_extracted_final_response,
                    fwd_extracted_final_resp_exp,bck_extracted_final_resp_exp):
    
    fwd_extracted_final_response,fwd_extracted_final_resp_exp,bck_extracted_final_response,bck_extracted_final_resp_exp = \
            map(str,(fwd_extracted_final_response,fwd_extracted_final_resp_exp,bck_extracted_final_response,
                     bck_extracted_final_resp_exp))
    
    # if len(fwd_extracted_final_response.split()) < 5:
    fwd_extracted_final_response = fwd_extracted_final_response + " " + fwd_extracted_final_resp_exp
    # if len(bck_extracted_final_response.split()) < 5:
    bck_extracted_final_response = bck_extracted_final_response + " " + bck_extracted_final_resp_exp
    # bck_extracted_final_response = bck_extracted_final_resp_exp

    prompt_var_list = [query,bck_extracted_final_question]

    if args.QUERY_INTENT_METHOD == "SemSim":
        model = SentenceTransformer('bert-base-nli-mean-tokens')
        embeddings = model.encode(prompt_var_list)
        similarity = 1 - cosine(embeddings[0], embeddings[1])

        if similarity <= args.SIMILARITY_THRESHOLD:
            prompt_var_list = [query,true_ans,bck_extracted_final_response]
        else:
            prompt_var_list = [query,true_ans,fwd_extracted_final_response]

    elif args.QUERY_INTENT_METHOD == "LLM":

        if "mistral" in args.MODEL_API:
            similarity = perform_mistral_response(prompt_var_list,args.MODEL_API,args.TEMPERATURE,args.BACKWARD_REASONING_QUERY_INTENT_PROMPT_PATH)
        elif "llama2" in args.MODEL_API:
            similarity = perform_llama_response(prompt_var_list,args.MODEL_API,args.TEMPERATURE,args.LLAMA_BACKWARD_REASONING_QUERY_INTENT_PROMPT_PATH)
        elif "gpt" in args.MODEL_API:
            similarity = perform_gpt_response(prompt_var_list,args.MODEL_API,args.TEMPERATURE,args.BACKWARD_REASONING_QUERY_INTENT_PROMPT_PATH)

        extracted_binary_similarity = extract_value_from_single_key(similarity, key = "evaluation:")

        if extracted_binary_similarity == "different":
            prompt_var_list = [query,true_ans,bck_extracted_final_response]
        else:
            prompt_var_list = [query,true_ans,fwd_extracted_final_response]
        similarity = extracted_binary_similarity

    accuracy_resp = perform_gpt_response(prompt_var_list,args.EVAL_MODEL,args.TEMPERATURE,args.AUTO_EVALUATION_PROMPT_PATH)
    extracted_accuracy_resp = extract_value_from_single_key(accuracy_resp, key = "evaluation:")
    accuracy = "Correct" if extracted_accuracy_resp == "correct" else "Incorrect"

    print(query,accuracy)
    return similarity, accuracy

@dispatch(argparse.Namespace, object, object, object)
def auto_evaluation(args,query,true_ans,final_resp_text):
    prompt_var_list = [query,true_ans,final_resp_text]
    accuracy_resp = perform_gpt_response(prompt_var_list,args.EVAL_MODEL,args.TEMPERATURE,args.AUTO_EVALUATION_PROMPT_PATH)
    extracted_accuracy_resp = extract_value_from_single_key(accuracy_resp, key = "evaluation:")
    accuracy = "Correct" if extracted_accuracy_resp == "correct" else "Incorrect"
    print(query,accuracy)
    return accuracy