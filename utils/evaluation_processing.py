import os
from multipledispatch import dispatch
import argparse

from utils.utils import extract_value_from_single_key

from sentence_transformers import SentenceTransformer
from scipy.spatial.distance import cosine
from openai import OpenAI

def perform_gpt_response(prompt_var_list,model,temperature,prompt_path):
    client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

    with open(prompt_path, 'r') as file:
        file_content = file.read()

    message = [
        {
            "role": "system",
            "content": file_content.format(*prompt_var_list) 
        }
    ]
    chat_completion = client.chat.completions.create(
        messages=message,
        model=model,
        temperature=temperature
        )
    
    return chat_completion.choices[0].message.content.strip()


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

    model = SentenceTransformer('bert-base-nli-mean-tokens')
    embeddings = model.encode(prompt_var_list)
    similarity = 1 - cosine(embeddings[0], embeddings[1])

    if similarity <= args.SIMILARITY_THRESHOLD:
        prompt_var_list = [query,true_ans,bck_extracted_final_response]
        # extracted_gt_ans_resp1 = "different"
    else:
        prompt_var_list = [query,true_ans,fwd_extracted_final_response]
        # extracted_gt_ans_resp1 = "identical"

    accuracy_resp = perform_gpt_response(prompt_var_list,args.EVAL_MODEL,args.TEMPERATURE,args.AUTO_EVALUATION_PROMPT_PATH)
    # print("Accuracy response text: ", accuracy_resp)
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