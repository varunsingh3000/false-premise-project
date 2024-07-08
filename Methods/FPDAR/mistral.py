import json
import os
import pandas as pd

from mistralai.client import MistralClient
from mistralai.models.chat_completion import ChatMessage

from utils.utils import process_response
from utils.utils import extract_value_from_single_key
from utils.utils import check_dict_keys_condition

def perform_mistral_response(client,prompt_var_list,model,temperature,prompt_path):
    
    with open(prompt_path, 'r') as file:
        file_content = file.read()
    message = [
        ChatMessage(role="user", content=file_content.format(*prompt_var_list))
    ]
    chat_completion = client.chat(
        model=model,
        temperature=temperature,
        messages=message
    )
    
    return chat_completion.choices[0].message.content.strip()


def perform_qa_task(args,client,query,external_evidence,final_response_ans,final_response_exp):
    fwd_main_answers_list = []
    bck_main_answers_list = []

    fwd_main_answers_list.append(final_response_ans)
    fwd_main_answers_list.append(final_response_exp)

    #backward reasoning
    fwd_extracted_final_response = final_response_ans[:]
    fwd_extracted_final_resp_exp = final_response_exp[:]
    if len(fwd_extracted_final_response.split()) < 5:
        fwd_extracted_final_response = fwd_extracted_final_response + " " + fwd_extracted_final_resp_exp

    external_evidence = json.dumps(external_evidence, indent=4)
    prompt_var_list = [external_evidence, fwd_extracted_final_response]
    
    back_reasoning_response_query = perform_mistral_response(client,prompt_var_list,args.MODEL_API,
                            args.CANDIDATE_TEMPERATURE,args.BACKWARD_REASONING_QUERY_PROMPT_PATH)
    back_reasoning_response = perform_mistral_response(client,prompt_var_list,args.MODEL_API,
                            args.CANDIDATE_TEMPERATURE,args.BACKWARD_REASONING_RESP_PROMPT_PATH)
    bck_main_answers_list.append(back_reasoning_response_query)
    bck_main_answers_list.append(back_reasoning_response)

    return fwd_main_answers_list, bck_main_answers_list


def start_model_response(args,query,external_evidence):
    print("Mistral model response process starts ",query)
    client = MistralClient(api_key=os.environ["MISTRAL_API_KEY"])

    if args.USE_INITIAL_RESP_DATA:
        df = pd.read_csv(args.INITIAL_RESP_DATA_PATH + "file name here"+".csv")
        final_response_ans = df["Answer"]
        final_response_exp = df["Explanation"]
    else:
        prompt_var_list = [query, external_evidence]
        chat_completion = perform_mistral_response(client,prompt_var_list,args.MODEL_API,
                                            args.TEMPERATURE,args.QUERY_PROMPT_PATH)
        og_response_dict = process_response(chat_completion)
        # print(og_response_dict)
        if not check_dict_keys_condition(og_response_dict):
            og_response_dict['Answer:'] = next(iter(og_response_dict.items()))[1]

        final_response_ans = og_response_dict['Answer:']
        og_response_dict['Explanation:'] = ""
        final_response_exp = og_response_dict['Explanation:']

    fwd_main_answers_list, bck_main_answers_list = perform_qa_task(args,client,query,external_evidence,
                                       final_response_ans,final_response_exp)
    return og_response_dict, fwd_main_answers_list, bck_main_answers_list
