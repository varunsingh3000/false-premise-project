import json
import yaml
import os
from openai import OpenAI

from web_search_serp import start_web_search

with open('params.yaml', 'r') as file:
    config = yaml.safe_load(file)

# Retrieve parameters 
EVIDENCE_BATCH_SAVE_PATH = config['EVIDENCE_BATCH_SAVE_PATH']
EVAL_MODEL = config['EVAL_MODEL']
TEMPERATURE = config['TEMPERATURE']
CANDIDATE_TEMPERATURE = config['CANDIDATE_TEMPERATURE']
AUTO_EVALUATION_PROMPT_PATH = config['AUTO_EVALUATION_PROMPT_PATH']

def generate_evidence_batch(ques_id_list,query_list):
    # evidence list for saving results in batch
    evidence_batch_list = []
    for query_id,query in zip(ques_id_list,query_list):
        external_evidence = start_web_search(query_id,query)
        evidence_batch_list.append(external_evidence)
    
    # Write the list to the JSON file
    with open(EVIDENCE_BATCH_SAVE_PATH, 'w') as json_file:
        json.dump(evidence_batch_list, json_file, indent=4)

def modify_evidence_batch_dict(evidence_batch_list):
    modified_evidence_batch_list = {}
    for key, value in evidence_batch_list.items():
        if isinstance(value, list):
            for item in value:
                item.pop("url", None)

    if 'QueryID' in evidence_batch_list:
        modified_evidence_batch_list['QueryID'] = evidence_batch_list['QueryID']
    if 'answer_box' in evidence_batch_list and evidence_batch_list['answer_box']:
        modified_evidence_batch_list['answer_box'] = evidence_batch_list['answer_box']
        if 'related_questions' in evidence_batch_list:
            modified_evidence_batch_list['related_questions'] = evidence_batch_list['related_questions'][:]
        return modified_evidence_batch_list
    if 'related_questions' in evidence_batch_list:
        modified_evidence_batch_list['related_questions'] = evidence_batch_list['related_questions'][:]
    if 'organic_results' in evidence_batch_list:
        modified_evidence_batch_list['organic_results'] = evidence_batch_list['organic_results'][:]
    return modified_evidence_batch_list


# function to parse through the api response and extract certain keywords in a dict
def process_response(chat_completion):
    #use the chat_completion object to retrieve the textual LLM response
    text = chat_completion[:]
    # Remove all newline characters ("\n")
    text_without_newlines = text.replace('\n', '')

    # Define key terms to split the text into sections
    # key_terms = ['Answer:', 'Confidence Level:', 'Source:', 'Premise of the Question:']
    key_terms = ['Answer:', 'Source:', 'Premise of the Question:']
    response_dict = {}
    # Splitting the text into sections based on key terms
    for i in range(len(key_terms) - 1):
        term = key_terms[i]
        next_term = key_terms[i + 1]
        if term in text_without_newlines and next_term in text_without_newlines:
            split_text = text_without_newlines.split(term, 1)[1].split(next_term, 1)
            response_dict[term.strip()] = split_text[0].strip() if len(split_text) > 1 else ''

    # For the last key term
    last_term = key_terms[-1]
    if last_term in text_without_newlines:
        split_text = text_without_newlines.split(last_term, 1)
        response_dict[last_term.strip()] = split_text[1].strip() if len(split_text) > 1 else ''
    
    if not response_dict: #if in case none of the keys were present simply insert the text as it is
        response_dict['message'] = text

    return response_dict

def matching_condition_check(match_count,MAX_CANDIDATE_RESPONSES,MATCH_CRITERIA):
    if MATCH_CRITERIA == "Half":
        match_check = MAX_CANDIDATE_RESPONSES//2
        if match_count > match_check:
            return True
    if MATCH_CRITERIA == "Full":
        match_check = MAX_CANDIDATE_RESPONSES
        if match_count == match_check:
            return True

def check_dict_keys_condition(response_dict):
    key_terms = ['Answer:', 'Source:', 'Premise of the Question:']
    keys_present = all(key in response_dict for key in key_terms)
    if keys_present:
        return True
    return False

def extract_value_from_single_key(response, key):
    # Find the index of key
    index = response.find(key)

    if index != -1:
        # Extract the text after key
        text = response[index + len(key):].strip()
        
        # Find the next space after the value
        next_space_index = text.find("\n")
        if next_space_index == -1:
            next_space_index = len(text)

        # Extract the value after the key
        if next_space_index != -1:
            extracted_text = text[:next_space_index].strip()
            return extracted_text
        else:
            print("No value found after the {} key.".format(key))
    else:
        print("{} key not found".format(key))

    return response

def create_dummy_response_dict(og_response_dict,external_evidence,query,WORKFLOW_RUN_COUNT,MAX_CANDIDATE_RESPONSES):
    dummy_response_dict = {
        WORKFLOW_RUN_COUNT: [
            og_response_dict,
            external_evidence,
            query,
            *['candidate_response{} empty candidate response'.format(i) for i in range(MAX_CANDIDATE_RESPONSES)]
        ]
    }
    return dummy_response_dict

def auto_evaluation(query,true_ans,final_resp_text):
    prompt_var_list = [query,true_ans,final_resp_text]
    same_ques_resp = perform_gpt_response(prompt_var_list,TEMPERATURE,AUTO_EVALUATION_PROMPT_PATH)
    extracted_gt_ans_resp1 = extract_value_from_single_key(same_ques_resp, key = "evaluation:")
    accuracy = "Correct" if extracted_gt_ans_resp1 == "correct" else "Incorrect"
    print(query,accuracy)
    return accuracy

# this func is provided for easy access to the gpt model api for any use case
# presently this is used for automatic evaluation
def perform_gpt_response(prompt_var_list,temperature,prompt_path):
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
        model=EVAL_MODEL,
        temperature=temperature
        )
    
    # print(chat_completion.choices[0].message)
    # print("The token usage: ", chat_completion.usage)
    return chat_completion.choices[0].message.content.strip()