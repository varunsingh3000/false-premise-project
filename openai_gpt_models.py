import yaml
import os 
import requests
from openai import OpenAI
from rake_nltk import Rake

from utils.utils import process_response
from utils.utils import uncertainty_confidence_cal
from utils.utils import matching_condition_check
from utils.utils import check_dict_keys_condition
from utils.utils import create_dummy_response_dict
from web_search import start_web_search
# from web_search_serp import start_web_search


with open('params.yaml', 'r') as file:
    config = yaml.safe_load(file)

MODEL = config['MODEL']
TEMPERATURE = config['TEMPERATURE']
CANDIDATE_TEMPERATURE = config['CANDIDATE_TEMPERATURE']
QUERY_PROMPT_PATH = config['QUERY_PROMPT_PATH']
UNCERTAINTY_PROMPT_PATH = config['UNCERTAINTY_PROMPT_PATH']
MATCH_CRITERIA = config['MATCH_CRITERIA'] 
MAX_CANDIDATE_RESPONSES = config['MAX_CANDIDATE_RESPONSES'] 

# call the openai api with either gpt 3.5 or gpt 4 latest model
def perform_gpt_response(client,prompt_var_list,temperature,prompt_path):
    # all the variables needed for the prompt are in the prompt_var_list
    # this can be query and external evidence or original response and candidate response
    #Read the prompts from txt files
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
        model=MODEL,
        temperature=temperature,
        max_tokens = 600
        )
    # print("#"*20)
    # print("INITIAL LLM RESPONSE")
    # print(chat_completion.choices[0].message)
    # print("The token usage: ", chat_completion.usage)
    return chat_completion.choices[0].message.content.strip()

# function to call the LLM to generate multiple responses which will then be compared with the original response
# using their core concepts
def perform_uncertainty_estimation(og_response_dict,client,query,external_evidence,WORKFLOW_RUN_COUNT):
    print("Uncertainty Estimation process starts")
    if check_dict_keys_condition(og_response_dict):
        responses_dict = {}
        # print(WORKFLOW_RUN_COUNT)
        responses_dict.update({WORKFLOW_RUN_COUNT:[]})
        # original response is added to the zero index of the responses_dict, the rest will be candidate responses
        responses_dict[WORKFLOW_RUN_COUNT].append(og_response_dict)
        # external evidence is added to the first index of the responses_dict, the rest will be candidate responses
        responses_dict[WORKFLOW_RUN_COUNT].append(external_evidence)
        # question is added to the second index of the responses_dict, the rest will be candidate responses
        responses_dict[WORKFLOW_RUN_COUNT].append(query)
        intial_explanation = og_response_dict['Explanation:']
        # initial_core_concept = og_response_dict['Core Concept:']

        match_count = 0
        confi_list = []
        confi_match_list = []
        max_confi_value = 0
        final_confidence_value = -1
        final_response = og_response_dict.copy()

        for i in range(MAX_CANDIDATE_RESPONSES):
            prompt_var_list = [query, external_evidence]
            # candidate responses are generated
            chat_completion_resp_obj = perform_gpt_response(client,prompt_var_list,CANDIDATE_TEMPERATURE,QUERY_PROMPT_PATH)
            response_dict = process_response(chat_completion_resp_obj)
            # if all keys are not present in the candidate response dict then skip the current iteration
            # the code logic ahead will not work without all keys and there would be too many conditions to make things work
            if not check_dict_keys_condition(response_dict):
                message = "It seems the candidate response {} was missing some keys in the response dict {} so the current \
                      iteration of the candidate response generation has been skipped. The next iteration \
                      will continue.".format(i,response_dict)
                responses_dict[WORKFLOW_RUN_COUNT].append(message)
                print(message)
                continue
            # print("Candidate response {}: {}".format(i,response_dict))
            responses_dict[WORKFLOW_RUN_COUNT].append(response_dict)
            # concepts are passed instead of query and external evidence since the function basically just needs to call the api
            prompt_var_list = [intial_explanation, response_dict['Answer:'] + response_dict['Explanation:']]
            uncertainty_response = perform_gpt_response(client,prompt_var_list,TEMPERATURE,UNCERTAINTY_PROMPT_PATH)
            # print("Uncertainty estimation response {}: {}".format(i,uncertainty_response))
            response_dict.update({"Certainty_Estimation":uncertainty_response})
            confi_value = int(response_dict['Confidence Level:'][:-1]) if response_dict['Confidence Level:'][:-1].isdigit() else 0
            confi_list.append(confi_value)
            # checking if the candidate response agrees with the original response
            if uncertainty_response.startswith("Yes") or uncertainty_response.upper() == "YES":
                #Max confidence value itself is not used but this condition is used to identify the response with the highest confidence
                #and that response will be chosen as the potential final response
                if confi_value >= max_confi_value: 
                    max_confi_value = confi_value
                    potential_final_response = response_dict.copy()
                confi_match_list.append(confi_value)
                match_count += 1
            elif uncertainty_response.startswith("No") or uncertainty_response.upper() == "NO":
                confi_value = 0
                confi_match_list.append(confi_value)
            # checking whether sufficent candidate responses agree with the original response
            if matching_condition_check(match_count,MAX_CANDIDATE_RESPONSES,MATCH_CRITERIA): 
                final_confidence_value = uncertainty_confidence_cal(confi_match_list,confi_list)
                potential_final_response['Confidence Level:'] = f"{final_confidence_value}%"
                final_response = potential_final_response.copy()
                return responses_dict, final_response, final_confidence_value
    
        # if we are here then that means the matching condition was unsuccessful, this means that the final response
        # will be the one with the highest verbalise confidence
        
        # Flatten the list of dictionaries
        all_candi_resp = [entry for sublist in responses_dict.values() for entry in sublist]
        # Filter out non-dictionary elements from the list, this is done to ensure get method below doesn't throw an error
        all_candi_resp = [entry for entry in all_candi_resp if isinstance(entry, dict)]
        # Find the dictionary entry with the highest confidence
        final_response = max(all_candi_resp, key=lambda x: x.get('Confidence Level:', '-1'))
        final_confidence_value = final_response['Confidence Level:']

    else:
        print("It seems all the keys in the original response were not available so candidate response generation \
              for the self-consistency approach cannot happen. In this case the original response is considered \
              as it is and the next question is answered. {}".format(og_response_dict))
        responses_dict = create_dummy_response_dict(og_response_dict,external_evidence,query,
                                                    WORKFLOW_RUN_COUNT, MAX_CANDIDATE_RESPONSES)
        final_confidence_value = '-1'
        final_response = og_response_dict.copy()
    # Now the adversarial attack part will start
    print("The first run of the workflow has finished. Now the adversarial attacks will start.")
    return responses_dict, final_response, final_confidence_value



def start_openai_api_model_response(query,external_evidence,WORKFLOW_RUN_COUNT):
    print("OpenAI model response process starts")
    client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY_NEW")) # defaults to os.environ.get("OPENAI_API_KEY")
    prompt_var_list = [query, external_evidence]
    chat_completion = perform_gpt_response(client,prompt_var_list,TEMPERATURE,QUERY_PROMPT_PATH)
    # extract the key terms from the generated response into a dict
    # this is needed later for uncertainty estimation calculation
    og_response_dict = process_response(chat_completion)
    # print(WORKFLOW_RUN_COUNT)
    result = perform_uncertainty_estimation(og_response_dict,client,query,external_evidence,WORKFLOW_RUN_COUNT)
    print("OpenAI model response process ends")
    if result is None:
        print("Error: Result is None")
    else:
        responses_dict, final_response, final_confidence_value = result
        return responses_dict, final_response, final_confidence_value
