import yaml
import os 
from openai import OpenAI

from utils.utils import process_response
from utils.utils import check_dict_keys_condition

from sentence_transformers import SentenceTransformer
from scipy.spatial.distance import cosine


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
def perform_uncertainty_estimation(client,query,external_evidence,WORKFLOW_RUN_COUNT):
    # print("Uncertainty Estimation process starts")
    og_response_dict = "test of baseline"
    final_confidence_value = -1
    responses_dict = {}
    # print(WORKFLOW_RUN_COUNT)
    responses_dict.update({WORKFLOW_RUN_COUNT:[]})
    # original response is added to the zero index of the responses_dict, the rest will be candidate responses
    responses_dict[WORKFLOW_RUN_COUNT].append(og_response_dict)
    # external evidence is added to the first index of the responses_dict, the rest will be candidate responses
    responses_dict[WORKFLOW_RUN_COUNT].append(external_evidence)
    # question is added to the second index of the responses_dict, the rest will be candidate responses
    responses_dict[WORKFLOW_RUN_COUNT].append(query)

    
    prompt_var_list = [query]
    final_response = perform_gpt_response(client,prompt_var_list,CANDIDATE_TEMPERATURE,QUERY_PROMPT_PATH)
    # uncertainty_response = perform_llama_response(client,candi_resp_list,TEMPERATURE,UNCERTAINTY_PROMPT_PATH)
    # print(uncertainty_response)
    # extract_value_from_single_key(uncertainty_response, 'Final Response:')
    # final_response = uncertainty_response
    print("Final response is: ", final_response)
    return responses_dict, final_response, final_confidence_value


def start_openai_api_model_response(query,external_evidence,WORKFLOW_RUN_COUNT):
    print("OpenAI model response process starts: ", query)
    client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY")) # defaults to os.environ.get("OPENAI_API_KEY")
    result = perform_uncertainty_estimation(client,query,external_evidence,WORKFLOW_RUN_COUNT)
    # print("OpenAI model response process ends")
    if result is None:
        print("Error: Result is None")
    else:
        responses_dict, final_response, final_confidence_value = result
        return responses_dict, final_response, final_confidence_value
