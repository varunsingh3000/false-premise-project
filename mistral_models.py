import yaml
import os
from mistralai.client import MistralClient
from mistralai.models.chat_completion import ChatMessage

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


def perform_mistral_response(client,prompt_var_list,temperature,prompt_path):
    # variable1 and variable2 in general are the first and second input passed to the prompt
    # this can be query and external evidence or original response and candidate response
    #Read the prompts from txt files
    with open(prompt_path, 'r') as file:
        file_content = file.read()
    
    message = [
        ChatMessage(role="user", content=file_content.format(*prompt_var_list))
    ]

    chat_completion = client.chat(
        model=MODEL,
        temperature=temperature,
        messages=message
        #max_tokens=600
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
    og_response_dict = "test of new sc"
    final_confidence_value = -1
    candi_resp_list = []
    responses_dict = {}
    # print(WORKFLOW_RUN_COUNT)
    responses_dict.update({WORKFLOW_RUN_COUNT:[]})
    # original response is added to the zero index of the responses_dict, the rest will be candidate responses
    responses_dict[WORKFLOW_RUN_COUNT].append(og_response_dict)
    # external evidence is added to the first index of the responses_dict, the rest will be candidate responses
    responses_dict[WORKFLOW_RUN_COUNT].append(external_evidence)
    # question is added to the second index of the responses_dict, the rest will be candidate responses
    responses_dict[WORKFLOW_RUN_COUNT].append(query)

    for i in range(MAX_CANDIDATE_RESPONSES):
        prompt_var_list = [query, external_evidence]
        # candidate responses are generated
        chat_completion_resp_obj = perform_mistral_response(client,prompt_var_list,CANDIDATE_TEMPERATURE,QUERY_PROMPT_PATH)
        response_dict = process_response(chat_completion_resp_obj)
        print(response_dict)
        if not check_dict_keys_condition(response_dict):
            message = "It seems a proper response could not be generated."
            print(i,response_dict)
            responses_dict[WORKFLOW_RUN_COUNT].append(response_dict)
            candi_resp_list.append(message)
            continue
        candi_resp_list.append(response_dict['Answer:'])
        responses_dict[WORKFLOW_RUN_COUNT].append(response_dict)

    model = SentenceTransformer('bert-base-nli-mean-tokens')
    embeddings = model.encode(candi_resp_list)
    similarities = [
    1 - cosine(embeddings[0], embeddings[1]),
    1 - cosine(embeddings[0], embeddings[2]),
    1 - cosine(embeddings[1], embeddings[2])
    ]
    max_index = similarities.index(max(similarities))

    final_response = candi_resp_list[max_index]
    # uncertainty_response = perform_llama_response(client,candi_resp_list,TEMPERATURE,UNCERTAINTY_PROMPT_PATH)
    # print(uncertainty_response)
    # extract_value_from_single_key(uncertainty_response, 'Final Response:')
    # final_response = uncertainty_response
    print("Final response is: ", final_response)
    return responses_dict, final_response, final_confidence_value


def start_mistral_api_model_response(query,external_evidence,WORKFLOW_RUN_COUNT):
    print("Mistral model response process starts ", query)
    client = MistralClient(api_key=os.environ["MISTRAL_API_KEY"])
    result = perform_uncertainty_estimation(client,query,external_evidence,WORKFLOW_RUN_COUNT)
    # print("Mistral model response process ends")
    if result is None:
        print("Error: Result is None")
    else:
        responses_dict, final_response, final_confidence_value = result
        return responses_dict, final_response, final_confidence_value
