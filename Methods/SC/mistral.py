# Calling the OpenAI GPT models 3.5 and 4

import numpy as np
import os
from mistralai.client import MistralClient
from mistralai.models.chat_completion import ChatMessage

from utils.utils import process_response
from utils.utils import check_dict_keys_condition

from sentence_transformers import SentenceTransformer
from scipy.spatial.distance import cosine

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

def perform_qa_task(args,client,query,external_evidence):
    # Variables WORKFLOW_RUN_COUNT and og_response_dict are remenants of an old design. They are not used.
    og_response_dict = "test of new sc"
    WORKFLOW_RUN_COUNT = 0

    candi_resp_list = []
    responses_dict = {}

    responses_dict.update({WORKFLOW_RUN_COUNT:[]})
    # original response is added to the zero index of the responses_dict, the rest will be candidate responses
    responses_dict[WORKFLOW_RUN_COUNT].append(og_response_dict)
    # external evidence is added to the first index of the responses_dict, the rest will be candidate responses
    responses_dict[WORKFLOW_RUN_COUNT].append(external_evidence)
    # question is added to the second index of the responses_dict, the rest will be candidate responses
    responses_dict[WORKFLOW_RUN_COUNT].append(query)

    for i in range(args.MAX_CANDIDATE_RESPONSES):
        prompt_var_list = [query, external_evidence]
        # candidate responses are generated
        chat_completion_resp_obj = perform_mistral_response(client,prompt_var_list,args.MODEL_API,
                                        args.CANDIDATE_TEMPERATURE,args.QUERY_PROMPT_PATH)
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

    # Calculate the cosine similarities between all pairs of responses
    num_responses = len(candi_resp_list)
    similarities = np.zeros((num_responses, num_responses))

    for i in range(num_responses):
        for j in range(i + 1, num_responses):
            similarity = 1 - cosine(embeddings[i], embeddings[j])
            similarities[i, j] = similarity
            similarities[j, i] = similarity

    # Calculate the average similarity for each response
    average_similarities = similarities.mean(axis=1)

    # Find the index of the response with the highest average similarity
    max_index = np.argmax(average_similarities)

    # Select the final response based on the highest average similarity
    final_response = candi_resp_list[max_index]
    
    # print("Final response is: ", final_response)
    return responses_dict, final_response


def start_model_response(args,query,external_evidence):
    print("Mistral model response process starts ",query)
    client = MistralClient(api_key=os.environ["MISTRAL_API_KEY"])
    result = perform_qa_task(args,client,query,external_evidence)
    if result is None:
        print("Error: Result is None", result)
    else:
        responses_dict, final_response = result
        return responses_dict, final_response
