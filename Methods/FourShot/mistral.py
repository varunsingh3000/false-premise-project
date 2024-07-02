import os
from mistralai.client import MistralClient
from mistralai.models.chat_completion import ChatMessage

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


def perform_qa_task(args,client,prompt_var_list):
    final_response = perform_mistral_response(client,prompt_var_list,args.MODEL_API,
                                          args.CANDIDATE_TEMPERATURE,args.QUERY_PROMPT_PATH)
    # print("Final response is: ", final_response)
    return final_response

def start_model_response(args,query,external_evidence):
    print("Mistral model response process starts ",query)
    client = MistralClient(api_key=os.environ["MISTRAL_API_KEY"])
    prompt_var_list = [query]
    result = perform_qa_task(args,client,prompt_var_list)
    
    if result is None:
        print("Error: Result is None")
    else:
        final_response = result
        return final_response
