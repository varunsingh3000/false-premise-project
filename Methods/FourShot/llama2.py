import boto3
import json

def perform_llama_response(client,prompt_var_list,model,temperature,prompt_path):
    
    with open(prompt_path, 'r') as file:
        file_content = file.read()
    
    message = json.dumps({
    "prompt": file_content.format(*prompt_var_list), 
    "temperature": temperature
    })

    accept = 'application/json'
    contentType = 'application/json'

    response = client.invoke_model(body=message, modelId=model, accept=accept, contentType=contentType)

    chat_completion = json.loads(response.get("body").read())

    return chat_completion.get("generation").strip()


def perform_qa_task(args,client,prompt_var_list):
    final_response = perform_llama_response(client,prompt_var_list,args.MODEL_API,
                                          args.CANDIDATE_TEMPERATURE,args.LLAMA_QUERY_PROMPT_PATH)
    # print("Final response is: ", final_response)
    return final_response

def start_model_response(args,query,external_evidence):
    print("Meta Llama model response process starts",query)
    client = boto3.client(service_name='bedrock-runtime', region_name='us-east-1')
    prompt_var_list = [query]
    result = perform_qa_task(args,client,prompt_var_list)

    if result is None:
        print("Error: Result is None")
    else:
        final_response = result
        return final_response