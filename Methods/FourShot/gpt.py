import os 
from openai import OpenAI

def perform_gpt_response(client,prompt_var_list,model,temperature,prompt_path):
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


def perform_qa_task(args,client,prompt_var_list):
    final_response = perform_gpt_response(client,prompt_var_list,args.MODEL_API,
                                          args.CANDIDATE_TEMPERATURE,args.QUERY_PROMPT_PATH)
    # print("Final response is: ", final_response)
    return final_response

def start_model_response(args,query,external_evidence):
    print("OpenAI model response process starts",query)
    client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY")) # defaults to os.environ.get("OPENAI_API_KEY")
    prompt_var_list = [query]
    result = perform_qa_task(args,client,prompt_var_list)

    if result is None:
        print("Error: Result is None")
    else:
        final_response = result
        return final_response