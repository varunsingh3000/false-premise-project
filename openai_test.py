########### Web Search API

import json
import os 
from pprint import pprint
import requests

'''
This sample makes a call to the Bing Web Search API with a query and returns relevant web search.
Documentation: https://docs.microsoft.com/en-us/bing/search-apis/bing-web-search/overview
'''

# Add your Bing Search V7 subscription key and endpoint to your environment variables.
subscription_key = os.environ.get("BING_WEB_SEARCH_API_KEY")
endpoint = "https://api.bing.microsoft.com" + "/v7.0/search"

# Query term(s) to search for.
query = "Why does Mars have three moons?"

# Construct a request
mkt = 'en-US'
params = { 'q': query, 'mkt': mkt}
headers = { 'Ocp-Apim-Subscription-Key': subscription_key }

# Call the API
try:
    response = requests.get(endpoint, headers=headers, params=params)
    response.raise_for_status()
    response_json = response.json()

    with open('Web_Search_Response/bing_search_results2.json', 'w', encoding='utf-8') as file:
        json.dump(response_json, file, ensure_ascii=False, indent=4)
        print("Data has been saved to 'bing_search_results2.json'")
    # print("\nJSON Response:\n")
    # pprint(response.json())
except Exception as ex:
    raise ex

##### Reading the json file for now as a substitute for calling the api

# Open the JSON file and read its contents
# with open('bing_search_results2.json', 'r', encoding='utf-8') as file:
#     data = json.load(file)

# Load the external evidence into a new var for data parsing
data = response_json.copy()

# We will gather all the relevant info and store it in a dictionary

retrieved_info_dict = {}

# Iterating through the webpages section
print('webPages' in data and 'value' in data['webPages'])
if 'webPages' in data and 'value' in data['webPages']:
    retrieved_info_dict.update({"webPages":[]})
    for webpageitem in data["webPages"]["value"]: #iterating through a list here
        name = webpageitem["name"]
        url = webpageitem["url"]
        snippet = webpageitem["snippet"]
        temp_dict = {"name":name,"url":url,"snippet":snippet}
        retrieved_info_dict["webPages"].append(temp_dict)

# Iterating through the entities section
print('entities' in data and 'value' in data['entities'])
if 'entities' in data and 'value' in data['entities']:
    retrieved_info_dict.update({"entities":[]})
    for entityitem in data["entities"]["value"]: #iterating through a list here
        name = entityitem["name"]
        url = entityitem["image"]["provider"][0]["url"]
        snippet = entityitem["description"]
        temp_dict = {"name":name,"url":url,"snippet":snippet}
        retrieved_info_dict["entities"].append(temp_dict)


# print(retrieved_info_dict)

from openai import OpenAI

client = OpenAI() # defaults to os.environ.get("OPENAI_API_KEY")

#Read the prompts from txt files
# with open(r'C:\GAMES_SETUP\Thesis\Code\Prompts\GPT3.5\initial_prompt_w_fewshot.txt', 'r') as file:
with open(r'C:\GAMES_SETUP\Thesis\Code\Prompts\GPT3.5\initial_prompt_w_cot.txt', 'r') as file:
    # Read the entire content of the file
    file_content = file.read()

# Printing the content of the file (optional)
# print(file_content)

message = [
    {
        "role": "system",
        "content": file_content.format(query,retrieved_info_dict)
    }
    # {   "role": "user",
    #       "content": f"When was {test_var} discovered?"
    #     },
    # {
    #   "role":"user",
    #   "content":f"Also give me the url of the source for this claim"},
    # {
    #   "role":"user",
    #   "content":f"Before answering please check if the question contains a valid premise."}
]

chat_completion = client.chat.completions.create(
    messages=message,
    model="gpt-3.5-turbo",
    temperature=0.0
    )

print("#"*20)
print("INITIAL LLM RESPONSE")
print(chat_completion.choices[0].message)
print("The token usage: ", chat_completion.usage)

### Clarification question starts

# Finding the index where 'Core Concept:' starts
text = chat_completion.choices[0].message.content
core_concept_start = text.find('Core Concept:')

# Finding the index where the core concept ends (assuming it ends before the next section starts)
next_section_start = text.find('Premise of the question:')
core_concept_end = next_section_start if next_section_start != -1 else len(text)

# Extracting the core concept
core_concept = text[core_concept_start:core_concept_end].strip()
# core_concept = "Moons of Mars"

print("#"*20)

confidence_lvl = 100
confidence_thr = 80

if confidence_lvl <= confidence_thr:
    with open(r'C:\GAMES_SETUP\Thesis\Code\Prompts\GPT3.5\cq_modified_prompt.txt', 'r') as file:
    # with open(r'C:\GAMES_SETUP\Thesis\Code\Prompts\GPT3.5\uncertainty_estimation_prompt_wo_cot.txt', 'r') as file:
        # Read the entire content of the file
        modified_file_content = file.read()
    
    # Printing the content of the file (optional)
    # print(modified_file_content)
    
    # Asking for user input with a prompt
    # user_input = input("Please enter your name: ")
    cq_user = f"What would you like to know about '{core_concept}'"
    cq_user_ans = "I would like to know about Mars and it's moons"

    message = [
        {
            "role": "system",
            "content": modified_file_content.format(cq_user_ans,retrieved_info_dict)
        }
    ]
    
    mod_chat_completion = client.chat.completions.create(
        messages=message,
        model="gpt-3.5-turbo",
        temperature=0.0
        )
    
    print("#"*20)
    print("MODIFIED LLM RESPONSE")
    print(mod_chat_completion.choices[0].message)
    print("The token usage: ", mod_chat_completion.usage)
