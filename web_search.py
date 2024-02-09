# Web Search via Bing Web Search API
# This file has the code to perform the web search and also process it into a format needed for the LLMs later

import json
import os 
import requests

'''
This sample makes a call to the Bing Web Search API with a query and returns relevant web search.
Documentation: https://docs.microsoft.com/en-us/bing/search-apis/bing-web-search/overview
'''

# Function to perform web search using bing web search API, returns the response as json
def perform_web_search(query,mkt,endpoint,subscription_key):

    # Query term(s) to search for.
    # query = "Why does Mars have three moons?"

    # Construct a request
    params = { 'q': query, 'mkt': mkt}
    headers = { 'Ocp-Apim-Subscription-Key': subscription_key }

    # Call the API
    try:
        response = requests.get(endpoint, headers=headers, params=params)
        response.raise_for_status()
        response_json = response.json()

        # Save the web response results in json for review
        # with open('Web_Search_Response/bing_search_results.json', 'w', encoding='utf-8') as file:
        #     json.dump(response_json, file, ensure_ascii=False, indent=4)
        #     print("Data has been saved to 'bing_search_results.json'")
    except Exception as ex:
        print("An error occured at perform_web_search: ", ex)
        evidence_missing_message = "Web search was unsuccessful due to an error with the web search API, LLM use your knowledge."
        response_json = {"message":evidence_missing_message}
    
    return response_json

#Function to process the json response into a proper format
def process_json(response_json):
    # Load the external evidence into a new var for data parsing
    data = response_json.copy()

    # We will gather all the relevant info and store it in a dictionary
    # We are only interested in the webpages and entities (knowledgegraphs)
    retrieved_info_dict = {}

    # Iterating through the webpages section
    # print('webPages' in data and 'value' in data['webPages'])
    if 'webPages' in data and 'value' in data['webPages']:
        print("Web pages are found as external evidence")
        retrieved_info_dict.update({"webPages":[]})
        for webpageitem in data["webPages"]["value"][:2]: #iterating through a list here
            name = webpageitem["name"]
            url = webpageitem["url"]
            snippet = webpageitem["snippet"]
            temp_dict = {"name":name,"url":url,"snippet":snippet}
            retrieved_info_dict["webPages"].append(temp_dict)

    # Iterating through the entities section
    # print('entities' in data and 'value' in data['entities'])
    if 'entities' in data and 'value' in data['entities']:
        print("Entities (knowledge graph) is found as external evidence")
        retrieved_info_dict.update({"entities":[]})
        for entityitem in data["entities"]["value"]: #iterating through a list here
            name = entityitem["name"]
            url = entityitem["image"]["provider"][0]["url"] if "image" in entityitem and "provider" in entityitem["image"] and \
            entityitem["image"]["provider"] and entityitem["image"]["provider"][0] and "url" in entityitem["image"]["provider"][0] \
            else entityitem["webSearchUrl"]
            
            snippet = entityitem["description"]
            temp_dict = {"name":name,"url":url,"snippet":snippet}
            retrieved_info_dict["entities"].append(temp_dict)

    if "message" in data:
        retrieved_info_dict = data.copy()

    return retrieved_info_dict


def start_web_search(query):
    print("Web search process starts")
    # Add your Bing Search V7 subscription key to your environment variables.
    subscription_key = os.environ.get("BING_WEB_SEARCH_API_KEY")
    endpoint = "https://api.bing.microsoft.com" + "/v7.0/search" 
    mkt = 'en-US'
    # call the web search api and get the response as json back
    response_json = perform_web_search(query,mkt,endpoint,subscription_key)
    # process the json response into a suitable format of dictionary later used
    external_evidence = process_json(response_json)
    print("Web search process ends")
    return external_evidence


