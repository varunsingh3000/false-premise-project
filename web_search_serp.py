# Web Search via Bing Web Search API
# This file has the code to perform the web search and also process it into a format needed for the LLMs later

import json
import os 
import requests
import serpapi


# Function to perform web search using bing web search API, returns the response as json
def perform_web_search(query,engine,language,location_country,subscription_key):

    # Query term(s) to search for.
    # query = "Why does Mars have three moons?"

    # Construct a request
    params = { 'q': query, 'engine': engine, 'hl': language, 'gl': location_country, 'api_key': subscription_key}

    # Call the API
    try:
        response_json = dict(serpapi.search(params))
        # Save the web response results in json for review
        with open('Web_Search_Response/serpapi_search_results.json', 'w', encoding='utf-8') as file:
            json.dump(response_json, file, ensure_ascii=False, indent=4)
            print("Data has been saved to 'serpapi_search_results.json'")
    except Exception as ex:
        print("An error occured at perform_web_search: ", ex)
    
    return response_json

#Function to process the json response into a proper format
def process_json(response_json):
    # Load the external evidence into a new var for data parsing
    data = response_json.copy()
    # We will gather all the relevant info and store it in a dictionary
    # We are only interested in the webpages and entities (knowledgegraphs)
    retrieved_info_dict = {}
        
    if 'answer_box' in data:
        print("Answer Box is found as external evidence")
        retrieved_info_dict.update({"answer_box":[]})
        try:
            name = data["answer_box"]["title"]
            url = data["answer_box"]["link"]
            snippet = data["answer_box"]["snippet"]
            temp_dict = {"name": name, "url": url, "snippet": snippet}
            retrieved_info_dict["answer_box"].append(temp_dict)
        except KeyError as e:
            # If any of the keys are missing, print a message and continue to the next item
            print(f"Key {e} not found. Skipping this item.")

    # Iterating through the webpages section
    if 'organic_results' in data:
        print("Web pages are found as external evidence")
        retrieved_info_dict.update({"organic_results":[]})
        for webpageitem in data["organic_results"][:]: #iterating through a list here
            try:
                name = webpageitem["title"]
                url = webpageitem["link"]
                snippet = webpageitem["snippet"]
                temp_dict = {"name": name, "url": url, "snippet": snippet}
                retrieved_info_dict["organic_results"].append(temp_dict)
            except KeyError as e:
                # If any of the keys are missing, print a message and continue to the next item
                print(f"Key {e} not found. Skipping this item.")
                continue

    if 'related_questions' in data:
        print("Related Questions are found as external evidence")
        retrieved_info_dict.update({"related_questions":[]})
        for relatedquesitem in data["related_questions"][:2]: #iterating through a list here
            try:
                name = relatedquesitem["question"]
                url = relatedquesitem["link"]
                snippet = relatedquesitem["snippet"]
                temp_dict = {"name": name, "url": url, "snippet": snippet}
                retrieved_info_dict["related_questions"].append(temp_dict)
            except KeyError as e:
                # If any of the keys are missing, print a message and continue to the next item
                print(f"Key {e} not found. Skipping this item.")
                continue

    return retrieved_info_dict


def start_web_search(query):
    print("Web search process starts")
    # Add your Bing Search V7 subscription key to your environment variables.
    # subscription_key = os.environ.get("BING_WEB_SEARCH_API_KEY")
    subscription_key = "6ad4eeb47eaa503b840586b572627d31abff00c5e205cf8b1d3809f09d0c38eb"
    # endpoint = "https://api.bing.microsoft.com" + "/v7.0/search" 
    # mkt = 'en-US'
    engine = "google"
    language = "en"
    location_country = "us"
    # call the web search api and get the response as json back
    response_json = perform_web_search(query,engine,language,location_country,subscription_key)
    # process the json response into a suitable format of dictionary later used
    external_evidence = process_json(response_json)
    print("Web search process ends")
    return external_evidence


