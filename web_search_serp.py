# Web Search via Bing Web Search API
# This file has the code to perform the web search and also process it into a format needed for the LLMs later

import json
import os 
import serpapi


# Function to perform web search using bing web search API, returns the response as json
def perform_web_search(params):

    # Query term(s) to search for.
    # query = "Why does Mars have three moons?"
    # Call the API
    try:
        response_json = dict(serpapi.search(params))
        # Save the web response results in json for review
        # with open('Web_Search_Response/serpapi_search_results.json', 'w', encoding='utf-8') as file:
        #     json.dump(response_json, file, ensure_ascii=False, indent=4)
        #     print("Data has been saved to 'serpapi_search_results.json'")
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

    if 'related_questions' in data:
        print("Related Questions are found as external evidence")
        retrieved_info_dict.update({"related_questions":[]})
        for relatedquesitem in data["related_questions"][:]: #iterating through a list here
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

    # IMP: Organic results are not considered since they act as noise in the prompts for the evidence 
    # organic results are only added if answer_box and related_question are not available
    if 'answer_box' not in data and 'related_questions' not in data:
        if 'organic_results' in data:
            print("Web pages are found as external evidence")
            retrieved_info_dict.update({"organic_results":[]})
            for webpageitem in data["organic_results"][:1]: #iterating through a list here
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
        
    if "message" in data:
        retrieved_info_dict = data.copy()
    
    return retrieved_info_dict


def start_web_search(query):
    print("Web search process starts")
    # Add your SerpAPI Web Search API key to your environment variables.
    subscription_key = os.environ.get("SERP_API_WEB_SEARCH")
    engine = "google"
    language = "en"
    location_country = "us"
    params = { 'q': query, 'engine': engine, 'hl': language, 'gl': location_country, 'api_key': subscription_key}
    # call the web search api and get the response as json back
    response_json = perform_web_search(params)
    # process the json response into a suitable format of dictionary later used
    external_evidence = process_json(response_json)
    print("Web search process ends")
    return external_evidence


