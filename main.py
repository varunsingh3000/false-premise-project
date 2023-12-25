#This is the main file from where the execution of the workflow starts

import json
import os 
import requests
import time


from web_search import start_web_search
# from openai_gpt_models import start_openai_api_model_response
from openai_gpt_models_save_results import start_openai_api_model_response

def start_workflow(query,model,temperature,query_prompt_path,uncertainty_prompt_path,num_runs):
    # start web search and retrieve external evidence
    external_evidence = start_web_search(query)
    start_openai_api_model_response(query,query_prompt_path,uncertainty_prompt_path,model,temperature,num_runs,external_evidence)


start_time = time.time()

query_list = ["Why does Mars have three moons?","When did Elon Musk buy Microsoft?","Who was the first woman president of the US?"]
# query = "Why does Mars have three moons?"
model = "gpt-3.5-turbo-1106"
temperature = 0.0
query_prompt_path = r"C:\GAMES_SETUP\Thesis\Code\Prompts\GPT3.5\initial_prompt_w_cot_single.txt"
uncertainty_prompt_path = r"C:\GAMES_SETUP\Thesis\Code\Prompts\GPT3.5\uncertainty_estimation_confidence.txt"
num_runs = 3
for query in query_list:
    print("NEW QUERY HAS STARTED"*4)
    start_workflow(query,model,temperature,query_prompt_path,uncertainty_prompt_path,num_runs)
    print("QUERY HAS FINISHED"*4)
end_time = time.time()
print(f"Total time taken: {end_time-start_time} seconds")
############### Testing the entire workflow #################
# import pandas as pd

# df_og = pd.read_csv("Downloads/freshqa.csv")
# new_header = df_og.iloc[1]  # Grab the third row for the new column names
# df = df_og.copy().loc[2:][:100]
# df.columns = new_header  # Set the new column names
# print(df.info())

# column_values_list = df["question"].tolist()











#############################################################

# import argparse

# def start_workflow(query, model, temperature, query_prompt_path, uncertainty_prompt_path, num_runs):
#     external_evidence = start_web_search(query)
#     start_openai_api_model_response(query, query_prompt_path, uncertainty_prompt_path, model, temperature, num_runs, external_evidence)

# def main():
#     parser = argparse.ArgumentParser(description="Run workflow with specified parameters")
    
#     parser.add_argument('--query', type=str, default="Why does Mars have three moons?", help='Query for the workflow')
#     parser.add_argument('--model', type=str, default="gpt-3.5-turbo-1106", help='Model name')
#     parser.add_argument('--temperature', type=float, default=0.0, help='Temperature value')
#     parser.add_argument('--query-prompt-path', type=str, default=r"C:\GAMES_SETUP\Thesis\Code\Prompts\GPT3.5\initial_prompt_w_cot_single.txt", help='Path to query prompt')
#     parser.add_argument('--uncertainty-prompt-path', type=str, default=r"C:\GAMES_SETUP\Thesis\Code\Prompts\GPT3.5\uncertainty_estimation_confidence.txt", help='Path to uncertainty prompt')
#     parser.add_argument('--num-runs', type=int, default=3, help='Number of runs')
    
#     args = parser.parse_args()

#     start_time = time.time()

#     start_workflow(
#         args.query,
#         args.model,
#         args.temperature,
#         args.query_prompt_path,
#         args.uncertainty_prompt_path,
#         args.num_runs
#     )

#     end_time = time.time()
#     print(f"Total time taken: {end_time - start_time} seconds")

# if __name__ == "__main__":
#     main()