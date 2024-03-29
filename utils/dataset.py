import pandas as pd
import yaml


with open('params.yaml', 'r') as file:
    config = yaml.safe_load(file)

DATASET_PATH = config['DATASET_PATH']

def process_freshqa(dataset_name):
    # processing specific to freshqa dataset
    df_og = pd.read_csv(DATASET_PATH+dataset_name+".csv")
    new_header = df_og.iloc[1]  # Grab the second row for the new column names
    df = df_og.copy().loc[2:][:50] #the final indexing can be used to control how many/which questions to test
    df.columns = new_header  # Set the new column names
    query_list = df["question"].tolist()
    ans_list = df["answer_0"].tolist()
    ques_id_list = df["id"].tolist()
    return ques_id_list, query_list, ans_list

def process_QAQA(dataset_name):
    # processing specific to freshqa dataset
    df_og = pd.read_csv(DATASET_PATH+dataset_name+".csv")
    df = df_og.copy()[:10]
    query_list = df["question"].tolist()
    ans_list = df["abstractive_answer"].tolist()
    ques_id_list = df["idx"].tolist()
    return ques_id_list, query_list, ans_list

# function to start the processing of dataset based on which dataset is called
def start_dataset_processing(dataset_name):
    print("Dataset processing started")
    if dataset_name == "freshqa":
        ques_id_list, query_list, ans_list = process_freshqa(dataset_name)
    elif dataset_name == "QAQA":
        ques_id_list, query_list, ans_list = process_QAQA(dataset_name)
    
    print("Dataset processing ended")
    return ques_id_list, query_list, ans_list

