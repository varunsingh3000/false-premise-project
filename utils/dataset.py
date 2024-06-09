import pandas as pd
import yaml


with open('params.yaml', 'r') as file:
    config = yaml.safe_load(file)

DATASET_PATH = config['DATASET_PATH']

def process_freshqa(dataset_name):
    # processing specific to freshqa dataset
    df_og = pd.read_csv(DATASET_PATH+dataset_name+".csv")
    new_header = df_og.iloc[1]  # Grab the second row for the new column names
    df = df_og.copy().loc[2:][:500] #the final indexing can be used to control how many/which questions to test
    df.columns = new_header  # Set the new column names
    query_list = df["question"].tolist()
    ans_list = df["answer_0"].tolist()
    ques_id_list = df["id"].tolist()
    effective_year_list = df["effective_year"].tolist()
    num_hops_list = df["num_hops"].tolist()
    fact_type_list = df["fact_type"].tolist()
    return ques_id_list, query_list, ans_list, effective_year_list, num_hops_list, fact_type_list

def process_QAQA(dataset_name):
    # processing specific to freshqa dataset
    df_og = pd.read_csv(DATASET_PATH+dataset_name+".csv")
    df = df_og.copy()[:5]
    query_list = df["question"].tolist()
    ans_list = df["abstractive_answer"].tolist()
    ques_id_list = df["idx"].tolist()
    premise_list = df["all_assumptions_valid"].tolist()
    return ques_id_list, query_list, ans_list, premise_list

# function to start the processing of dataset based on which dataset is called
def start_dataset_processing(dataset_name):
    print("Dataset processing started")
    if dataset_name == "freshqa":
        dataset_elements = process_freshqa(dataset_name)
    elif dataset_name == "QAQA":
        dataset_elements = process_QAQA(dataset_name)
    
    print("Dataset processing ended")
    return dataset_elements