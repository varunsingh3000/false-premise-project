import pandas as pd

def process_freshqa(dataset_path,dataset_name):
    # processing specific to freshqa dataset
    df_og = pd.read_csv(dataset_path+dataset_name+".csv")
    new_header = df_og.iloc[1]  # Grab the second row for the new column names
    df = df_og.copy().loc[2:][:5] #the final indexing can be used to control how many/which questions to test
    df.columns = new_header  # Set the new column names
    query_list = df["question"].tolist()
    ans_list = df["answer_0"].tolist()
    ques_id_list = df["id"].tolist()
    effective_year_list = df["effective_year"].tolist()
    num_hops_list = df["num_hops"].tolist()
    fact_type_list = df["fact_type"].tolist()
    premise_list = df["false_premise"].tolist()
    return ques_id_list, query_list, ans_list, effective_year_list, num_hops_list, fact_type_list, premise_list

def process_QAQA(dataset_path,dataset_name):
    # processing specific to freshqa dataset
    df_og = pd.read_csv(dataset_path+dataset_name+".csv")
    df = df_og.copy()[:2]
    query_list = df["question"].tolist()
    ans_list = df["abstractive_answer"].tolist()
    ques_id_list = df["idx"].tolist()
    premise_list = df["all_assumptions_valid"].tolist()
    return ques_id_list, query_list, ans_list, premise_list

# function to start the processing of dataset based on which dataset is called
def start_dataset_processing(args):
    print("Dataset processing started")
    if args.DATASET_NAME == "freshqa":
        dataset_elements = process_freshqa(args.DATASET_PATH,args.DATASET_NAME)
    elif args.DATASET_NAME == "QAQA":
        dataset_elements = process_QAQA(args.DATASET_PATH,args.DATASET_NAME)
    
    print("Dataset processing ended")
    return dataset_elements