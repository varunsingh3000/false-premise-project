#!/bin/bash

# Examples of different arguments being provided

# arguments to pass to change method, model and/or dataset
# ./run_main.sh --METHOD FPDAR --MODEL_API gpt-3.5-turbo-1106 --DATASET_NAME freshqa

# arguments to generate a new batch of evidence. Default value is 0 i.e. do not generate new batch everytime
# ./run_main.sh --METHOD FPDAR --MODEL_API gpt-3.5-turbo-1106 --DATASET_NAME freshqa --EVIDENCE_BATCH_GENERATE 1 

# arguments to generate a use evidence batch assuming it has been generated and saved using above command. --EVIDENCE_BATCH_USE by default is set to 1
# ./run_main.sh --METHOD FPDAR --MODEL_API gpt-3.5-turbo-1106 --DATASET_NAME freshqa --EVIDENCE_BATCH_USE 1 

# arguments to generate evidence iteratively for each question during the inference. --EVIDENCE_BATCH_USE by default is set to 1
# ./run_main.sh --METHOD FPDAR --MODEL_API gpt-3.5-turbo-1106 --DATASET_NAME freshqa --EVIDENCE_BATCH_GENERATE 0 --EVIDENCE_BATCH_USE 0

# Execute the Python script with the provided arguments
python3 main.py "$@"
