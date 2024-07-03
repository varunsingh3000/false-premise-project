# FPDAR Repository

This repository will explain the process to run the inference, the evaluation of the LLM responses and how new methods, LLMs and/or QA datasets could be incorporated into the workflow.

Some implementation details to note:

1. This codebase was implemented in `Python 3.9.15`.
2. Since FPDAR works at an inference level and no inherent changes to the model were being made, all LLMs were accessed using API services. GPT 3.5 turbo and 4 were accessed using
   [OpenAI](https://platform.openai.com/docs/models), Llama2 70B was accessed using [AWS Bedrock service](https://aws.amazon.com/bedrock/), and Mistral Small was accessed using [Mistral API](https://docs.mistral.ai/getting-started/models/).
3. For AWS Bedrock, an AWS account would be needed and the aws cli would have to be setup to programmatically access the LLama2 model via Bedrock API. For [CLI Setup refer](https://docs.aws.amazon.com/cli/latest/userguide/getting-started-install.html) and [configuration setup refer](https://docs.aws.amazon.com/cli/latest/userguide/getting-started-quickstart.html). Alternately since Llama2 is open source it could be hosted locally as well or any other less cumbersome API service could be used to access the LLM.
4. The specific api versions used are as follows `gpt-3.5-turbo-1106`, `gpt-4-turbo-preview`, `meta.llama2-70b-chat-v1` and `mistral-small-latest`. The API keys for each LLM was added as an Environment Variable at the OS level.
5. RAG was implemented using [SerpAPI](https://serpapi.com/search-api). The API Access Key was added as an Environment Variable at the OS level.
6. Similarity threshold was implemented using BERT [model](https://huggingface.co/sentence-transformers/bert-base-nli-mean-tokens) from hugging face.

## Parameters

`METHOD`: FPDAR, Self-Consistency, FourShot method to use. This will decide which prompts are used going further.

`MODEL_API`: LLM API to use. The model api is used to assign the model name and dynamically import the relevant `METHOD` function.

`EVAL_MODEL`: Evaluation LLM to use

`DATASET_NAME`: Dataset name to use

`DATASET_PATH`: Path to the dataset. 

`PROMPT_PATH`: Path to the prompts

`SIMILARITY_THRESHOLD`: Similarity threshold value for FPDAR FP Detection. To generate the results for the Baseline method, the threshold can be set to `0.0`

`TEMPERATURE`: Temperature setting for LLM

`CANDIDATE_TEMPERATURE`: Candidate temperature setting for LLM

`MAX_CANDIDATE_RESPONSES`: Maximum candidate responses to be generated for SC method

`EVIDENCE_ALLOWED`: Boolean to decide to whether evidence should be used. This is used in FourShot method to stop the use of evidence.

`EVIDENCE_BATCH_GENERATE`: Boolean to decide whether evidence is generated in batch or individually for each question. `0` means by default batch will not be generated.

`EVIDENCE_BATCH_USE`: Boolean to decide whether batch evidence has to be used. One means by default batch will be used. If this is set to `0` and `EVIDENCE_ALLOWED` is set to `1`, then evidence is generated iteratively for each question.

`EVIDENCE_BATCH_SAVE_PATH`: Path to save evidence batch results

`QUERY_PROMPT_PATH`: Path to the query prompt file for generating initial response Y

`BACKWARD_REASONING_RESP_PROMPT_PATH`: Path to the abductive reasoning response prompt file for generating explanation Z

`BACKWARD_REASONING_QUERY_PROMPT_PATH`: Path to the abductive reasoning query prompt file for generating question Q`

`LLAMA_QUERY_PROMPT_PATH`: Path to the Llama query prompt file for generating initial response Y

`LLAMA_BACKWARD_REASONING_RESP_PROMPT_PATH`: Path to the Llama abductive reasoning response prompt file for generating explanation Z

`LLAMA_BACKWARD_REASONING_QUERY_PROMPT_PATH`: Path to the Llama abductive reasoning query prompt file for generating question Q`

`AUTO_EVALUATION_PROMPT_PATH`: Path to the auto evaluation prompt file

`RESULT_SAVE_PATH`: Path to save results


## Inference and Evaluation

Since default parameters are set for the parser arguments in `main.py`, cloning the repository, installing the relevant packages and ensuring the LLMs are callable should ensure that the execution happens without any modifications. The specific subset of questions to be used for each dataset can be changed in the `process_{DATASET_NAME}` function in `utils/dataset.py`.

1. Execute requirements.txt to install the relevant Python packages.
2. Set the necessary parameters and run `main.py` which will invoke the starter functions for `inference.py` and `evaluation.py`.
3. `inference.py` will generate the LLM response for the selected dataset and method. This will be saved in `Result` directory as a CSV file.
4. `evaluation.py` will perform the evaluation on the previously generated responses CSV file.
5. The evaluation CSV can then be used in the jupyter notebooks present in the `Analysis` directory for generating benchmarks results.

## Addition of new methods, LLM, and/or datasets

Implementing new methods, LLMs, and/or datasets would require certain changes to be made to the codebase which will be detailed below as follows:

### Method


### LLM
1. `inference.py` has function `start_fp_detc.py`, which is used to call the function responsible for starting the response generation for each `METHOD` and `MODEL_API`. This starter function is `start_model_response` for all method and LLM variations. Now, the `MODEL_API` is used to extract the model name, which is then used to dynamically import the required `start_model_response` function from the required `METHOD` and LLM.
2. This was done to avoid a large number of explicit function imports, which could possibly get larger if more `METHODS` or LLMs were added.
3. Note that for Llama, different versions of the prompts were created with special tokens. In the current implementation, the Llama prompts have different prompts; if another LLM would similarly need special tokens, then a different version of the prompt would have to be created and added as an argument to the parser. Alternatively, a function could be implemented to add the required token where required in the prompt.

### Dataset
1. The new dataset would need to be added to the `Data` directory as a CSV file. Ensure the CSV file name is the same you would input for the `DATASET_NAME` parameter. This is needed for easily accessing the dataset.
2. A new processing function would need to be created in `utils/dataset.py`, which would need to extract the relevant fields as Python lists. The two required lists would be for the question id and the questions themselves.
3. The new dataset would likely possess other fields possibly containing information such as the question's premise, no of hops, etc. These are passed to the `start_response_processing` function from `response_processing.py`. The intent is to create a dictionary with the other information that can then be later used to create a pandas dataframe. This dataframe is then converted to the LLM response CSV file.
