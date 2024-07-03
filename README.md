# FPDAR Repository

This repository will explain the process to run the inference, the evaluation of the LLM responses and how new methods, LLMs and/or QA datasets could be incorporated into the workflow.

Some implementation details to note:

1. This codebase was implemented in `Python 3.9.15`.
2. Since FPDAR works at an inference level and no inherent changes to the model were being made, all LLMs were accessed using API services. GPT 3.5 turbo and 4 were accessed using
   [OpenAI](https://platform.openai.com/docs/models), Llama2 70B was accessed using [AWS Bedrock service](https://aws.amazon.com/bedrock/), and Mistral Small was accessed using [Mistral API](https://docs.mistral.ai/getting-started/models/).
3. For AWS Bedrock, an AWS account would be needed and the aws cli would have to be setup to programmatically access the LLama2 model via Bedrock API. For [CLI Setup refer](https://docs.aws.amazon.com/cli/latest/userguide/getting-started-install.html) and [configuration setup refer](https://docs.aws.amazon.com/cli/latest/userguide/getting-started-quickstart.html). Alternately since Llama2 is open source it could be hosted locally as well.
4. The specific api versions used are as follows `gpt-3.5-turbo-1106`, `gpt-4-turbo-preview`, `meta.llama2-70b-chat-v1` and `mistral-small-latest`.
5. RAG was implemented using [SerpAPI](https://serpapi.com/search-api).
6. Similarity threshold was implemented using BERT [model](https://huggingface.co/sentence-transformers/bert-base-nli-mean-tokens) from hugging face.

## Parameters

`METHOD`: FPDAR, Self-Consistency, FourShot method to use. This will decide which prompts are used going further.

`MODEL_API`: LLM API to use. The model api is used to assign the model name and dynamically import the relevant `METHOD` function.

`EVAL_MODEL`: Evaluation LLM to use

`DATASET_NAME`: Dataset name to use

`DATASET_PATH`: Path to the dataset. 

`PROMPT_PATH`: Path to the prompts

`SIMILARITY_THRESHOLD`: Similarity threshold value for FPDAR FP Detection

`TEMPERATURE`: Temperature setting for LLM

`CANDIDATE_TEMPERATURE`: Candidate temperature setting for LLM

`MAX_CANDIDATE_RESPONSES`: Maximum candidate responses to be generated for SC method

`EVIDENCE_ALLOWED`: Boolean to decide to whether evidence should be used. This is used in FourShot method to stop the use of evidence.

`EVIDENCE_BATCH_GENERATE`: Boolean to decide whether evidence is generated in batch or individually for each question. Zero means by default batch will not be generated.

`EVIDENCE_BATCH_USE`: Boolean to decide whether batch evidence has to be used. One means by default batch will be used. If this is set to `0` and `EVIDENCE_ALLOWED` is set to `1`, then evidence is generated iteraively for each question.

`EVIDENCE_BATCH_SAVE_PATH`: Path to save evidence batch results

`QUERY_PROMPT_PATH`: Path to the query prompt file

`BACKWARD_REASONING_RESP_PROMPT_PATH`: Path to the abductive reasoning response prompt file

`BACKWARD_REASONING_QUERY_PROMPT_PATH`: Path to the abductive reasoning query prompt file

`LLAMA_QUERY_PROMPT_PATH`: Path to the LLAMA query prompt file

`LLAMA_BACKWARD_REASONING_RESP_PROMPT_PATH`: Path to the LLAMA abductive reasoning response prompt file

`LLAMA_BACKWARD_REASONING_QUERY_PROMPT_PATH`: Path to the LLAMA abductive reasoning query prompt file

`AUTO_EVALUATION_PROMPT_PATH`: Path to the auto evaluation prompt file

`RESULT_SAVE_PATH`: Path to save results


### Inference

1. Execute requirements.txt to install the relevant Python packages.
2. Set the necessary parameters and run `main.py`.
