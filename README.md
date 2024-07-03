# FPDAR Repository

This repository will explain the process to run the inference, the evaluation of the LLM responses and how new methods, LLMs and/or QA datasets could be incorporated into the workflow.

Some implementation details to note:

1. Since FPDAR works at an inference level and no inherent changes to the model was being made, all LLMs were accessed using API services. GPT 3.5 turbo and 4 were accessed using
   [OpenAI]([url](https://platform.openai.com/docs/models)), Llama2 70B was accessed using AWS Bedrock service, and Mistral Small was accessed using Mistral API ([](url)) 

## Inference

1. Run requirements.txt to install the relevant python packages
