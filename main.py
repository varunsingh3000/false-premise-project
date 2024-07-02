import argparse

from inference import start_workflow
from evaluation import start_evaluation

def create_parser():
    parser = argparse.ArgumentParser(description="Script parameters")
    
    parser.add_argument('--METHOD', type=str, choices=["FPDAR", "SC", "FourShot"],
                        default="FourShot", help='Method to use. This will decide which prompts are used going further.')
    parser.add_argument('--MODEL_API', type=str, choices=["gpt-3.5-turbo-1106", "mistral-small-latest", "meta.llama2-70b-chat-v1"],
                        default="gpt-3.5-turbo-1106", help='Model API to use')
    parser.add_argument('--EVAL_MODEL', type=str, default="gpt-4-turbo-preview", help='Evaluation model to use')
    parser.add_argument('--DATASET_NAME', type=str, choices=["freshqa", "QAQA"], default="freshqa", help='Dataset name to use')
    parser.add_argument('--DATASET_PATH', type=str, default="Data\\", help='Path to the dataset')
    parser.add_argument('--PROMPT_PATH', type=str, default="Prompts\\", help='Path to the prompts')
    parser.add_argument('--SIMILARITY_THRESHOLD', type=float, default=0.7, help='Similarity threshold value for FPDAR FP Detection')
    parser.add_argument('--TEMPERATURE', type=float, default=0.0, help='Temperature setting for the model')
    parser.add_argument('--CANDIDATE_TEMPERATURE', type=float, default=1.0, help='Candidate temperature setting for the model')
    parser.add_argument('--MAX_CANDIDATE_RESPONSES', type=int, default=3, help='Maximum candidate responses to be generated for SC method')
    parser.add_argument('--EVIDENCE_ALLOWED', type=int, default=1, help='Boolean to decide to whether evidence should be used')
    parser.add_argument('--EVIDENCE_BATCH_GENERATE', type=int, default=0, help='Boolean to decide whether evidence is generated in batch or indiviudally for each question. Zero means by default batch will not be generated.')
    parser.add_argument('--EVIDENCE_BATCH_USE', type=int, default=1, help='Boolean to decide whether batch evidence has to be used. One means by default batch will be used.')
    parser.add_argument('--EVIDENCE_BATCH_SAVE_PATH', type=str, default="Web_Search_Response\\evidence_results_batch_serp_all_freshqa.json", help='Path to save evidence batch results')
    parser.add_argument('--QUERY_PROMPT_PATH', type=str, default="Prompts\\Common\\resp_generation.txt", help='Path to the query prompt file')
    parser.add_argument('--BACKWARD_REASONING_RESP_PROMPT_PATH', type=str, default="Prompts\\FPDAR\\backward_reasoning_resp.txt", help='Path to the backward reasoning response prompt file')
    parser.add_argument('--BACKWARD_REASONING_QUERY_PROMPT_PATH', type=str, default="Prompts\\FPDAR\\backward_reasoning_query.txt", help='Path to the backward reasoning query prompt file')
    parser.add_argument('--LLAMA_QUERY_PROMPT_PATH', type=str, default="Prompts\\Common\\resp_generation_meta.txt", help='Path to the LLAMA query prompt file')
    parser.add_argument('--LLAMA_BACKWARD_REASONING_RESP_PROMPT_PATH', type=str, default="Prompts\\FPDAR\\backward_reasoning_resp_meta.txt", help='Path to the LLAMA backward reasoning response prompt file')
    parser.add_argument('--LLAMA_BACKWARD_REASONING_QUERY_PROMPT_PATH', type=str, default="Prompts\\FPDAR\\backward_reasoning_query_meta.txt", help='Path to the LLAMA backward reasoning query prompt file')
    parser.add_argument('--AUTO_EVALUATION_PROMPT_PATH', type=str, default="Prompts\\Common\\response_evaluation.txt", help='Path to the auto evaluation prompt file')
    parser.add_argument('--RESULT_SAVE_PATH', type=str, default="Results\\", help='Path to save results')
    
    return parser

def main():
    parser = create_parser()
    # Parse the command-line arguments
    args = parser.parse_args()
    # Print the parsed arguments (or use them as needed)
    print(args)
    # Start the main fp detection workflow and prepare the responses to be processed for evaluation
    start_workflow(args)
    # Start the main evaluation process
    start_evaluation(args)


if __name__ == "__main__":
    main()