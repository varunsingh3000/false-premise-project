import numpy as np

# function to parse through the api response and extract certain keywords in a dict
def process_response(chat_completion):
    #use the chat_completion object to retrieve the textual LLM response
    text = chat_completion[:]
    # Remove all newline characters ("\n")
    text_without_newlines = text.replace('\n', '')

    # Define key terms to split the text into sections
    key_terms = ['Explanation:', 'Answer:', 'Confidence Level:', 'Source:', 'Core Concept:', 'Premise of the Question:']
    response_dict = {}
    # Splitting the text into sections based on key terms
    for i in range(len(key_terms) - 1):
        term = key_terms[i]
        next_term = key_terms[i + 1]
        if term in text_without_newlines and next_term in text_without_newlines:
            split_text = text_without_newlines.split(term, 1)[1].split(next_term, 1)
            response_dict[term.strip()] = split_text[0].strip() if len(split_text) > 1 else ''

    # For the last key term
    last_term = key_terms[-1]
    if last_term in text_without_newlines:
        split_text = text_without_newlines.split(last_term, 1)
        response_dict[last_term.strip()] = split_text[1].strip() if len(split_text) > 1 else ''

    return response_dict


def uncertainty_confidence_cal(confi_match_list,confi_list):
    # first we remove the % symbol from the list values
    if np.sum(confi_list) != 0:
        final_confidence_value = round(np.divide(np.sum(confi_match_list),np.sum(confi_list)) * 100, 2)
    else:
        return 0
    return final_confidence_value


def matching_condition_check(match_count,MAX_CANDIDATE_RESPONSES,MATCH_CRITERIA):
    if MATCH_CRITERIA == "Half":
        match_check = MAX_CANDIDATE_RESPONSES//2
        if match_count > match_check:
            return True
    if MATCH_CRITERIA == "Full":
        match_check = MAX_CANDIDATE_RESPONSES
        if match_count == match_check:
            return True

def check_dict_keys_condition(response_dict):
    key_terms = ['Explanation:', 'Answer:', 'Confidence Level:', 'Source:', 'Core Concept:', 'Premise of the Question:']
    keys_present = all(key in response_dict for key in key_terms)
    if keys_present:
        return True
    return False

def extract_question_after_binary(prompt):
    # Find the index of "Binary Question: "
    binary_index = prompt.find("Binary Question:")

    if binary_index != -1:
        # Extract the text after "Binary Question: "
        binary_question_text = prompt[binary_index + len("Binary Question:"):].strip()
        
        # Find the next space after the binary question text
        next_space_index = binary_question_text.find("")

        # Extract the question after the binary question key
        if next_space_index != -1:
            extracted_question = binary_question_text[next_space_index:].strip()
            return extracted_question
        else:
            return "No question found after the Binary Question key."
    else:
        return "Binary question key not found."
