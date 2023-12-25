import numpy as np

def uncertainty_confidence_cal(confi_match_list,confi_list):
    # first we remove the % symbol from the list values
    final_confidence_value = round(np.divide(np.sum(confi_match_list),np.sum(confi_list)) * 100, 2)
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
