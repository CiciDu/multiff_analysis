# Capture vs miss
CAPTURE_VS_MISS = {
    'new_label_mapping': {
        'capture_over_miss': 'Capture',
        'miss_over_attempt': 'Miss',
    },
    'category_order': ['Capture', 'Miss'],
    'title': 'Attempt Outcomes Across Early and Late Sessions',
    'y_label': 'Proportion of Attempts',
    'category_colors': ['#0072B2', '#C76E00']
}

CAPTURE_VS_MISS = {
    'new_label_mapping': {
        'first_shot_capture_over_attempt': 'Capture',
        'miss_over_attempt': 'Miss',
    },
    'category_order': ['Capture', 'Miss'],
    'title': 'Attempt Outcomes Across Early and Late Sessions',
    'y_label': 'Proportion of Attempts',
    'category_colors': ['#0072B2', '#C76E00']
}

# Attempt breakdown
ATTEMPT_BREAKDOWN = {
    'new_label_mapping': {
        'first_shot_capture_over_attempt': 'Capture',
        'no_retry_over_attempt': 'No Retry',
        'retry_capture_over_attempt': 'Retry + Capture',
        'retry_fail_over_attempt': 'Retry + Fail',
    },
    'category_order': ['Capture', 'Retry + Capture', 'Retry + Fail', 'No Retry'],
    'title': 'Attempt Outcomes Across Early and Late Sessions',
    'y_label': 'Proportion of Attempts',
    'category_colors': None
}


# After miss
AFTER_MISS = {
    'new_label_mapping': {
        'retry_capture_over_miss': 'Retry + Capture',
        'retry_fail_over_miss': 'Retry + Fail',
        'no_retry_over_miss': 'No Retry',
    },
    'category_order': ['Retry + Capture', 'Retry + Fail', 'No Retry'],
    'title': 'Actions After a Miss Across Early and Late Sessions',
    'y_label': 'Proportion of Misses',
    'category_colors': ['#CC79A7', '#E69F00', '#009E73']
}


# Eventual outcome
EVENTUAL_OUTCOME = {
    'new_label_mapping': {
        'eventual_capture_over_attempt': 'Capture',
        'eventual_miss_over_attempt': 'Miss',
    },
    'category_order': ['Capture', 'Miss'],
    'title': 'Eventual Attempt Outcomes Across Early and Late Sessions',
    'y_label': 'Proportion of Attempts',
    'category_colors': ['#4e79a7', '#98df8a']
}


# Retry outcome
RETRY_OUTCOME = {
    'new_label_mapping': {
        'rcap_over_both': 'Retry + Capture',
        'rsw_over_both': 'Retry + Fail',
    },
    'category_order': ['Retry + Capture', 'Retry + Fail'],
    'title': 'Actions After a Miss Across Early and Late Sessions',
    'y_label': 'Proportion of Misses',
    'category_colors': ['#CC79A7', '#E69F00']
}