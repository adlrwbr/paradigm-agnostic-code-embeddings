"""
Utilities to
- Download Huggingface dataset
- Perform filtering / preprocessing
- Stream data for downstream tasks
"""

from datasets import load_dataset

# Huggingface reference name
DATASET_TAG: str = "code-search-net/code_search_net"

# Languages to be evaluated for downstream tasks (use all centered languages)
FILTERED_LANGUAGES = {"java", "python", "php", "javascript"}

def load_filtered_codesearchnet(languages: set[str]):
    """
    Download necessary files from Huggingface code_search_net dataset locally

    Languages: python, java, php, javascript
    Returns dictionary that maps language name to datasubset
    """

    subset_mapping = dict({lang: load_dataset(DATASET_TAG, data_files=f"{lang}.zip")} for lang in languages)
    return subset_mapping