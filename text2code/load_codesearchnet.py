"""
Utilities to
- Download Huggingface dataset
- Perform filtering / preprocessing
- Stream data for downstream tasks
"""

from datasets import load_dataset
from tqdm import tqdm
import torch
from transformers import AutoTokenizer, AutoModel
import os
import numpy as np

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

    subset_mapping = dict()
    for lang in languages:
        print(f"Downloading {lang}...")
        lang_dataset = load_dataset(DATASET_TAG, data_files=f"data/{lang}.zip")
        subset_mapping[lang] = lang_dataset

    return subset_mapping

@torch.no_grad()
def gen_embeds(model, tokenizer, dataset, lang, dataset_len: int = 50):
    text_embeds = list()
    code_embeds = list()

    for i in tqdm(range(dataset_len)):
        example = dataset[lang]['test'][i]
        
        text_input = tokenizer(example['func_documentation_string'], padding='max_length', truncation=True, max_length=64,
                            return_tensors="pt").to('cuda')
        code_input = tokenizer(example['func_code_string'], padding='max_length', truncation=True, max_length=360,
                            return_tensors="pt").to('cuda')
        
        text_embed = model(text_input.input_ids, attention_mask=text_input.attention_mask)
        code_embed = model(code_input.input_ids, attention_mask=code_input.attention_mask)

        text_embeds.append(text_embed)
        code_embeds.append(code_embed)
    
    text_embeds = torch.stack(text_embeds)
    code_embeds = torch.stack(code_embeds)

    torch.save(text_embeds, f"outputs/{lang}_text.pt")
    torch.save(code_embeds, f"outputs/{lang}_code.pt")

def load_embeds(lang: str):
    text_embeds = torch.load(f"outputs/{lang}_text.pt") 
    code_embeds = torch.load(f"outputs/{lang}_code.pt") 

    return text_embeds, code_embeds

def load_mean_embed(lang: str, center_method: str):
    if center_method == "mean":
        mean_embeddings = torch.load("../codet5_mean_embeddings.dict")
        return mean_embeddings[lang]
    elif center_method == "LRD":
        return torch.load(f'../embeddings/codet5/codet5_LRD10_{lang}.pt')

@torch.no_grad()
def contrast_evaluation(text_embeds, code_embeds):
    score_matrix_i2t = text_embeds @ code_embeds.t()
    scores_i2t = score_matrix_i2t.cpu().numpy()

    ranks = np.ones(scores_i2t.shape[0]) * -1
    for index, score in enumerate(scores_i2t):
        inds = np.argsort(score)[::-1]
        ranks[index] = np.where(inds == index)[0][0]

    # Compute metrics
    tr1 = 100.0 * len(np.where(ranks < 1)[0]) / len(ranks)
    tr5 = 100.0 * len(np.where(ranks < 5)[0]) / len(ranks)
    tr10 = 100.0 * len(np.where(ranks < 10)[0]) / len(ranks)
    mrr = 100.0 * np.mean(1 / (ranks + 1))

    eval_result = {'r1': tr1,
                   'r5': tr5,
                   'r10': tr10,
                   'mrr': mrr}
    return eval_result

if __name__ == "__main__":
    # subsets = load_filtered_codesearchnet(FILTERED_LANGUAGES)

    # model_name = "Salesforce/codet5p-110m-embedding"
    # tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    # model = AutoModel.from_pretrained(model_name, trust_remote_code=True).to('cuda')
    # model.eval()

    # gen_embeds(model, tokenizer, subsets, 'javascript', 50)
    # gen_embeds(model, tokenizer, subsets, 'python', 50)
    # gen_embeds(model, tokenizer, subsets, 'java', 50)
    # gen_embeds(model, tokenizer, subsets, 'php', 50)

    text_embeds, code_embeds = load_embeds('python')
    text_embeds = text_embeds.view((50, 256))
    code_embeds = code_embeds.view((50, 256))

    mean_embed = load_mean_embed('python', 'mean').to('cuda')

    centered_embeds = code_embeds - mean_embed * 50000

    print(contrast_evaluation(text_embeds, code_embeds))
    print(contrast_evaluation(text_embeds, centered_embeds))