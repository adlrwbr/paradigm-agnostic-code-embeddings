import os
from transformers import AutoModel, AutoTokenizer
import torch

checkpoint = "Salesforce/codet5p-110m-embedding"
device = "cpu"  # for GPU usage or "cpu" for CPU usage

# Import embedding generator
tokenizer = AutoTokenizer.from_pretrained(checkpoint, trust_remote_code=True)
model = AutoModel.from_pretrained(checkpoint, trust_remote_code=True).to(device)

# Create generator
DATASET_PATH = "estimationset"
EMBEDDING_PATH = "embeddings/codet5"
LANGUAGES = ["java", "javascript", "php", "python"]

def create_dirs() -> None:
    for lang in LANGUAGES:
        os.makedirs(os.path.join(EMBEDDING_PATH, "codet5", lang))

def stream_snippets(lang: str):
    lang_path: str = os.path.join(DATASET_PATH, lang)
    snippet_paths = map(lambda p: os.path.join(lang, p), os.listdir(lang_path))

    for snippet_path in snippet_paths:
        with open(os.path.join(DATASET_PATH, snippet_path), "r") as f:
            yield snippet_path, f.read()

def gen_embedding(source: str):
    inputs = tokenizer.encode(source, return_tensors="pt").to(device)
    return model(inputs)[0]

def save_embedding(snippet_path: str, embedding):
    embedding_path = os.path.join(EMBEDDING_PATH, snippet_path + ".pt")
    torch.save(embedding, embedding_path)

if __name__ == "__main__":
   create_dirs()

   for lang in LANGUAGES:
       for snippet_path, snippet in stream_snippets(lang):
           embedding = gen_embedding(snippet)
           save_embedding(snippet_path, embedding)