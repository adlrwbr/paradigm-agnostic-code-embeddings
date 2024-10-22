import argparse
import os
from typing import Literal

import torch
from transformers import AutoModel, AutoTokenizer

CHECKPOINTS = {
    "codet5": "Salesforce/codet5p-110m-embedding",
    "codellama": "meta-llama/CodeLlama-7b-hf",
}
device = "cpu"  # for GPU usage or "cpu" for CPU usage


def load_model_and_tokenizer(model_type: str, device: str):
    checkpoint = CHECKPOINTS[model_type]
    tokenizer = AutoTokenizer.from_pretrained(checkpoint, trust_remote_code=True)
    model = AutoModel.from_pretrained(checkpoint, trust_remote_code=True).to(device)
    return model, tokenizer


# Create generator
DATASET_PATH = "data/stack/estimationset"
EMBEDDING_PATH = "embeddings"
LANGUAGES = ["java", "javascript", "php", "python"]


def create_dirs(model_type: str) -> None:
    for lang in LANGUAGES:
        os.makedirs(os.path.join(EMBEDDING_PATH, model_type, lang), exist_ok=True)


def stream_snippets(lang: str):
    lang_path: str = os.path.join(DATASET_PATH, lang)
    snippet_paths = map(lambda p: os.path.join(lang, p), os.listdir(lang_path))

    for snippet_path in snippet_paths:
        with open(os.path.join(DATASET_PATH, snippet_path), "r") as f:
            yield snippet_path, f.read()


def gen_embedding(source: str, model, tokenizer, device: str):
    inputs = tokenizer.encode(source, return_tensors="pt").to(device)
    with torch.no_grad():
        return model(inputs)[0]


def save_embedding(snippet_path: str, embedding, model_type: str):
    embedding_path = os.path.join(EMBEDDING_PATH, model_type, snippet_path + ".pt")
    torch.save(embedding, embedding_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate embeddings using CodeT5 or CodeLlama"
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        choices=["codet5", "codellama"],
        help="The model to use for generating embeddings: codet5 or codellama.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        choices=["cpu", "cuda"],
        help="Device to use for inference: cpu or cuda (GPU)",
    )

    args = parser.parse_args()
    model_type = args.model
    device = args.device

    model, tokenizer = load_model_and_tokenizer(model_type, device)

    create_dirs(model_type)

    # Generate and save embeddings for each language
    for lang in LANGUAGES:
        for snippet_path, snippet in stream_snippets(lang):
            embedding = gen_embedding(snippet, model, tokenizer, device)
            save_embedding(snippet_path, embedding, model_type)
