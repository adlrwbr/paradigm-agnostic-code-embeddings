# Paradigm Agnostic Code Embeddings

This is a final project for Computational Linguistics II. The project comes in two parts:

1. Replicating the results of [Language Agnostic Code Embeddings by Utpala et al., 2023](https://arxiv.org/abs/2310.16803).
2. Extension to analyze the paradigm (e.g. functional, procedural, OO) components of code embeddings.

## Setting up dev environment

1. Install [mamba](https://github.com/mamba-org/mamba), an efficient alternative to conda.
2. With `mamba` installed, the `micromamba` binary should be available. To get started, `micromamba create -f environment.yml`.
3. Activate your virtual environment with `micromamba activate cl2-final`.
4. Then, `cp .env.example .env` and enter your Hugging Face access token to [use gated models](https://huggingface.co/docs/transformers.js/en/guides/private) like CodeLlama.
