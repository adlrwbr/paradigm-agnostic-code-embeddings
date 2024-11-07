import torch
import os
EMBEDDING_PATH = "embeddings"
MODEL = "codebert"
LANGUAGES = ["python", "java", "javascript", "php"]

model_path = os.path.join(EMBEDDING_PATH, MODEL)

tensors = []
n = 1000
r=1
for lang in LANGUAGES:
    embed_path = os.path.join(model_path, lang, lang)
    tensors = []
    for i in range(0,n):
        tensors.append(torch.load(embed_path + str(i) + ".txt.pt", weights_only=True))
    mean_embedding = torch.unsqueeze(torch.mean(torch.stack(tensors), 0), 0)
    torch.save(mean_embedding, os.path.join(model_path, MODEL + "_mean_" + lang + ".pt"))
    
    _, _, Vh = torch.linalg.svd(torch.stack(tensors))
    P = torch.matmul(Vh[:,:r], torch.transpose(Vh[:,:r], 0, 1))
    torch.save(P, os.path.join(model_path, MODEL + "_LRD" + str(r) + "_" + lang + ".pt"))
