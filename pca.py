import torch
import os
import matplotlib.pyplot as plt
import matplotlib
from torch_pca import PCA

MODEL = "codet5"
number = 1000
offset = 0

def T(mat):
    return torch.transpose(mat, 0, 1)

centering = True

tensors = []
languages = ["go", "python", "c-sharp"]
embeddings = []
abmat = torch.load(os.path.join("embeddings", MODEL, MODEL + "_CSLRD9.pt"))
lang_means = []
for lang in languages:
    mean_embed = torch.load(os.path.join("embeddings", MODEL, MODEL + "_LRD19_" + lang + ".pt"))
    embed_path = os.path.join("embeddings", MODEL, lang, lang)
    lang_tens = []
    for i in range(0,number):
        tensor = torch.load(embed_path + str(i+offset) + ".txt.pt", weights_only=True, map_location=torch.device('cpu'))
        if centering:
            tensor = torch.matmul(mean_embed, tensor)
        tensors.append(tensor)
        lang_tens.append(tensor)
    embeddings.append(torch.stack(lang_tens))

tensors=torch.stack(tensors)
#embeddings = torch.stack(tensors)


pca_model = PCA(n_components=2, svd_solver='full')

colors = ['#ff00d1', '#0011ff' , '#ff0000']
x,y = T(pca_model.fit_transform(tensors))
for i in range(len(languages)):
    plt.scatter(x[i*1000:i*1000+1000],y[i*1000:i*1000+1000],label=languages[i],c=colors[i])
#plt.scatter(x,y,label='all')
#plt.legend(loc='lower right')
plt.show()
