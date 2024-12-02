import torch
import argparse
import matplotlib.pyplot as plt
import os
import numpy as np

LLM = None
LANGUAGES = ["python", "java", "javascript", "php"]
EMBEDDING_PATH = "embeddings"
DEVICE = 'cpu'

RANK_LIMITS = {
    "codebert": 6,
    "codet5": 9
    }

class LinearClassifier(torch.nn.Module):
    def __init__(self, input_dim=768, output_dim=4):
        super(LinearClassifier, self).__init__()
        if LLM == "codet5":
            input_dim=256
        self.linear = torch.nn.Linear(input_dim, output_dim)
    
    def forward(self, x):
        x = self.linear(x)
        return x

# makeAgnostic determines whether to modify the vectors to remove language specific elements
# rank = 0 uses mean embedding, otherwise uses LRD/CSLRD with that rank
def load_embeddings(start_range, end_range, makeAgnostic=False, rank=0, CSLRD=False):
    if CSLRD:
        makeAgnostic = True
    labels = []
    tensors = []
    language_tensors = []
    if makeAgnostic:
        if CSLRD:
            mat_path = os.path.join(EMBEDDING_PATH, LLM, LLM + "_CSLRD" + str(rank) + ".pt")
            language_tensors.append(torch.load(mat_path, weights_only=True, map_location=torch.device(DEVICE)))
        else:
            for lang in LANGUAGES:
                if rank==0:
                    language_tensors.append(torch.load("embeddings/" + LLM + "/" + LLM + "_mean_" + lang + ".pt", weights_only=True, map_location=torch.device(DEVICE)))
                else: 
                    mat_path = os.path.join(EMBEDDING_PATH, LLM, LLM + "_LRD" + str(rank) + "_" + lang + ".pt")
                    language_tensors.append(torch.load(mat_path, weights_only=True, map_location=torch.device(DEVICE)))
    
    for i in range(start_range, end_range):
        for lang in range(0,len(LANGUAGES)):
            embedding_path = os.path.join(EMBEDDING_PATH, LLM, LANGUAGES[lang], LANGUAGES[lang] + str(i) + ".txt.pt")
            embedding = torch.load(embedding_path, weights_only=True, map_location=torch.device(DEVICE))
            if makeAgnostic:
                if rank == 0:
                    embedding = torch.sub(embedding, language_tensors[lang])
                elif CSLRD: 
                    embedding = embedding - torch.matmul(language_tensors[0], embedding)
                else:
                    embedding = torch.matmul(language_tensors[lang], embedding)
            res = [0.0,0.0,0.0,0.0]
            res[lang] = 1.0
            labels.append(res)
            tensors.append(embedding)
    x = torch.stack(tensors)
    y = torch.tensor(labels)
    if makeAgnostic:
        if CSLRD:
            return x, y, language_tensors[0]
        return x, y, language_tensors
    return x, y
    

def generate_model(filename, drawFigure=True, epochs=600):
    model = LinearClassifier()
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.02)
            
    x_train, y_train = load_embeddings(1000,2200)

    all_loss = []
    for epoch in range(0,epochs):
      output = model(x_train)

      loss = criterion(output, y_train)
      if drawFigure:
        all_loss.append(loss.item())
      loss.backward()

      optimizer.step()
      optimizer.zero_grad()
    
    if drawFigure:
        plt.plot(all_loss)
        plt.xlabel('epochs')
        plt.ylabel('loss')
        plt.savefig('loss.png')

    torch.save(model, filename)
    return model

def load_model(filename):
    model = torch.load(filename)
    model.eval()
    return model

def format_decimal(num, length):
    if num == 1:
        val = "1."
    else:
        val = str(num)[1:length+1]
    while len(val) < length:
        val = val + "0"
    return val
        

def draw_model(model):
    rank_limit = RANK_LIMITS[LLM]
    
    x_test, labels = load_embeddings(2200,2800)
    res = model.forward(x_test)
    
    x_center, _, _ = load_embeddings(2200,2800, True)
    center_res = model.forward(x_center)
    
    lrd_successes = []
    lrd_res = []
    x2 = []
    for i in range(1, 21, 2):
        x2.append(i)
        x_lrd, _, _ = load_embeddings(2200,2800, True, i)
        y = model.forward(x_lrd)
        lrd_res.append(y)
        lrd_successes.append(0)
    
    cslrd_successes = []
    cslrd_res = []
    for i in range(1, rank_limit+1):
        x, _, _ = load_embeddings(2200,2800, True, i, True)
        y = model.forward(x)
        cslrd_res.append(y)
        cslrd_successes.append(0)
    
    label_successes = 0
    center_successes = 0
    for i in range(len(labels)):
        correct = torch.argmax(labels[i])
        if correct == torch.argmax(res[i]):
            label_successes+=1
        if correct == torch.argmax(center_res[i]):
            center_successes+=1
        for j in range(0,rank_limit):
            if correct == torch.argmax(cslrd_res[j][i]):
                cslrd_successes[j] +=1
        for j in range(0, 10):
            if correct == torch.argmax(lrd_res[j][i]):
                lrd_successes[j] +=1
    plt.clf()
    plt.xlabel('Rank')
    plt.ylabel('Language Identification Accuracy')
    
    x = []
    y, y2, y3, y4, y5 = np.zeros(19), np.zeros(19), np.zeros(10), np.zeros(rank_limit), 25*np.ones(19)
    # Normal, Centering, LRD, CSLRD, Baseline
    
    for i in range(0, 19):
        x.append(i+1)
        y[i] = label_successes/24.0
        y2[i] = center_successes/24.0
        y5[i] = 25
    
    for i in range(0, 10):
        y3[i] = lrd_successes[i]/24.0
    
    for i in range(0, rank_limit):
        y4[i] = cslrd_successes[i]/24.0
    
    ax = plt.gca()
    ax.set_xlim([1, 19])
    ax.set_ylim([0, 100])
    plt.xticks(range(1, 20, 2))
    plt.title(LLM)
    
    plt.plot(x,y, label='unmodified')
    plt.plot(x,y2, label='centering')
    plt.plot(x2,y3, marker='o', label='LRD')
    plt.plot(x[:rank_limit],y4, marker='o', label='CS-LRD')
    plt.plot(x,y5, '-.', label='baseline')
    plt.legend(loc='lower right')
    plt.savefig(LLM + '_rank_vs_accuracy.png')
    

def test_model(model):
    rank_limit = RANK_LIMITS[LLM]
    rank = 5
    x_test, labels = load_embeddings(2200,2800)
    res = model.forward(x_test)
    
    x_mod, _, means = load_embeddings(2200,2800, True)
    mod_res = model.forward(x_mod)
    
    x_lrd, _, agnostic_matrices = load_embeddings(2200,2800, True, rank)
    lrd_res = model.forward(x_lrd)
    
    cslrd_res = []
    
    x_subbed = []
    subbed_successes = []
    sub_res = []
    x_mulled = []
    mulled_successes = []
    mulled_predicted = []
    mul_res = []
    for i in range(len(LANGUAGES)):
        x_subbed.append(torch.sub(x_test, means[i]))
        sub_res.append(model.forward(x_subbed[-1]))
        subbed_successes.append([0,0,0,0])
        
        x_mulled.append(torch.matmul(x_test, agnostic_matrices[i]))
        mul_res.append(model.forward(x_mulled[-1]))
        mulled_successes.append([0,0,0,0])
        mulled_predicted.append([0,0,0,0])
    
    cslrd_successes = []
    for i in range(1, rank_limit+1):
        x, _, _ = load_embeddings(2200,2800, True, i, True)
        y = model.forward(x)
        cslrd_res.append(y)
        cslrd_successes.append([0,0,0,0])
    
    label_successes = [0,0,0,0]
    modded_successes = [0,0,0,0]
    lrd_successes = [0,0,0,0]
    
    
    for i in range(len(labels)):
        correct = torch.argmax(labels[i])
        if correct == torch.argmax(res[i]):
            label_successes[correct]+=1
        if correct == torch.argmax(mod_res[i]):
            modded_successes[correct]+=1
        if correct == torch.argmax(lrd_res[i]):
            lrd_successes[correct]+=1
        for j in range(len(LANGUAGES)):
            if correct == torch.argmax(sub_res[j][i]):
                subbed_successes[j][correct] += 1
            if correct == torch.argmax(mul_res[j][i]):
                mulled_successes[j][correct] += 1
            mulled_predicted[j][torch.argmax(mul_res[j][i])]+=1
        for j in range(0,rank_limit):
            if correct == torch.argmax(cslrd_res[j][i]):
                cslrd_successes[j][correct] +=1
    
    print(lrd_successes)
    
    for i in range(0,4):
        label_successes[i] = str(label_successes[i]/600.0)[1:4]
        for j in range(0,4):
            subbed_successes[j][i] = str(subbed_successes[j][i]/600.0)[1:4]
            mulled_successes[j][i] = str(mulled_successes[j][i]/600.0)[0:4]
    print("Accuracy of Language Classification")
    print("Centering")
    print("  Original:  python  java  JS   PHP")
    print(" ↓Removed↓ |=======================")
    print("None       |", end="")
    print(" " + str(label_successes[0]) + "     " + str(label_successes[1]) +
         "   " + str(label_successes[2]) + "  " + str(label_successes[3]))
    for j in range(len(LANGUAGES)):
        print(LANGUAGES[j],end="")
        for i in range(11-len(LANGUAGES[j])):
            print(" ",end="")
        print("|",end="")
        print(" " + str(subbed_successes[j][0]) + "     " + str(subbed_successes[j][1]) +
         "   " + str(subbed_successes[j][2]) + "  " + str(subbed_successes[j][3]))
    
    print("LRD, rank " + str(rank))
    print("  Original:  python  java  JS   PHP")
    print(" ↓Removed↓ |=======================")
    print("None       |", end="")
    print(" " + str(label_successes[0]) + "     " + str(label_successes[1]) +
         "   " + str(label_successes[2]) + "  " + str(label_successes[3]))
    for j in range(len(LANGUAGES)):
        print(LANGUAGES[j],end="")
        for i in range(11-len(LANGUAGES[j])):
            print(" ",end="")
        print("|",end="")
        print(" " + str(mulled_successes[j][0]) + "     " + str(mulled_successes[j][1]) +
         "   " + str(mulled_successes[j][2]) + "  " + str(mulled_successes[j][3]))
         
         
    print("CS-LRD")
    print("Language:  python  java  JS   PHP  Overall")
    print(" ↓Rank↓ |==================================")
    print("None    | ", end="")
    print(" " + str(label_successes[0]) + "     " + str(label_successes[1]) +
         "   " + str(label_successes[2]) + "  " + str(label_successes[3]))
    
    langlens = ["     ", "   ", "  ", "   "]
    for i in range(0, rank_limit):
        print("    " + str(i+1) + "   |", end ="  ")
        sumSuccess = 0
        for j in range(0, len(LANGUAGES)):
            sumSuccess += cslrd_successes[i][j]
            print(format_decimal(cslrd_successes[i][j]/600.0, 3), end=langlens[j])
        print(str(sumSuccess/2400.0)[1:4])
    
    
        
    
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train/Load and Test a basic linnear classifier to predict language from embeddings."
    )
    parser.add_argument(
        "--loadFromFile",
        type=bool,
        default=False,
        help="whether to train a new Linear Classifier or load from file",
    )
    parser.add_argument(
        "--filename",
        type=str,
        required=True,
        help="file to save/load linear classifier to/from",
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        choices=["codebert", "codet5"],
        help="The model that sourced the embeddings: codebert, or [Implement Later]",
    )
    parser.add_argument(
        "--draw",
        type=str,
        default=False,
        help="whether to perform text based analysis or create an image",
    )

    args = parser.parse_args()
    LLM = args.model
    
    if args.loadFromFile:
        model = load_model(filename=args.filename)
    else:
        model = generate_model(filename=args.filename)
        
    if args.draw:
        draw_model(model)
    else:
        test_model(model)
    

