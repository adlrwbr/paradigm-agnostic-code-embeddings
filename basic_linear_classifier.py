import torch
import argparse
import matplotlib.pyplot as plt

class LinearClassifier(torch.nn.Module):
    def __init__(self, input_dim=768, output_dim=4):
        super(LinearClassifier, self).__init__()
        self.linear = torch.nn.Linear(input_dim, output_dim)
    
    def forward(self, x):
        x = self.linear(x)
        return x
        
LLM = "codebert" 
LANGUAGES = ["python", "java", "javascript", "php"]


def load_embeddings(start_range, end_range, subMeaned=False):
    labels = []
    tensors = []
    mean_tensors = []
    if subMeaned:
        for lang in range(0,len(LANGUAGES)):
            mean_tensors.append(torch.load("embeddings/" + LLM + "/" + LLM + "_mean_" + LANGUAGES[lang] + ".pt", weights_only=True))
        
    for i in range(start_range, end_range):
        for lang in range(0,len(LANGUAGES)):
            embedding = torch.load("embeddings/" + LLM + "/" + LANGUAGES[lang] + "/" + LANGUAGES[lang] + str(i) + ".txt.pt", weights_only=True)
            if subMeaned:
                embedding = torch.sub(embedding, mean_tensors[lang])
            res = [0.0,0.0,0.0,0.0]
            res[lang] = 1.0
            labels.append(res)
            tensors.append(embedding)
    x = torch.stack(tensors)
    y = torch.tensor(labels)
    if subMeaned:
        return x, y, mean_tensors
    return x, y
    

def generate_model(filename=LLM+"LinClass.model", drawFigure=True, epochs=600):
    model = LinearClassifier()
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
            
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

def load_model(filename=LLM + "LinClass.model"):
    model = torch.load(filename)
    model.eval()
    return model

def test_model(model):
    x_test, labels = load_embeddings(2200,2800)
    res = model.forward(x_test)
    
    x_mod, _, means = load_embeddings(2200,2800, True)
    mod_res = model.forward(x_mod)
    
    x_subbed = []
    subbed_successes = []
    sub_res = []
    for i in range(len(LANGUAGES)):
        x_subbed.append(torch.sub(x_test, means[i]))
        sub_res.append(model.forward(x_subbed[-1]))
        subbed_successes.append([0,0,0,0])
    
    label_successes = [0,0,0,0]
    modded_successes = [0,0,0,0]
    for i in range(len(labels)):
        if torch.argmax(labels[i]) == torch.argmax(res[i]):
            label_successes[torch.argmax(res[i])]+=1
        if torch.argmax(labels[i]) == torch.argmax(mod_res[i]):
            modded_successes[torch.argmax(mod_res[i])]+=1
        for j in range(len(LANGUAGES)):
            if torch.argmax(labels[i]) == torch.argmax(sub_res[j][i]):
                subbed_successes[j][torch.argmax(labels[i])] += 1
    #print(label_successes)
    #print(modded_successes)
    
    debug = False
    if debug:
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
        print("\n\n")
    
    for i in range(0,4):
        label_successes[i] = str(label_successes[i]/600.0)[1:4]
        for j in range(0,4):
            subbed_successes[j][i] = str(subbed_successes[j][i]/600.0)[1:4]
    print("Accuracy of Language Classification")
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
        choices=["codebert"],
        help="The model that sourced the embeddings: codebert, or [Implemented Later]",
    )

    args = parser.parse_args()
    LLM = args.model
    
    if args.loadFromFile:
        model = load_model(filename=args.filename)
    else:
        model = generate_model(filename=args.filename)
    test_model(model)

