from datasets import load_dataset
import os

def get_sampler(lang):
    ds = load_dataset("bigcode/the-stack", streaming=True, data_dir="data/" + lang, split="train")
    return iter(ds)
    
langs = ["python","java","javascript","php"]

for lang in langs:
    try:
        os.mkdir(lang)
        print(f"Directory '{lang}' created successfully.")
    except FileExistsError:
        print(f"Directory '{lang}' already exists.")
    except PermissionError:
        print(f"Permission denied: Unable to create '{lang}'.")

# Exclusions are files that gave Windows Defender a scare. 
# I doubt they are actually malicious but I figured just for convenience's sake I would exclude those files from the set. 
exclude = {}
exclude["python"] = [2152,2856, 4038]
exclude["java"] = []
exclude["javascript"] = [7405]
exclude["php"] = []

for lang in langs:
    i = 0
    sampler = get_sampler(lang)
    print(f"Pulling '{lang}' files.")
    for sample in sampler:
        #print(sample["content"])
        #print(sample["ext"])
        if (i) in exclude[lang]:
            exclude[lang].remove(i)
            continue
        f = open(lang + "/" + lang + str(i) + ".txt", 'w', encoding="utf-8")
        f.write(sample["content"])
        f.close()
        i += 1
        if i > 9999:
            break
