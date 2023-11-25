import json
import yaml
from tqdm import tqdm
from transformers import AutoTokenizer

claim_max_len = 128
evidence_max_len = 1024
mode = "train"

label2num = {
    "Support": 0,
    "Neutral": 1,
    "Refute": 2,
}

num2label = {
    0: "Support",
    1: "Neutral",
    2: "Refute",
}

if __name__ == '__main__':
    with open("./finetune/config.yaml", "r") as file:
        config = yaml.safe_load(file)
    print("config:", config)

    load_data_path = f"./data/{config['mode']}.json"
    print(f"Load data from {load_data_path}...")
    with open(load_data_path, 'r') as f:
        data = json.load(f)

    print(f"Data length: {len(data)}")
    
    tokenizer = AutoTokenizer.from_pretrained(config["model"])

    preprocess_data = []

    for i in tqdm(range(len(data))):
        claim = data[i]["claim"]
        evidence = data[i]["evidence"]
        label = label2num[data[i]["label"]]

        if len(claim) > config["claim_max_len"]:
            claim = claim[:config["claim_max_len"]]

        if len(evidence) > config["evidence_max_len"]:
            evidence = claim[:config["evidence_max_len"]]

        text = claim + " " + tokenizer.sep_token + " " + evidence

        preprocess_data.append({
            "text": text,
            "label": label,
        })

    save_data_path = f"./finetune/data/{config['mode']}_preprocess_1.json"
    print(f"Save data to {save_data_path}...")
    with open(save_data_path, 'w') as f:
        json.dump(preprocess_data, f, indent=2)
