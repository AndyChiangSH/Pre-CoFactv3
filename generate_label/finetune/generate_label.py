import os
import json
import torch
import yaml
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline
import argparse


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


def get_argument():
    opt = argparse.ArgumentParser()
    opt.add_argument("--model",
                     type=str,
                     help="model")
    opt.add_argument("--checkpoint",
                     type=str,
                     help="checkpoint")
    opt.add_argument("--mode",
                     type=str,
                     help="mode")
    opt.add_argument("--device",
                     type=str,
                     help="device")
    args = vars(opt.parse_args())

    return args


def preprocess_data(data):
    sep_token = " " + tokenizer.sep_token + " "

    preprocess_data = []

    for i in tqdm(range(len(data))):
        claim = data[i]["claim"]
        evidence = data[i]["evidence"]
        questions = data[i]["question"]
        claim_answers = data[i]["claim_answer"]
        evidence_answers = data[i]["evidence_answer"]

        if len(claim) > config["claim_max_len"]:
            claim = claim[:config["claim_max_len"]]

        if len(evidence) > config["evidence_max_len"]:
            evidence = evidence[:config["evidence_max_len"]]

        text = claim + sep_token + evidence

        if config["add_qa"]:
            for j in range(len(questions)):
                try:
                    question = questions[j]
                    claim_answer = claim_answers[j]
                    evidence_answer = evidence_answers[j]

                    if config["add_q"]:
                        if len(question) > config["question_max_len"]:
                            question = question[:config["question_max_len"]]

                        text += sep_token + question

                    if config["add_a"]:
                        if len(claim_answer) > config["claim_answer_max_len"]:
                            claim_answer = claim_answer[:config["claim_answer_max_len"]]

                        if len(evidence_answer) > config["evidence_answer_max_len"]:
                            evidence_answer = evidence_answer[:
                                                              config["evidence_answer_max_len"]]

                        text += sep_token + claim_answer + sep_token + evidence_answer
                except:
                    pass

        # label = label2num[data[i]["label"]]

        preprocess_data.append({
            "text": text,
            # "label": label,
        })

    return preprocess_data


if __name__ == '__main__':
    # clean GPU memory
    torch.cuda.empty_cache()

    args = get_argument()

    print("Reading config...")
    with open(f"./generate_label/finetune/model/{args['model'].split('/')[0]}/config.yaml", "r") as file:
        config = yaml.safe_load(file)

    print("config:", config)

    input_data_path = f"./data/{args['mode']}.json"
    print(f"Load data from {input_data_path}...")
    with open(input_data_path, 'r') as f:
        data = json.load(f)
        
    print(f"data length: {len(data)}")
    
    device = torch.device(
        args['device'] if torch.cuda.is_available() else "cpu")
    print("device:", device)
        
    model_path = f"./generate_label/finetune/model/{args['model']}"
    # model_path = f"./generate_label/finetune/model/{args['model']}/checkpoint-{args['checkpoint']}"
    print("model_path:", model_path)
    
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    classifier = pipeline("text-classification",
        model=model, tokenizer=tokenizer, device=device)
    
    print("Preprocess data...")
    preprocessed_data = preprocess_data(data)
    
    new_data = []
    new_prob = []
      
    for i in tqdm(range(len(data))):
        obj = {
            "id": data[i]["id"],
            "claim_id": data[i]["claim_id"],
            "claim": data[i]["claim"],
            "evidence": data[i]["evidence"],
            "question": data[i]["question"],
            "claim_answer": data[i]["claim_answer"],
            "evidence_answer": data[i]["evidence_answer"],
        }
        
        result = classifier(
            preprocessed_data[i]["text"], num_workers=8, top_k=3)
        # print("result:", result)
        
        obj["label"] = result[0]["label"]
        
        prob = [0, 0, 0]
        for r in result:
            prob[label2num[r["label"]]] = r["score"]
        # print("prob:", prob)
            
        new_data.append(obj)
        new_prob.append(prob)
                
    output_folder_path = f"./generate_label/finetune/label/{args['model']}"
    if not os.path.exists(output_folder_path):
        os.makedirs(output_folder_path)

    output_data_path = f"{output_folder_path}/{args['mode']}.json"
    print(f"Save data to {output_data_path}...")
    with open(output_data_path, 'w') as f:
        json.dump(new_data, f, indent=2)
        
    # print("new_prob:", new_prob)
    output_path = f"{output_folder_path}/{args['mode']}_prob.json"
    print(f"Save data to {output_path}...")
    with open(output_path, 'w') as f:
        json.dump(new_prob, f, indent=2)
