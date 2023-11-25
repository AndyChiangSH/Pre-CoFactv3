import numpy as np
import os
import json
import torch
import yaml
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from transformers import DataCollatorWithPadding, TrainingArguments, Trainer
from datasets import load_dataset, Dataset
import argparse
import evaluate


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


def preprocess_data(data):
    preprocess_data = []
    
    for i in tqdm(range(len(data))):
        claim = data[i]["claim"]
        evidence = data[i]["evidence"]
        
        if len(claim) > config["claim_max_len"]:
            claim = claim[:config["claim_max_len"]]

        if len(evidence) > config["evidence_max_len"]:
            evidence = claim[:config["evidence_max_len"]]

        text = claim + " " + tokenizer.sep_token + " " + evidence
        
        for j in range(len(data[i]["question"])):
            pass
        
        label = label2num[data[i]["label"]]

        preprocess_data.append({
            "text": text,
            "label": label,
        })
        
    return preprocess_data


def preprocess_function(examples):
    return tokenizer(examples["text"], padding=True, truncation=True)

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    return accuracy.compute(predictions=predictions, references=labels)


if __name__ == '__main__':
    print("Reading config...")
    with open("./finetune/config.yaml", "r") as file:
        config = yaml.safe_load(file)
        
    print("config:", config)
    
    output_folder_path = f"./finetune/model/{config['finetune_model']}"
    if not os.path.exists(output_folder_path):
        os.makedirs(output_folder_path)
    
    print("output_folder_path:", output_folder_path)
        
    config_path = f"{output_folder_path}/config.yaml"
    print(f"Save config to {config_path}")
    with open(config_path, 'w') as config_file:
        yaml.dump(config, config_file)
        
    train_data_path = f"./data/train.json"
    print(f"Load data from {train_data_path}...")
    with open(train_data_path, 'r') as f:
        train_data = json.load(f)

    print(f"Train data length: {len(train_data)}")
    
    val_data_path = f"./data/val.json"
    print(f"Load data from {val_data_path}...")
    with open(val_data_path, 'r') as f:
        val_data = json.load(f)

    print(f"Val data length: {len(val_data)}")
        
    # data_files = {
    #     "train": "./data/train.json",
    #     "val": "./data/val.json"
    # }
    
    # dataset = load_dataset("json", data_files=data_files)
    # print("dataset:", dataset)
    
    tokenizer = AutoTokenizer.from_pretrained(config["model"])
    model = AutoModelForSequenceClassification.from_pretrained(
        config["model"], num_labels=3, id2label=num2label, label2id=label2num)

    preprocess_train_data = preprocess_data(train_data)
    preprocess_val_data = preprocess_data(val_data)
    
    train_dataset = Dataset.from_list(preprocess_train_data)
    val_dataset = Dataset.from_list(preprocess_val_data)

    # device = torch.device(config['device'] if torch.cuda.is_available() else "cpu")
    # print("device:", device)

    tokenized_train_dataset = train_dataset.map(preprocess_function, batched=True)
    tokenized_val_dataset = val_dataset.map(preprocess_function, batched=True)
    print("tokenized_train_dataset:", tokenized_train_dataset)
    print("tokenized_val_dataset:", tokenized_val_dataset)

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    accuracy = evaluate.load("accuracy")
    
    print(f'Start finetuning {config["finetune_model"]}...')

    training_args = TrainingArguments(
        output_dir=output_folder_path,
        learning_rate=config["learning_rate"],
        per_device_train_batch_size=config["batch_size"],
        per_device_eval_batch_size=config["batch_size"],
        num_train_epochs=config["epoch"],
        weight_decay=0.01,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        push_to_hub=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train_dataset,
        eval_dataset=tokenized_val_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    trainer.train()
