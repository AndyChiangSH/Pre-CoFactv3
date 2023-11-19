import os
import json
import torch
import yaml
from tqdm import tqdm
from transformers import AutoModelForQuestionAnswering, AutoTokenizer
from transformers import DefaultDataCollator, TrainingArguments, Trainer
from datasets import load_dataset
import argparse


def get_argument():
    opt = argparse.ArgumentParser()
    opt.add_argument("--config",
                     type=str,
                     help="config path")
    config = vars(opt.parse_args())
    return config


def preprocess_function(examples):
    # questions = [q.strip() for q in examples["question"]]
    questions = []
    for q in examples["question"]:
        try:
            questions.append(q.strip())
        except:
            questions.append("")
        
    inputs = tokenizer(
        questions,
        examples["context"],
        max_length=512,
        truncation="only_second",
        return_offsets_mapping=True,
        padding="max_length",
    )

    offset_mapping = inputs.pop("offset_mapping")
    answers = examples["answers"]
    start_positions = []
    end_positions = []

    for i, offset in enumerate(offset_mapping):
        answer = answers[i]
        start_char = answer["answer_start"][0]
        end_char = answer["answer_start"][0] + len(answer["text"][0])
        sequence_ids = inputs.sequence_ids(i)

        # Find the start and end of the context
        idx = 0
        while sequence_ids[idx] != 1:
            idx += 1
        context_start = idx
        while sequence_ids[idx] == 1:
            idx += 1
        context_end = idx - 1

        # If the answer is not fully inside the context, label it (0, 0)
        if offset[context_start][0] > end_char or offset[context_end][1] < start_char:
            start_positions.append(0)
            end_positions.append(0)
        else:
            # Otherwise it's the start and end token positions
            idx = context_start
            while idx <= context_end and offset[idx][0] <= start_char:
                idx += 1
            start_positions.append(idx - 1)

            idx = context_end
            while idx >= context_start and offset[idx][1] >= end_char:
                idx -= 1
            end_positions.append(idx + 1)

    inputs["start_positions"] = start_positions
    inputs["end_positions"] = end_positions
    
    return inputs


if __name__ == '__main__':
    # input_argument = get_argument()
    # with open(f"./config/{input_argument['config']}", "r") as file:
    #     config = yaml.safe_load(file)
    # print("config:", config)
    
    with open("./data/generate_answers/config.yaml", "r") as file:
        config = yaml.safe_load(file)
    
    # load_data_path = f"./data/preprocess_train.json"
    # print(f"Load data from {load_data_path}...")
    # train_data = load_dataset("json", data_files=load_data_path)
    # # with open(load_data_path, 'r') as f:
    # #     train_data = json.load(f)

    # print(f"Data length: {len(train_data)}")
    
    # load_data_path = f"./data/preprocess_val.json"
    # print(f"Load data from {load_data_path}...")
    # val_data = load_dataset("json", data_files=load_data_path)
    # # with open(load_data_path, 'r') as f:
    # #     val_data = json.load(f)

    # print(f"Data length: {len(val_data)}")
    
    data_files = {
        "train": "./data/preprocess_train.json",
        "val": "./data/preprocess_val.json"
    }
    dataset = load_dataset("json", data_files=data_files)
    print("dataset:", dataset)
    
    device = torch.device(
        config['device'] if torch.cuda.is_available() else "cpu")
    print("device:", device)
    
    output_folder_path = f"./data/generate_answers/{config['finetune_model']}"
    if not os.path.exists(output_folder_path):
        os.makedirs(output_folder_path)

    model = AutoModelForQuestionAnswering.from_pretrained(config["model"])
    tokenizer = AutoTokenizer.from_pretrained(config["model"])

    # tokenized_train_data = train_data.map(preprocess_function, batched=True)
    # tokenized_val_data = val_data.map(preprocess_function, batched=True)
    tokenized_dataset = dataset.map(preprocess_function, batched=True)
    
    data_collator = DefaultDataCollator()
    
    training_args = TrainingArguments(
        output_dir=output_folder_path,
        evaluation_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=config["batch_size"],
        per_device_eval_batch_size=config["batch_size"],
        num_train_epochs=config["epoch"],
        weight_decay=0.01,
        push_to_hub=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["val"],
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    trainer.train()
    
    trainer.save_model(output_folder_path)
