import os
import json
import torch
import yaml
from tqdm import tqdm
from transformers import AutoModelForQuestionAnswering, AutoTokenizer
from transformers import DefaultDataCollator, TrainingArguments, Trainer, Seq2SeqTrainingArguments
from datasets import load_dataset
import argparse
import evaluate
import numpy as np


def preprocess_function(examples):
    # questions = [q.strip() for q in examples["question"]]
    questions = []
    for q in examples["question"]:
        try:
            questions.append(q.strip())
        except:
            questions.append("")
            
    contexts = []
    for c in examples["context"]:
        if len(c) > config["max_len"]:
            c = c[:config["max_len"]]
        contexts.append(c)
        
    inputs = tokenizer(
        questions,
        contexts,
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


def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    
    predictions = tokenizer.batch_decode(predictions)
    labels = tokenizer.batch_decode(labels)
    
    # print("predictions:", predictions)
    # print("labels:", labels)

    return bleu.compute(predictions=predictions, references=labels)


if __name__ == '__main__':
    # input_argument = get_argument()
    # with open(f"./config/{input_argument['config']}", "r") as file:
    #     config = yaml.safe_load(file)
    # print("config:", config)
    
    with open("./generate_answer/config.yaml", "r") as file:
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
    
    bleu = evaluate.load("bleu")
    
    data_files = {
        "train": "generate_answer/data/train_preprocess.json",
        "val": "generate_answer/data/val_preprocess.json"
    }
    dataset = load_dataset("json", data_files=data_files)
    print("dataset:", dataset)
    
    device = torch.device(
        config['device'] if torch.cuda.is_available() else "cpu")
    print("device:", device)
    
    output_folder_path = f"./generate_answer/model/finetune/{config['finetune_model']}"
    if not os.path.exists(output_folder_path):
        os.makedirs(output_folder_path)

    model = AutoModelForQuestionAnswering.from_pretrained(config["model"])
    tokenizer = AutoTokenizer.from_pretrained(config["model"])

    # tokenized_train_data = train_data.map(preprocess_function, batched=True)
    # tokenized_val_data = val_data.map(preprocess_function, batched=True)
    tokenized_dataset = dataset.map(preprocess_function, batched=True)
    
    data_collator = DefaultDataCollator()
    
    training_args = Seq2SeqTrainingArguments(
        output_dir=output_folder_path,
        learning_rate=config["learning_rate"],
        per_device_train_batch_size=config["batch_size"],
        per_device_eval_batch_size=config["batch_size"],
        num_train_epochs=config["epoch"],
        weight_decay=0.01,
        push_to_hub=False,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        # eval_steps=1,
        # auto_find_batch_size=True,
        # predict_with_generate=True,
        # load_best_model_at_end=True,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["val"],
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    trainer.train()
    
    trainer.save_model(output_folder_path)
