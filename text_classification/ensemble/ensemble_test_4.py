import os
import json
import yaml
from tqdm import tqdm
# from datasets import load_metric
import matplotlib.pyplot as plt
import argparse
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix, ConfusionMatrixDisplay
import numpy


label2num = {
    "Support": 0,
    "Neutral": 1,
    "Refute": 2,
}

num2label = ["Support", "Neutral", "Refute"]


def get_argument():
    opt = argparse.ArgumentParser()
    opt.add_argument("--model_1",
                     type=str,
                     help="model_1")
    opt.add_argument("--model_2",
                     type=str,
                     help="model_2")
    opt.add_argument("--model_3",
                     type=str,
                     help="model_3")
    opt.add_argument("--mode",
                     type=str,
                     help="mode")
    args = vars(opt.parse_args())
    
    return args


if __name__ == '__main__':
    print("Start ensembling...")
    
    args = get_argument()
        
    data_path = f"./data/{args['mode']}.json"
    print(f"Load data from {data_path}...")
    with open(data_path, 'r') as f:
        data = json.load(f)
        
    print(f"data length: {len(data)}")
    
    model_1_path = f"ensemble/model/{args['model_1']}/{args['mode']}_prob.json"
    print(f"Load data from {model_1_path}...")
    with open(model_1_path, 'r') as f:
        model_1 = json.load(f)
        
    print(f"model_1 length: {len(model_1)}")

    model_2_path = f"ensemble/model/{args['model_2']}/{args['mode']}_prob.json"
    print(f"Load data from {model_2_path}...")
    with open(model_2_path, 'r') as f:
        model_2 = json.load(f)

    print(f"model_2 length: {len(model_2)}")
    
    model_3_path = f"ensemble/model/{args['model_3']}/{args['mode']}_prob.json"
    print(f"Load data from {model_3_path}...")
    with open(model_3_path, 'r') as f:
        model_3 = json.load(f)

    print(f"model_3 length: {len(model_3)}")
    
    config_path = f"ensemble/ensemble_4/{args['model_1']}+{args['model_2']}+{args['model_3']}/val.json"
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)

    best_w1 = config["best_w1"]
    best_w2 = config["best_w2"]
    best_w3 = config["best_w3"]
    best_e1 = config["best_e1"]
    best_e2 = config["best_e2"]
    best_e3 = config["best_e3"]
    best_accuracy = config["best_accuracy"]
    best_cm = config["best_cm"]
    
    print("=============================================================")
    print(f"{args['model_1']}+{args['model_2']}+{args['model_3']}/{args['mode']}")
    print(f"best_w1: {best_w1}, best_w2: {best_w2}, best_w3: {best_w3}")
    print(f"best_e1: {best_e1}, best_e2: {best_e2}, best_e3: {best_e3}")
    print(f"best_accuracy: {best_accuracy}")
    print(f"best_cm: {best_cm}")
    print("=============================================================")

    ensemble_labels = []
    for i in range(len(model_1)):
        probs_1 = model_1[i]
        probs_2 = model_2[i]
        probs_3 = model_3[i]
        
        new_ps = (best_w1 * (probs_1[0] ** best_e1)) + \
            (best_w2 * (probs_2[0] ** best_e2)) + \
            (best_w3 * (probs_3[0] ** best_e3))
        new_pn = (best_w1 * (probs_1[1] ** best_e1)) + \
            (best_w2 * (probs_2[1] ** best_e2)) + \
            (best_w3 * (probs_3[1] ** best_e3))
        new_pr = (best_w1 * (probs_1[2] ** best_e1)) + \
            (best_w2 * (probs_2[2] ** best_e2)) + \
            (best_w3 * (probs_3[2] ** best_e3))
        
        new_probs = [new_ps, new_pn, new_pr]
        max_probs_label = numpy.argmax(new_probs)
        
        ensemble_labels.append(max_probs_label)

    for i in range(len(data)):
        data[i]["label"] = num2label[ensemble_labels[i]]
    
    output_folder_path = f"ensemble/ensemble_4/{args['model_1']}+{args['model_2']}+{args['model_3']}"
    if not os.path.exists(output_folder_path):
        os.makedirs(output_folder_path)

    output_path = f"{output_folder_path}/{args['mode']}.json"
    print(f"Save data to {output_path}...")
    with open(output_path, 'w') as f:
        json.dump(data, f, indent=2)
