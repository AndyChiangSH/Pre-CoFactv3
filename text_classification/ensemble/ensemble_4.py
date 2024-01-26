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
    args = get_argument()
    
    ref_data_path = f"./data/{args['mode']}.json"
    print(f"Load ref_data from {ref_data_path}...")
    with open(ref_data_path, 'r') as f:
        ref_data = json.load(f)
        
    print(f"ref_data length: {len(ref_data)}")
    
    ref_labels = []
    for i in range(len(ref_data)):
        ref_labels.append(label2num[ref_data[i]["label"]])

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

    weight = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
    exponent = [1/4, 1/2, 1, 2, 4]
    
    best_accuracy = 0
    best_w1, best_w2, best_w3 = 0, 0, 0
    best_e1, best_e2, best_e3 = 0, 0, 0
    best_cm = None

    for w1 in tqdm(weight):
        for w2 in weight:
            for w3 in weight:
                for e1 in exponent:
                    for e2 in exponent:
                        for e3 in exponent:
                            ensemble_labels = []
                            for i in range(len(model_1)):
                                probs_1 = model_1[i]
                                probs_2 = model_2[i]
                                probs_3 = model_3[i]
                                
                                new_ps = (w1 * (probs_1[0] ** e1)) + \
                                         (w2 * (probs_2[0] ** e2)) + \
                                         (w3 * (probs_3[0] ** e3))
                                new_pn = (w1 * (probs_1[1] ** e1)) + \
                                         (w2 * (probs_2[1] ** e2)) + \
                                         (w3 * (probs_3[1] ** e3))
                                new_pr = (w1 * (probs_1[2] ** e1)) + \
                                         (w2 * (probs_2[2] ** e2)) + \
                                         (w3 * (probs_3[2] ** e3))
                                
                                new_probs = [new_ps, new_pn, new_pr]
                                max_probs_label = numpy.argmax(new_probs)
                                
                                ensemble_labels.append(max_probs_label)
                                
                            accuracy = accuracy_score(
                                y_true=ref_labels, y_pred=ensemble_labels)
                            cm = confusion_matrix(
                                y_true=ref_labels, y_pred=ensemble_labels)

                            # print(f"ws: {ws}, wn: {wn}, wr: {wr}, accuracy: {accuracy}")
                            
                            if accuracy > best_accuracy:
                                best_accuracy = accuracy
                                best_cm = cm
                                best_w1, best_w2, best_w3 = w1, w2, w3
                                best_e1, best_e2, best_e3 = e1, e2, e3

    print("=============================================================")
    print(f"{args['model_1']}+{args['model_2']}/{args['mode']}")
    print(f"best_w1: {best_w1}, best_w2: {best_w2}, best_w3: {best_w3}")
    print(f"best_e1: {best_e1}, best_e2: {best_e2}, best_e3: {best_e3}")
    print(f"best_accuracy: {best_accuracy}")
    print(f"best_cm: {best_cm.tolist()}")
    print("=============================================================")

    output = {
        "best_w1": best_w1,
        "best_w2": best_w2,
        "best_w3": best_w3,
        "best_e1": best_e1,
        "best_e2": best_e2,
        "best_e3": best_e3,
        "best_accuracy": best_accuracy,
        "best_cm": best_cm.tolist(),
    }
    
    output_folder_path = f"ensemble/ensemble_4/{args['model_1']}+{args['model_2']}+{args['model_3']}"
    if not os.path.exists(output_folder_path):
        os.makedirs(output_folder_path)

    output_path = f"{output_folder_path}/{args['mode']}.json"
    print(f"Save data to {output_path}...")
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)

    cmd = ConfusionMatrixDisplay(
        confusion_matrix=best_cm, display_labels=num2label)
    cmd.plot(cmap="cividis")
    plt.title(
        f"{args['model_1']}+{args['model_2']}+{args['model_3']}/{args['mode']}")
    # plt.show()
    plt.savefig(f"{output_folder_path}/{args['mode']}_cm.jpg")
